from dataclasses import dataclass

from typing import Any
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import torch
import json

HF_TOKEN = "key here"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

import os
os.environ["RAY_DISABLE_DASHBOARD"] = "1"

@dataclass
class GPUPoolContext:
    model: Any
    device: Any
    tokenizer: Any
    hf_tokenizer: Any

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

@dataclass
class Prompt:
    prompt: str
    question_key: str
    template_type: str
    label: list[str]
    incorrect_answer: str
    centering_vector: dict[int, np.ndarray] = None
    sycophancy_vector: dict[int, np.ndarray] = None
    persona_vector: dict[int, np.ndarray] = None
    model_answer: str = ""
    sycophanous: int = -1  # -1 means not yet labeled

#from kien
def canonical_template(record):
    tmpl = record.get('metadata', {}).get('prompt_template', '{question}')
    if "I think the answer is {incorrect_answer}" in tmpl:
        return 'assert_incorrect'
    if "I think the answer is {correct_answer}" in tmpl:
        return 'assert_correct'
    if "I don't think the answer is {correct_answer}" in tmpl:
        return 'doubt_correct'
    return 'neutral'

def question_key(record):
    base = record.get('base', {})
    if base.get('question'):
        return base['question'].strip()
    prompt = record.get('prompt', [])
    human_segments = [turn['content'] for turn in prompt if turn.get('type') == 'human']
    return '\n\n'.join(human_segments).strip()

def load_syc_dataset(filename: str, data_dir: str) -> list[Prompt]:
    prompts: list[Prompt] = []
    with open(data_dir / filename, "r") as f:
        for line in f:
            tmp = json.loads(line)
            p = Prompt(
                prompt=tmp['prompt'][-1]['content'],
                question_key=question_key(tmp),
                template_type=canonical_template(tmp),
                label=(tmp['base']['answer'] if 'answer' in tmp['base'] else [])
                    + [tmp['base']['correct_answer']],
                incorrect_answer=tmp['base']['incorrect_answer']
            )
            prompts.append(p)
    return prompts

def tokenize_prompts(prompts: list[Prompt], system_prompt: str = None) -> dict[str, torch.Tensor]:
    texts = []
    for p in prompts:
        # Build messages list from the raw prompt content
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': p.prompt})
        
        # Apply chat template to get properly formatted text
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    return encodings

def load_hf_ctx(model_name, hf_token):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda:0")  # Ray maps the assigned GPU to cuda:0

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    
    ctx = GPUPoolContext(
        model=model,
        device=device,
        tokenizer=tokenize_prompts,
        hf_tokenizer=tokenizer
    )
    
    return ctx

import ray, itertools

def ensure_ray():
    if not ray.is_initialized():
        # Silence the "metrics exporter agent" spam in notebook/terminal output
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            _system_config={
                "metrics_report_interval_ms": 0,
            },
        )
@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, load_fn, load_kwargs):
        # load_fn runs inside this GPU process
        self.ctx = load_fn(**load_kwargs)

    def run(self, fn, x, kwargs):
        return fn(self.ctx, x, **kwargs)

    def run_many(self, fn, xs, kwargs):
        return [fn(self.ctx, x, **kwargs) for x in xs]

class GPUModelPool:
    def __init__(self, load_fn, load_kwargs, num_workers=None):
        ensure_ray()
        if num_workers is None:
            num_workers = int(ray.available_resources().get("GPU", 0))
        if num_workers <= 0:
            raise RuntimeError("No GPUs visible to Ray.")
        self.workers = [ModelWorker.remote(load_fn, load_kwargs) for _ in range(num_workers)]
        self._rr = itertools.cycle(self.workers)

    def call(self, fn, x, **kwargs):
        """Run one request on one GPU (round-robin)."""
        w = next(self._rr)
        return ray.get(w.run.remote(fn, x, kwargs))

    def map(self, fn, xs, chunk_size=8, use_tqdm=False, **kwargs):
        """Run many requests across GPUs. Returns results in order."""
        # chunk for less overhead
        chunks = [xs[i:i+chunk_size] for i in range(0, len(xs), chunk_size)]
        refs = []
        for c in chunks:
            w = next(self._rr)
            refs.append(w.run_many.remote(fn, c, kwargs))
        
        if use_tqdm:
            # Wait for results with a progress bar
            chunk_outs = []
            # We need to maintain order, so we can't just use ray.wait in a loop easily for ordered results
            # unless we map refs back to their index.
            # Simpler approach: just iterate over refs with tqdm, but ray.get on a single ref blocks.
            # Better approach for progress: use a list of not-done refs and ray.wait
            
            # However, to keep it simple and ordered:
            # We can just wait for all of them, but that defeats the purpose of a progress bar (it would jump 0->100).
            # So we will use an unordered wait loop to update the bar, but store results in a dict by index.
            
            ref_to_index = {ref: i for i, ref in enumerate(refs)}
            results_by_index = [None] * len(refs)
            pending = list(refs)
            
            pbar = tqdm(total=len(xs))
            
            while pending:
                done, pending = ray.wait(pending, num_returns=1)
                for ref in done:
                    idx = ref_to_index[ref]
                    res = ray.get(ref)
                    results_by_index[idx] = res
                    pbar.update(len(res)) # update by chunk size (or actual returned size)
            
            pbar.close()
            chunk_outs = results_by_index
        else:
            chunk_outs = ray.get(refs)
            
        # flatten in order
        return [y for chunk in chunk_outs for y in chunk]

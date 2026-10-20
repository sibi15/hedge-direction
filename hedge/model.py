"""The model wrapper: templating, batching, generation, activation capture.

Model-agnostic: any Hugging Face causal LM with a chat template. Things it handles that
a plain `from_pretrained` does not:

* **transformers version differences.** `torch_dtype` became `dtype`, and older
  versions can ignore an unknown kwarg silently, so the dtype is enforced after loading.
* **Left padding**, so position `-1` is always the final real prompt token.
* **Different layouts**: the decoder blocks are found by probing the usual attribute
  paths (Llama/Qwen/Mistral/Gemma/Phi, GPT-2, GPT-NeoX, multimodal wrappers).
* **Models without a chat template**: `Settings.chat_template` (a .jinja path or an
  inline template); `Settings.system_prompt` adds a system turn.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hooks import applied
from .settings import HedgeError

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
BLOCK_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers",
               # multimodal wrappers around a text decoder (e.g. Gemma 3 >= 4B)
               "model.language_model.layers", "language_model.model.layers")


def resolve_device(want: str) -> torch.device:
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        print("[model] cuda requested but not available -> falling back to cpu")
        want = "cpu"
    return torch.device(want)


def resolve_dtype(want: str, device: torch.device) -> torch.dtype:
    if want == "auto":
        # float16 is the safe GPU default (pre-Ampere GPUs have no bfloat16)
        return torch.float16 if device.type == "cuda" else torch.float32
    dt = DTYPES[want]
    if dt is torch.bfloat16 and device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
        print("[model] bfloat16 requested on a pre-Ampere GPU -> using float16")
        return torch.float16
    return dt


def load_kwargs(s) -> dict:
    """from_pretrained arguments shared by the tokenizer and the model.

    `cache_dir` is only passed when Settings.cache_dir is set. HF_HOME must NOT be passed
    as cache_dir: the library keeps models in $HF_HOME/hub, and reads HF_HOME itself --
    passing the variable makes it look one folder too high and find nothing.
    """
    return dict(cache_dir=s.cache_dir, revision=s.revision,
                trust_remote_code=s.trust_remote_code, local_files_only=s.offline)


def _tqdm(iterable, desc: str, progress: bool):
    if not progress:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, unit="batch", leave=False)
    except ImportError:
        return iterable


class ChatModel:
    def __init__(self, s, verbose: bool = True):
        """`s` is a hedge.Settings; only its model-loading fields are used."""
        self.s = s
        self.model_id = s.model
        self.device = resolve_device(s.device)
        self.dtype = resolve_dtype(s.dtype, self.device)
        if s.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"   # no network attempts at all
        load_kw = load_kwargs(s)

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_kw)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if s.chat_template:
            p = Path(s.chat_template).expanduser()
            self.tokenizer.chat_template = (p.read_text(encoding="utf-8") if p.exists()
                                            else s.chat_template)
        if not getattr(self.tokenizer, "chat_template", None):
            raise HedgeError(f"{self.model_id} has no chat template. Either the model is not fully "
                             f"downloaded (run `python run.py --download --model {self.model_id}`), "
                             f"or it is a base model: set Settings.chat_template")

        model_kw = dict(load_kw)
        if s.attn_implementation:
            model_kw["attn_implementation"] = s.attn_implementation
        try:  # transformers >= 4.56
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=self.dtype, **model_kw)
        except TypeError:  # older transformers
            model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=self.dtype,
                                                         **model_kw)
        if model.dtype != self.dtype:   # an ignored dtype kwarg would leave float32
            model = model.to(self.dtype)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)

        self.blocks = self._find_blocks()
        self.embed = self.model.get_input_embeddings()
        self.n_layers = len(self.blocks)
        self.d_model = int(self.embed.weight.shape[1])

        if verbose:
            print(f"[model] {self.model_id} on {self.device} ({str(self.dtype).split('.')[-1]}), "
                  f"{self.n_layers} layers, d_model={self.d_model}, loaded in {time.time() - t0:.1f}s")
            if self.device.type == "cuda":
                free, total = torch.cuda.mem_get_info()
                print(f"[model] VRAM: {free / 2**30:.2f} GB free / {total / 2**30:.2f} GB total")

    def _find_blocks(self):
        for path in BLOCK_PATHS:
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if len(obj):
                    return obj
            except (AttributeError, TypeError):
                continue
        raise HedgeError(f"could not locate the decoder blocks on {self.model_id}; "
                         f"known layouts: {', '.join(BLOCK_PATHS)}")

    # ---------------------------------------------------------------- prompting

    def messages(self, prompt: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.s.system_prompt}] if self.s.system_prompt else []
        return msgs + [{"role": "user", "content": prompt}]

    def build_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(self.messages(prompt), tokenize=False,
                                                  add_generation_prompt=True)

    def encode(self, prompts: Sequence[str]):
        enc = self.tokenizer([self.build_prompt(p) for p in prompts], padding=True,
                             add_special_tokens=False, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    # --------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], hooks=(), max_new_tokens: int | None = None,
                 desc: str = "generating", progress: bool = True) -> list[str]:
        """Greedy generation, optionally under an intervention. One string per prompt.

        `do_sample=False` is load-bearing: generation is deterministic, so the seed only
        affects the split and the random controls, never the model's answers, and a
        steered answer can be attributed to the intervention rather than sampling noise.
        """
        prompts = list(prompts)
        if not prompts:
            return []
        max_new_tokens = int(max_new_tokens or self.s.max_new_tokens)
        bs = int(self.s.batch_size)
        out: list[str] = []
        with applied(hooks):
            for i in _tqdm(range(0, len(prompts), bs), desc, progress):
                enc = self.encode(prompts[i:i + bs])
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          num_beams=1, pad_token_id=self.tokenizer.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                out += [t.strip() for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]
                del gen
        return out

    # -------------------------------------------------------------- activations

    @torch.no_grad()
    def last_token_acts(self, prompts: Sequence[str], progress: bool = True) -> torch.Tensor:
        """Residual-stream activations at the final prompt token.

        Returns ``(n_prompts, n_layers, d_model)`` in float32 on CPU, where index ``l`` is
        the stream *entering* block ``l`` (index 0 = embedding output). The last of the
        `n_layers + 1` hidden states has the final norm applied, so it is dropped.
        """
        prompts = list(prompts)
        if not prompts:
            return torch.zeros((0, self.n_layers, self.d_model), dtype=torch.float32)
        bs = int(self.s.act_batch_size)
        chunks = []
        for i in _tqdm(range(0, len(prompts), bs), "activations", progress):
            enc = self.encode(prompts[i:i + bs])
            hs = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            if len(hs) != self.n_layers + 1:
                raise RuntimeError(f"expected {self.n_layers + 1} hidden states, got {len(hs)}; "
                                   f"the block layout probe and the model disagree")
            chunks.append(torch.stack([h[:, -1, :].float() for h in hs[:-1]], dim=1).cpu())
        return torch.cat(chunks)

    # ------------------------------------------------------------------ damage

    @torch.no_grad()
    def next_token_kl(self, prompts: Sequence[str], hooks=()) -> float:
        """Mean KL(clean || intervened) of the next-token distribution, in nats.

        Averaged over every real (non-pad) position of the templated prompts: how much an
        intervention distorts the model where it should change nothing.
        """
        prompts = list(prompts)
        if not prompts:
            return float("nan")
        bs = int(self.s.kl_batch_size)
        total, count = 0.0, 0
        for i in range(0, len(prompts), bs):
            enc = self.encode(prompts[i:i + bs])
            mask = enc["attention_mask"].bool()
            clean = torch.log_softmax(self.model(**enc, use_cache=False).logits.float(), dim=-1)
            with applied(hooks):
                steered = torch.log_softmax(self.model(**enc, use_cache=False).logits.float(),
                                            dim=-1)
            kl = (clean.exp() * (clean - steered)).sum(-1)       # (batch, seq)
            total += float(kl[mask].sum())
            count += int(mask.sum())
            del clean, steered, kl
        return total / max(1, count)


def load_model(s, verbose: bool = True) -> ChatModel:
    return ChatModel(s, verbose=verbose)

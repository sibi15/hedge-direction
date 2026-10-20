"""Forward hooks for residual-stream interventions.

Conventions (kept identical to `hedge_poc.ipynb` so results stay comparable):

* ``hidden_states[l]`` = the residual stream *entering* decoder block ``l``,
  so index 0 is the embedding output and index ``n_layers-1`` is the input to the
  last block. ``ChatModel.last_token_acts`` returns exactly that stack.
* **Activation addition** for layer ``l`` hooks the *output* of block ``l-1``
  (for ``l=0``, the embedding output). ``h <- h + coeff * v``.
* **Directional ablation** removes the component along a unit vector:
  ``h <- h - (h . û) û``.

Two safety details carried over from a previous project where they were real bugs:

1. Normalisation is done in **float32**. In float16, ``1e-8`` underflows to exactly
   ``0.0``, so ``d / (d.norm() + 1e-8)`` is ``0/0 = NaN`` for a zero direction.
   Doing the norm in fp32 makes a zero vector a provable no-op instead.
2. Optional position masks are **length-checked**. A mask sized for the prompt,
   multiplied against a single-token decode activation, broadcasts silently to
   ``(1, L, d)`` instead of raising -- corrupting the run. We refuse instead.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor


def _act(x):
    return x[0] if isinstance(x, tuple) else x


def _repack(original, new_act):
    return (new_act, *original[1:]) if isinstance(original, tuple) else new_act


def unit_vector(direction: Tensor, like: Tensor | None = None) -> Tensor:
    """Unit-normalise in float32 (see module docstring), cast back if `like` given."""
    d = direction.detach().to(torch.float32)
    d = d / (d.norm() + 1e-8)
    if like is not None:
        return d.to(device=like.device, dtype=like.dtype)
    return d


def _resolve_mask(positions_mask, seq_len: int, device) -> Tensor | None:
    if positions_mask is None:
        return None
    m = positions_mask(seq_len) if callable(positions_mask) else positions_mask
    m = torch.as_tensor(m, device=device)
    if m.numel() != seq_len:
        raise ValueError(
            f"positions_mask has {m.numel()} entries but the activation sequence length "
            f"is {seq_len}; refusing to broadcast silently"
        )
    return m.view(1, -1, 1)


def activation_addition_hook(vector: Tensor, coeff: float, positions_mask=None) -> Callable:
    """``h <- h + coeff * v`` at every position (forward-hook form)."""
    def hook_fn(module, args, output):
        h = _act(output)
        v = vector.detach().to(device=h.device, dtype=h.dtype)
        add = (coeff * v).view(1, 1, -1)
        m = _resolve_mask(positions_mask, h.shape[1], h.device)
        if m is not None:
            add = add * m
        return _repack(output, h + add)

    return hook_fn


def _ablate(h: Tensor, direction: Tensor, positions_mask=None) -> Tensor:
    orig_dtype = h.dtype
    h32 = h.to(torch.float32)
    u = unit_vector(direction).to(device=h.device)
    proj = (h32 @ u).unsqueeze(-1) * u
    m = _resolve_mask(positions_mask, h.shape[1], h.device)
    if m is not None:
        proj = proj * m
    return (h32 - proj).to(orig_dtype)


def directional_ablation_hook(direction: Tensor, positions_mask=None) -> Callable:
    """``h <- h - (h . û) û`` at every position (forward-hook form)."""
    def hook_fn(module, args, output):
        return _repack(output, _ablate(_act(output), direction, positions_mask))

    return hook_fn


def directional_ablation_pre_hook(direction: Tensor, positions_mask=None) -> Callable:
    """Same ablation on a block's *input* (forward-pre-hook form, ``with_kwargs=True``).

    A pre-hook is called as ``fn(module, args, kwargs)`` -- there is no `output` -- and
    HF decoder layers may receive `hidden_states` positionally or as a keyword.
    """
    def hook_fn(module, args, kwargs):
        if args:
            return (_ablate(args[0], direction, positions_mask), *args[1:]), kwargs
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = _ablate(kwargs["hidden_states"], direction, positions_mask)
            return args, kwargs
        raise RuntimeError("block_inputs ablation: the block received no hidden_states input")

    return hook_fn


@contextlib.contextmanager
def applied(hooks: Iterable[tuple]):
    """Register `(module, fn, kind)` hooks (`kind` in {'forward', 'pre'}) and remove them."""
    handles = []
    try:
        for module, fn, kind in hooks:
            if kind == "pre":
                handles.append(module.register_forward_pre_hook(fn, with_kwargs=True))
            else:
                handles.append(module.register_forward_hook(fn))
        yield
    finally:
        for h in handles:
            h.remove()


# ------------------------------------------------------------------ hook builders

def steering_hooks(model, layer: int, vector: Tensor, coeff: float) -> list[tuple]:
    """Activation addition for `layer`, at the residual stream entering that block."""
    if layer < 0 or layer > model.n_layers - 1:
        raise ValueError(f"layer {layer} out of range for {model.n_layers} layers")
    module = model.embed if layer == 0 else model.blocks[layer - 1]
    return [(module, activation_addition_hook(vector, coeff), "forward")]


def ablation_hooks(model, direction: Tensor, sites: str = "notebook",
                   layers="all") -> list[tuple]:
    """Directional ablation hooks.

    sites='notebook'  -- embed output + every decoder block output (what the PoC did)
    sites='block_inputs' -- resid_pre of every block (the Arditi et al. convention)

    `layers` is 'all', or a list of block indices (0 = first block). Note that with
    sites='notebook' the embedding output is only touched when layer 0 is included.
    """
    if layers == "all":
        idx = list(range(model.n_layers))
    else:
        idx = [int(l) for l in layers]
        for l in idx:
            if l < 0 or l > model.n_layers - 1:
                raise ValueError(f"ablation layer {l} out of range for {model.n_layers} layers")

    if sites == "block_inputs":
        return [(model.blocks[l], directional_ablation_pre_hook(direction), "pre") for l in idx]

    hooks = [(model.blocks[l], directional_ablation_hook(direction), "forward") for l in idx]
    if 0 in idx:
        hooks.insert(0, (model.embed, directional_ablation_hook(direction), "forward"))
    return hooks


def all_layer_ablation_hooks(model, direction: Tensor, sites: str = "notebook",
                             layers="all") -> list[tuple]:
    """Alias kept for readability at call sites: the same direction everywhere."""
    return ablation_hooks(model, direction, sites=sites, layers=layers)

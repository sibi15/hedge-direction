"""Direction maths and the choices between steps: which layers, which (layer, coeff)."""
from __future__ import annotations

import hashlib

import torch

from .settings import HedgeError


def layer_grid(s, n_layers: int) -> list[int]:
    """Layers to sweep: `s.layers`, else every `layer_stride`-th layer in `layer_band`.

    Fractions of depth rather than layer numbers, so any model size works unchanged.
    """
    if s.layers:
        layers = sorted({int(l) for l in s.layers})
    else:
        lo, hi = s.layer_band
        layers = list(range(int(round(lo * n_layers)), int(round(hi * n_layers)), s.layer_stride))
    layers = [l for l in layers if 0 <= l < n_layers]
    if not layers:
        raise HedgeError(f"no layers to sweep for a {n_layers}-layer model; check layers / layer_band")
    if s.quick and len(layers) > 1:
        layers = [layers[len(layers) // 2]]
    return layers


def choose_final(s, dose_records: list[dict]) -> dict | None:
    """Pick the (layer, coeff) for the final test from the *validation* dose records.

    Highest specificity (direction hedge rate minus random hedge rate) among doses that
    keep the factual prompts mostly direct; ties go to the smaller coefficient. If every
    dose breaks the factual prompts the filter is relaxed, and that is recorded.
    """
    if not dose_records:
        return None

    def bad_frac(r):
        f = r.get("factual")
        return (f["n"] - f["direct"]["k"]) / f["n"] if f and f.get("n") else 0.0

    ok = [r for r in dose_records if bad_frac(r) <= s.max_factual_nondirect]
    best = max(ok or dose_records, key=lambda r: (r["score"], -float(r["coeff"])))
    return {"layer": int(best["layer"]), "coeff": float(best["coeff"]),
            "val_score": best["score"], "factual_filter_relaxed": not ok}


def task_seed(*parts) -> int:
    """Deterministic seed from any tuple of identifiers (hashed, so tasks never collide)."""
    blob = repr(tuple(str(p) for p in parts)).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") & ((1 << 63) - 1)


def generator(*parts) -> torch.Generator:
    return torch.Generator().manual_seed(task_seed(*parts))


def diff_in_means(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Mean(pos) - Mean(neg) per layer -> (n_layers, d_model)."""
    if pos_acts.shape[0] == 0 or neg_acts.shape[0] == 0:
        raise HedgeError("need at least one positive and one negative example")
    return pos_acts.mean(0) - neg_acts.mean(0)


def shuffled_direction(acts_pos: torch.Tensor, acts_neg: torch.Tensor, layer: int,
                       gen: torch.Generator) -> torch.Tensor:
    """Label-shuffle null: pool both classes, deal them into two piles of the original
    sizes at random, and take the same difference of means.

    Built exactly like the real direction from the same activations -- only the grouping
    is meaningless. It is much shorter than the real one (the piles nearly cancel), so
    callers rescale it before activation addition.
    """
    pool = torch.cat([acts_pos[:, layer], acts_neg[:, layer]]).to(torch.float32)
    perm = torch.randperm(pool.shape[0], generator=gen)
    n_pos = acts_pos.shape[0]
    return pool[perm[:n_pos]].mean(0) - pool[perm[n_pos:]].mean(0)


def unit_rows(dirs: torch.Tensor) -> torch.Tensor:
    """Row-wise unit normalisation in float32 (an fp16 epsilon underflows to 0)."""
    d = dirs.to(torch.float32)
    return d / (d.norm(dim=-1, keepdim=True) + 1e-8)


def add_vector(s, direction: torch.Tensor, layer: int, resid_norm: torch.Tensor | None = None,
               ref_norm: float | None = None) -> torch.Tensor:
    """The vector activation addition adds (before `coeff`), per `s.add_scale`.

    raw   -- the direction itself, or rescaled to `ref_norm` (puts a shuffled direction on
             the real direction's scale)
    unit  -- unit length
    resid -- unit length x the mean residual-stream norm at `layer`
    """
    d = direction.detach().to(torch.float32)
    u = d / (d.norm() + 1e-8)
    if s.add_scale == "unit":
        return u
    if s.add_scale == "resid":
        if resid_norm is None:
            raise HedgeError("add_scale='resid' needs the residual norms from the extract step")
        return u * float(resid_norm[layer])
    return u * float(ref_norm) if ref_norm is not None else d


def random_vector_like(vec: torch.Tensor, gen: torch.Generator, scale: str = "match") -> torch.Tensor:
    """A random direction, 'match'-ing the norm of `vec` or of 'unit' length."""
    rv = torch.randn(vec.shape, generator=gen)
    rv = rv / (rv.norm() + 1e-12)
    return rv * (vec.norm() if scale == "match" else 1.0)


def project(acts: torch.Tensor, units: torch.Tensor, layer: int) -> torch.Tensor:
    """Scalar projection of every example onto the layer's unit direction."""
    return acts[:, layer] @ units[layer]


def auroc_by_layer(acts_pos, acts_neg, units, layers=None) -> dict[int, float]:
    from .stats import auroc
    layers = list(range(1, units.shape[0])) if layers is None else list(layers)
    return {l: auroc(project(acts_pos, units, l), project(acts_neg, units, l)) for l in layers}

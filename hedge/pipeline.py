"""The whole experiment: `run(Settings(...))`.

1. baseline  answer every question; the keyword judge labels each answer
2. extract   split hedged/direct prompts into train / val / test and build one
             difference-in-means direction per layer from the train prompts
3. sweep     (val) per layer: add the direction to direct prompts, ablate it on hedged ones
4. dose      (val) top layers x coefficients vs same-norm random vectors -> picks the final
             (layer, coefficient)
5. final     (test, once) that configuration vs random vectors and label-shuffled directions
6. refusal   (optional, `harmful` set) refusal of harmful requests: baseline vs hedge
             direction ablated vs random direction ablated; with `harmless` too, also the
             cosine with the refusal direction
7. report    report.md, results.json, final_outputs.json (for human rating)

Also here: `pilot()` checks which question endings a model answers without "As an AI..."
disclaimers, `inspect()` shows side-by-side answers for your own prompts, and
`download()` fetches a model ahead of time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from . import data
from .experiment import (add_vector, auroc_by_layer, choose_final, diff_in_means, generator,
                         layer_grid, project, random_vector_like, shuffled_direction, unit_rows)
from .hooks import ablation_hooks, steering_hooks, unit_vector
from .judging import CLASSES, Judge, is_refusal, rate, rate_str, slim
from .language import breakdown, language_of, other_language
from .model import load_model
from .report import build_report
from .settings import HedgeError, Settings
from .stats import auroc, mcnemar_exact, paired_diff_ci, permutation_p, wilson


# ------------------------------------------------------------------------------ helpers

def _settings(settings: Settings | None, changes: dict) -> Settings:
    s = settings or Settings()
    return dataclasses.replace(s, **changes) if changes else s


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _paired_p(k_changed: int) -> float:
    """Exact McNemar for 'did the intervention change the outcome'.

    Every prompt in a held-out pool shares its baseline label (the direct pool is all
    'direct', the hedged pool all 'hedge'), so the only discordant pairs are
    baseline-no -> intervened-yes: b = k_changed, c = 0.
    """
    return mcnemar_exact(int(k_changed), 0)


def _fingerprint(s: Settings) -> str:
    """Hash of everything a result depends on: settings, prompt contents and this code."""
    h = hashlib.sha1()
    d = s.to_dict()
    for k in ("name", "out_dir", "save_completions", "resume"):
        d.pop(k, None)
    h.update(json.dumps(d, sort_keys=True, default=str).encode())
    for spec in (s.questions, s.factual, s.neutral, s.harmful, s.harmless):
        for prompt in data.read_prompts(spec):
            h.update(prompt.encode("utf-8") + b"\n")
        h.update(b"\0")
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


class _Run:
    """What every step needs: settings, output folder, model, judge, a log, checkpoints."""

    def __init__(self, s: Settings, out: Path, model, judge: Judge, ckpt: Path | None = None):
        self.s, self.out, self.model, self.judge = s, out, model, judge
        self.factual = data.read_prompts(s.factual)
        self.neutral = data.read_prompts(s.neutral) if s.neutral else self.factual
        self.ckpt = ckpt           # None = no checkpointing
        self.prefix = ""           # set after the baseline: a hash of the labels in use
        self._buffers: list[list[dict]] = []

    def unit(self, key: str, fn):
        """Run `fn` once per settings fingerprint: a rerun after a crash or a timeout loads
        the saved result (and replays its completions) instead of recomputing it."""
        if self.ckpt is None:
            return fn()
        path = self.ckpt / f"{self.prefix}{key}.pt"
        if path.exists():
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:          # a half-written file from a killed job: redo the unit
                blob = None
            if blob is not None:
                self._emit(blob["rows"])
                self.log(f"[resume] reusing {key}")
                return blob["result"]
        self._buffers.append([])
        try:
            result = fn()
        finally:
            rows = self._buffers.pop()
        self._emit(rows)
        tmp = path.with_suffix(".tmp")
        torch.save({"result": result, "rows": rows}, tmp)
        tmp.replace(path)
        return result

    def _emit(self, rows: list[dict]) -> None:
        if self._buffers:              # inside another unit: it saves them with its result
            self._buffers[-1].extend(rows)
        elif rows and self.s.save_completions:
            with open(self.out / "completions.jsonl", "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.out / "run.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def gen(self, prompts, hooks=(), desc="generating", progress=True,
            max_new_tokens=None) -> list[str]:
        return self.model.generate(prompts, hooks=hooks, desc=desc, progress=progress,
                                   max_new_tokens=max_new_tokens)

    def steer(self, layer, vector, coeff):
        return steering_hooks(self.model, layer, vector, coeff)

    def ablate(self, direction):
        return ablation_hooks(self.model, direction, sites=self.s.ablation_sites,
                              layers=self.s.ablation_layers)

    def record(self, step, condition, split, prompts, outputs, **extra) -> list[dict]:
        """Label outputs, append them to completions.jsonl, and return the rows."""
        rows = [{"step": step, "condition": condition, "split": split, **extra,
                 "prompt": q, "output": o, "label": self.judge.label(o),
                 "language": language_of(o)}
                for q, o in zip(prompts, outputs)]
        self._emit(rows)
        return rows


# -------------------------------------------------------------------------------- run

def run(settings: Settings | None = None, **changes) -> Path:
    """Run the whole experiment; returns the results folder.

        run()                                             # defaults
        run(model="meta-llama/Llama-3.2-3B-Instruct")     # change any Settings field
        run(Settings(...), quick=True)
    """
    s = _settings(settings, changes).effective()
    warnings = s.validate()
    out = s.run_dir()
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("completions.jsonl", "run.log"):   # rebuilt (resumed units replay theirs)
        (out / stale).unlink(missing_ok=True)
    write_json(out / "settings.json", s.to_dict())
    ckpt = None
    if s.resume:
        ckpt = out / "checkpoints" / _fingerprint(s)
        for old in (out / "checkpoints").glob("*"):      # other settings/code: no longer valid
            if old != ckpt:
                shutil.rmtree(old, ignore_errors=True)
        ckpt.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    r = _Run(s, out, load_model(s), Judge(s), ckpt)
    r.log(f"[run] {s.model} -> {out}")
    for w in warnings:
        r.log(f"[run] WARNING: {w}")

    questions, answers, labels = _baseline(r)
    ext = _extract(r, questions, answers, labels)
    sweep = _sweep(r, ext)
    dose = _dose(r, ext, sweep["top_layers"])
    final = _final(r, ext, dose["chosen"])
    refusal_test = _refusal_test(r, ext, final)
    refusal = _refusal(r, ext)

    results = {"model": s.model, "directions": ext["report"], "sweep": sweep, "dose": dose,
               "final": final, "refusal_test": refusal_test, "refusal": refusal,
               "seconds": round(time.time() - t0)}
    write_json(out / "results.json", results)
    build_report(s, out, results)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)   # finished: nothing to resume
    r.log(f"[run] done in {(time.time() - t0) / 60:.1f} min -> {out / 'report.md'}")
    return out


# ----------------------------------------------------------------------------- steps

def _baseline(r: _Run):
    questions = data.load_questions(r.s)
    r.log(f"[baseline] answering {len(questions)} questions")

    def answer():
        out = r.gen(questions, desc="baseline")
        r.record("baseline", "none", "all", questions, out)
        return out

    answers = r.unit("baseline", answer)
    auto = [r.judge.label(a) for a in answers]
    write_json(r.out / "baseline.json",
               [{"q": q, "label": l, "answer": a} for q, l, a in zip(questions, auto, answers)])
    data.write_labels_to_check(r.out / "labels_to_check.csv", questions, auto, answers)
    labels, agreement = data.apply_hand_labels(r.s, r.out, questions, auto)
    r.log(f"[baseline] judge labels: {dict(Counter(auto))}" +
          (f" | hand labels: {agreement['n_agree']}/{agreement['n_hand']} agree with the judge"
           if agreement else ""))
    r.agreement = agreement
    # everything after this depends on the labels (hand labels may change between runs)
    r.prefix = hashlib.sha1("\n".join(labels).encode()).hexdigest()[:8] + "-"
    return questions, answers, labels


def _extract(r: _Run, questions, answers, labels) -> dict:
    s = r.s
    pools = data.class_pools(questions, labels)
    n_h, n_d = len(pools["hedge"]), len(pools["direct"])
    r.log(f"[extract] {n_h} hedge / {n_d} direct / {len(pools['disclaimer'])} disclaimer")
    if s.min_per_class and min(n_h, n_d) < s.min_per_class:
        raise HedgeError(f"only {n_h} hedged / {n_d} direct prompts; need >= {s.min_per_class} "
                         f"of each. Add questions, check the question endings with pilot(), or "
                         f"fix labels in {r.out / 'labels_to_check.csv'}")

    split = {c: data.split3(pools[c], s.val_frac, s.test_frac, s.seed,
                            s.min_val_per_class, s.min_test_per_class) for c in ("hedge", "direct")}
    sp = {f"{c}_{part}": split[c][i] for c in split for i, part in enumerate(("train", "val", "test"))}
    sp["disclaimer"] = pools["disclaimer"]
    write_json(r.out / "splits.json", sp)
    r.log(f"[extract] hedge/direct: train {len(sp['hedge_train'])}/{len(sp['direct_train'])} | "
          f"val {len(sp['hedge_val'])}/{len(sp['direct_val'])} | "
          f"test {len(sp['hedge_test'])}/{len(sp['direct_test'])}")

    m = r.model
    acts = r.unit("activations",
                  lambda: {k: m.last_token_acts(v) for k, v in sp.items() if k != "disclaimer"})
    dirs = diff_in_means(acts["hedge_train"], acts["direct_train"])     # (n_layers, d_model)
    units = unit_rows(dirs)
    # mean residual-stream norm per layer (last prompt token, train): the add_scale='resid' unit
    resid_norm = torch.cat([acts["hedge_train"], acts["direct_train"]]).norm(dim=-1).mean(0)
    torch.save({"dirs": dirs, "units": units, "resid_norm": resid_norm, "model": s.model},
               r.out / "directions.pt")

    auroc_val = auroc_by_layer(acts["hedge_val"], acts["direct_val"], units)
    acts_x = (r.unit("activations_disclaimer", lambda: m.last_token_acts(pools["disclaimer"]))
              if len(pools["disclaimer"]) >= 3 else None)
    auroc_x = auroc_by_layer(acts_x, acts["direct_val"], units) if acts_x is not None else None
    best = max(auroc_val, key=auroc_val.get)
    r.log(f"[extract] validation AUROC peaks at L{best} = {auroc_val[best]:.2f}")
    report = {
        "n_layers": m.n_layers, "d_model": m.d_model,
        "baseline_counts": {c: labels.count(c) for c in CLASSES},
        "hand_label_agreement": r.agreement,
        **{f"n_{part}": {"hedge": len(sp[f"hedge_{part}"]), "direct": len(sp[f"direct_{part}"])}
           for part in ("train", "val", "test")},
        "auroc_hedge_vs_direct": auroc_val, "auroc_disclaimer_vs_direct": auroc_x,
        "direction_norm_by_layer": {l: float(dirs[l].norm()) for l in range(m.n_layers)},
    }
    return {"splits": sp, "acts": acts, "dirs": dirs, "units": units,
            "resid_norm": resid_norm, "report": report}


def _sweep(r: _Run, ext: dict) -> dict:
    """(val) Per layer: add on direct prompts, ablate on hedged prompts."""
    s, sp, records = r.s, ext["splits"], []

    def one(layer):
        v = add_vector(s, ext["dirs"][layer], layer, ext["resid_norm"])
        steer = r.steer(layer, v, s.sweep_coeff)
        added = r.gen(sp["direct_val"], steer, f"L{layer} add")
        ablated = r.gen(sp["hedge_val"], r.ablate(ext["units"][layer]), f"L{layer} ablate")
        facts = r.gen(r.factual, steer, f"L{layer} facts")
        r.record("sweep", "add_hedge", "val", sp["direct_val"], added, layer=layer, coeff=s.sweep_coeff)
        r.record("sweep", "ablate_hedge", "val", sp["hedge_val"], ablated, layer=layer)
        s_add, s_abl = r.judge.summarize(added), r.judge.summarize(ablated)
        rec = {"layer": layer, "auroc": ext["report"]["auroc_hedge_vs_direct"].get(layer),
               "add_on_direct": slim(s_add), "ablate_on_hedged": slim(s_abl),
               "add_on_factual": slim(r.judge.summarize(facts)) if facts else None,
               "mcnemar_add": _paired_p(s_add["hedge"]["k"]),
               "mcnemar_ablate": _paired_p(s_abl["direct"]["k"]),
               "score": rate(s_add, "hedge") + rate(s_abl, "direct"),
               "examples_added": added[:2], "examples_ablated": ablated[:2]}
        return rec

    for layer in layer_grid(s, r.model.n_layers):
        rec = r.unit(f"sweep_L{layer}", lambda: one(layer))
        records.append(rec)
        r.log(f"[sweep] L{layer:2d} +dir on direct: hedge {rate_str(rec['add_on_direct'], 'hedge')} | "
              f"-dir on hedged: direct {rate_str(rec['ablate_on_hedged'], 'direct')}")
    top = [x["layer"] for x in sorted(records, key=lambda x: -x["score"])][: s.top_k_layers]
    r.log(f"[sweep] top layers: {top}")
    return {"records": records, "top_layers": top}


def _dose(r: _Run, ext: dict, top_layers: list[int]) -> dict:
    """(val) Coefficient sweep at the top layers vs same-norm random vectors."""
    s, sp = r.s, ext["splits"]
    prompts, records, rand_abl = sp["direct_val"], [], []

    def one(layer, coeff):
        v = add_vector(s, ext["dirs"][layer], layer, ext["resid_norm"])
        steer = r.steer(layer, v, coeff)
        out_dir = r.gen(prompts, steer, f"L{layer} x{coeff}")
        out_fact = r.gen(r.factual, steer, f"L{layer} x{coeff} facts")
        kl_dir = r.model.next_token_kl(r.neutral, steer)
        out_rand, kl_rand = [], []
        for j in range(s.n_random):
            rv = random_vector_like(v, generator(s.seed, "dose_random", layer, coeff, j),
                                    s.random_scale)
            rsteer = r.steer(layer, rv, coeff)
            out_rand += r.gen(prompts, rsteer, progress=False)
            kl_rand.append(r.model.next_token_kl(r.neutral, rsteer))
        r.record("dose", "add_hedge", "val", prompts, out_dir, layer=layer, coeff=coeff)
        r.record("dose", "add_random", "val", prompts * max(1, s.n_random), out_rand,
                 layer=layer, coeff=coeff)
        s_dir, s_rand = r.judge.summarize(out_dir), r.judge.summarize(out_rand)
        return {
            "layer": layer, "coeff": float(coeff),
            "direction": slim(s_dir), "random": slim(s_rand),
            "factual": slim(r.judge.summarize(out_fact)) if out_fact else None,
            "kl_direction": kl_dir, "kl_random": sum(kl_rand) / len(kl_rand) if kl_rand else None,
            "score": rate(s_dir, "hedge") - (rate(s_rand, "hedge") if s_rand["n"] else 0.0),
            "examples_direction": out_dir[:2],
            "factual_outputs": dict(zip(r.factual, out_fact)),
        }

    def random_ablation(layer):
        # negative control for the ablation: a random unit direction
        ru = random_vector_like(ext["dirs"][layer], generator(s.seed, "ablate_random", layer), "unit")
        outs = r.gen(sp["hedge_val"], r.ablate(ru), f"L{layer} random ablate")
        r.record("dose", "ablate_random", "val", sp["hedge_val"], outs, layer=layer)
        return {"layer": layer, "ablate_random_on_hedged": slim(r.judge.summarize(outs))}

    for layer in top_layers:
        for coeff in s.coeffs:
            rec = r.unit(f"dose_L{layer}_x{coeff}", lambda: one(layer, coeff))
            records.append(rec)
            r.log(f"[dose] L{layer:2d} x{coeff:<5} hedge: dir {rate_str(rec['direction'], 'hedge')} "
                  f"vs random {rate_str(rec['random'], 'hedge')} | KL {rec['kl_direction']:.3f}")
        rand_abl.append(r.unit(f"random_ablation_L{layer}", lambda: random_ablation(layer)))
    chosen = choose_final(s, records)
    if chosen:
        r.log(f"[dose] chosen for the final test: L{chosen['layer']} x{chosen['coeff']}")
    return {"records": records, "random_ablation": rand_abl, "chosen": chosen}


def _final(r: _Run, ext: dict, chosen: dict | None) -> dict:
    """(test, once) The chosen configuration vs random vectors and label shuffles."""
    s, sp = r.s, ext["splits"]
    layer = s.final_layer if s.final_layer is not None else (chosen or {}).get("layer")
    coeff = s.final_coeff if s.final_coeff is not None else (chosen or {}).get("coeff")
    if layer is None or coeff is None:
        raise HedgeError("no final (layer, coeff): the dose step chose nothing; set "
                         "final_layer and final_coeff")
    source = ("validation" if s.final_layer is None and s.final_coeff is None
              else "settings" if s.final_layer is not None and s.final_coeff is not None
              else "settings+validation")
    layer, coeff = int(layer), float(coeff)
    d_test, h_test = sp["direct_test"], sp["hedge_test"]
    dirs, units = ext["dirs"], ext["units"]
    r.log(f"[final] L{layer} x{coeff} ({source}) on {len(h_test)} hedged / {len(d_test)} direct "
          f"test prompts, + {s.n_shuffles} label shuffles")

    v = add_vector(s, dirs[layer], layer, ext["resid_norm"])
    steer, ablate = r.steer(layer, v, coeff), r.ablate(units[layer])

    def main():
        outputs = {
            "add": r.record("final", "add_hedge", "test", d_test, r.gen(d_test, steer, "final add")),
            "ablate": r.record("final", "ablate_hedge", "test", h_test,
                               r.gen(h_test, ablate, "final ablate")),
            "factual": r.record("final", "add_hedge_factual", "factual", r.factual,
                                r.gen(r.factual, steer, "final facts")),
            "random_add": [], "random_ablate": [],
        }
        for j in range(max(1, s.n_random)):
            rv = random_vector_like(v, generator(s.seed, "final_random_add", layer, coeff, j),
                                    s.random_scale)
            ru = random_vector_like(dirs[layer], generator(s.seed, "final_random_ablate", layer, j),
                                    "unit")
            outputs["random_add"] += r.record("final", "add_random", "test", d_test,
                                              r.gen(d_test, r.steer(layer, rv, coeff), progress=False))
            outputs["random_ablate"] += r.record("final", "ablate_random", "test", h_test,
                                                 r.gen(h_test, r.ablate(ru), progress=False))
        rv0 = random_vector_like(v, generator(s.seed, "final_random_add", layer, coeff, 0),
                                 s.random_scale)
        ru0 = random_vector_like(dirs[layer], generator(s.seed, "final_random_ablate", layer, 0), "unit")
        kl = {"add": r.model.next_token_kl(r.neutral, steer),
              "add_random": r.model.next_token_kl(r.neutral, r.steer(layer, rv0, coeff)),
              "ablate": r.model.next_token_kl(r.neutral, ablate),
              "ablate_random": r.model.next_token_kl(r.neutral, r.ablate(ru0))}
        return outputs, kl

    outputs, kl = r.unit(f"final_L{layer}_x{coeff}", main)
    write_json(r.out / "final_outputs.json", outputs)

    summ = {k: r.judge.summarize([x["output"] for x in rows]) for k, rows in outputs.items()}

    # label-shuffle null: same procedure on the same test prompts, random grouping
    null_add, null_abl, null_cos = [], [], []
    real = dirs[layer].to(torch.float32)
    def shuffle(i):
        fake = shuffled_direction(ext["acts"]["hedge_train"], ext["acts"]["direct_train"], layer,
                                  generator(s.seed, "label_shuffle", layer, i))
        fv = add_vector(s, fake, layer, ext["resid_norm"], ref_norm=float(real.norm()))
        a = r.gen(d_test, r.steer(layer, fv, coeff), progress=False)
        b = r.gen(h_test, r.ablate(unit_vector(fake)), progress=False)
        r.record("final", "add_shuffle", "test", d_test, a, shuffle=i)
        r.record("final", "ablate_shuffle", "test", h_test, b, shuffle=i)
        return (rate(r.judge.summarize(a), "hedge"), rate(r.judge.summarize(b), "direct"),
                float(torch.nn.functional.cosine_similarity(fake, real, dim=0)))

    for i in range(s.n_shuffles):
        na, nb, nc = r.unit(f"shuffle_L{layer}_x{coeff}_{i}", lambda: shuffle(i))
        null_add.append(na)
        null_abl.append(nb)
        null_cos.append(nc)
        if (i + 1) % 10 == 0 or i + 1 == s.n_shuffles:
            r.log(f"[final] {i + 1}/{s.n_shuffles} shuffles done")

    test_acts_h, test_acts_d = ext["acts"]["hedge_test"], ext["acts"]["direct_test"]
    real_add, real_abl = rate(summ["add"], "hedge"), rate(summ["ablate"], "direct")
    res = {
        "layer": layer, "coeff": coeff, "source": source,
        "n_hedge_test": len(h_test), "n_direct_test": len(d_test),
        "auroc_test": auroc(project(test_acts_h, units, layer), project(test_acts_d, units, layer)),
        "add_on_direct": slim(summ["add"]), "random_add_on_direct": slim(summ["random_add"]),
        "ablate_on_hedged": slim(summ["ablate"]), "random_ablate_on_hedged": slim(summ["random_ablate"]),
        "factual": slim(summ["factual"]) if r.factual else None,
        "kl": kl,
        "mcnemar_add": _paired_p(summ["add"]["hedge"]["k"]),
        "mcnemar_ablate": _paired_p(summ["ablate"]["direct"]["k"]),
        "n_shuffles": s.n_shuffles,
        "null_add_hedge_rate": null_add, "null_ablate_direct_rate": null_abl,
        "null_cos_with_real": null_cos,
        "real_add_hedge_rate": real_add, "real_ablate_direct_rate": real_abl,
        "p_perm_add": permutation_p(real_add, null_add),
        "p_perm_ablate": permutation_p(real_abl, null_abl),
    }
    r.log(f"[final] +dir on direct: hedge {rate_str(summ['add'], 'hedge')} vs random "
          f"{rate_str(summ['random_add'], 'hedge')} (p vs shuffles {res['p_perm_add']:.3f}) | "
          f"-dir on hedged: direct {rate_str(summ['ablate'], 'direct')} vs random "
          f"{rate_str(summ['random_ablate'], 'direct')} (p vs shuffles {res['p_perm_ablate']:.3f})")
    return res


def _refusal_test(r: _Run, ext: dict, final: dict) -> dict | None:
    if not r.s.harmful:
        return None
    return r.unit(f"refusal_test_L{final['layer']}", lambda: _refusal_test_run(r, ext, final))


def _refusal_test_run(r: _Run, ext: dict, final: dict) -> dict:
    """(optional) Does removing the hedge direction change refusal of harmful requests?

    Harmful prompts (the paper evaluates on JailbreakBench) are answered three ways:
    baseline, hedge direction ablated (the final test's layer and ablation), and a random
    direction ablated. Refusal is scored with the paper's substring list. If hedging and
    refusal are separate mechanisms, ablating the hedge direction should leave refusal
    about where it is, like the random control.

    Completions are capped at `refusal_max_new_tokens` and written ONLY to
    refusal_outputs.json (not completions.jsonl, not the report): any request the model
    stops refusing produces harmful text.
    """
    s = r.s
    if not s.harmful:
        return None
    # spread evenly through the file: JailbreakBench is ordered by category (10 x 10), so
    # this keeps every category represented
    prompts = data.spread(list(dict.fromkeys(data.read_prompts(s.harmful))), s.refusal_max_prompts)
    if not prompts:
        raise HedgeError(f"no harmful prompts found in {s.harmful!r}")
    layer, n = final["layer"], len(prompts)
    rand = random_vector_like(ext["dirs"][layer], generator(s.seed, "refusal_random_ablate", layer),
                              "unit")
    conditions = {"baseline": (), "ablate_hedge": r.ablate(ext["units"][layer]),
                  "ablate_random": r.ablate(rand)}
    r.log(f"[refusal test] {n} harmful prompts x {len(conditions)} conditions "
          f"(hedge direction from L{layer}, {s.refusal_max_new_tokens} new tokens)")

    outputs, refused = {}, {}
    for cond, hooks in conditions.items():
        outputs[cond] = r.gen(prompts, hooks, f"refusal {cond}",
                              max_new_tokens=s.refusal_max_new_tokens)
        refused[cond] = [is_refusal(o, s.refusal_patterns) for o in outputs[cond]]
    write_json(r.out / "refusal_outputs.json",
               {c: [{"prompt": p, "output": o, "refused": f, "language": language_of(o)}
                    for p, o, f in zip(prompts, outputs[c], refused[c])] for c in conditions})

    base = refused["baseline"]
    res = {"layer": layer, "n_prompts": n, "max_new_tokens": s.refusal_max_new_tokens,
           "source": str(s.harmful), "conditions": {},
           # "As an AI ..." counts as a refusal here and as a disclaimer for the hedge judge
           "baseline_refusals_judged_disclaimer": sum(
               f and r.judge.label(o) == "disclaimer" for o, f in zip(outputs["baseline"], base))}
    for cond in conditions:
        k = sum(refused[cond])
        lo, hi = wilson(k, n)
        entry = {"refused": k, "n": n, "rate": k / n, "ci95": [lo, hi],
                 "other_language": sum(other_language(o, s.answer_language) for o in outputs[cond]),
                 "languages": breakdown(outputs[cond], s.answer_language)}
        if cond != "baseline":
            stopped = sum(x and not y for x, y in zip(base, refused[cond]))
            started = sum(y and not x for x, y in zip(base, refused[cond]))
            d, dlo, dhi = paired_diff_ci(base, refused[cond])
            entry.update({"stopped_refusing": stopped, "started_refusing": started,
                          "mcnemar_p": mcnemar_exact(stopped, started),
                          "diff": d, "diff_ci95": [dlo, dhi]})
        res["conditions"][cond] = entry
    c = res["conditions"]
    r.log(f"[refusal test] refusal rate: baseline {c['baseline']['refused']}/{n} | hedge ablated "
          f"{c['ablate_hedge']['refused']}/{n} (change {c['ablate_hedge']['diff']:+.2f}, 95% CI "
          f"{c['ablate_hedge']['diff_ci95'][0]:+.2f} to {c['ablate_hedge']['diff_ci95'][1]:+.2f}) | "
          f"random ablated {c['ablate_random']['refused']}/{n}")
    return res


def _refusal(r: _Run, ext: dict) -> dict | None:
    """(optional, needs harmful + harmless) cos(hedging direction, refusal direction)."""
    if not (r.s.harmful and r.s.harmless):
        return None
    return r.unit("refusal_cosine", lambda: _refusal_run(r, ext))


def _refusal_run(r: _Run, ext: dict) -> dict:
    s = r.s
    harmful, harmless = data.read_prompts(s.harmful)[:64], data.read_prompts(s.harmless)[:64]
    acts_harmless = r.model.last_token_acts(harmless)
    refusal = diff_in_means(r.model.last_token_acts(harmful), acts_harmless)
    perm = torch.randperm(len(harmless), generator=generator(s.seed, "refusal_split"))
    half = len(harmless) // 2
    control = acts_harmless[perm[:half]].mean(0) - acts_harmless[perm[half:]].mean(0)
    cos = torch.nn.functional.cosine_similarity
    dirs, n = ext["dirs"], ext["dirs"].shape[0]
    per_layer = {l: round(float(cos(dirs[l], refusal[l], dim=0)), 3) for l in range(1, n)}
    best = max(per_layer, key=lambda l: abs(per_layer[l]))
    r.log(f"[refusal] cos(hedge, refusal) peaks at L{best}: {per_layer[best]}")
    return {"n_harmful": len(harmful), "n_harmless": len(harmless),
            "per_layer_cos_hedge_refusal": per_layer, "best_layer": best,
            "best_layer_cos": per_layer[best],
            "cos_hedge_vs_split_half_control": {l: round(float(cos(dirs[l], control[l], dim=0)), 3)
                                                for l in range(1, n)},
            "random_vector_scale": round(1 / dirs.shape[-1] ** 0.5, 3)}


# --------------------------------------------------------------------- extra tools

def pilot(settings: Settings | None = None, pilot_file: str = "data/pilot_endings.json",
          **changes) -> str:
    """Which question ending does this model answer without 'As an AI...' disclaimers?

    Answers every prompt in `pilot_file` (the same statements under different endings)
    and prints, per ending, how contested and clear-cut items were labelled. A good
    ending has few disclaimers, contested items hedging and clear-cut items direct.
    Writes <run>/pilot.md with the table and every answer. Returns the table.
    """
    s = _settings(settings, changes)
    items = json.loads(data._resolve(pilot_file).read_text(encoding="utf-8"))
    model, judge = load_model(s), Judge(s)
    answers = model.generate([it["statement"] for it in items], desc="pilot")
    labels = [judge.label(a) for a in answers]

    counts: dict[tuple, Counter] = defaultdict(Counter)
    for it, lab, a in zip(items, labels, answers):
        key = (it["group"], it["variant"], it["expected"])
        counts[key][lab] += 1
        counts[key]["_other"] += other_language(a, s.answer_language)
    lines = [f"Pilot: question endings for {s.model} (direct/hedge/disclaimer counts)", ""]
    for group in dict.fromkeys(it["group"] for it in items):
        lines += [f"== {group} ==",
                  f"{'ending':20s} {'contested':>15s} {'clear-cut':>15s} {'disclaimers':>12s}  "
                  f"usable  not {s.answer_language}"]
        for v in dict.fromkeys(it["variant"] for it in items if it["group"] == group):
            con, clr = counts[(group, v, "contested")], counts[(group, v, "clear-cut")]
            n = sum(con.values()) + sum(clr.values())
            cell = lambda c: f"{c['direct']}/{c['hedge']}/{c['disclaimer']}"
            lines.append(f"{v:20s} {cell(con):>15s} {cell(clr):>15s} "
                         f"{(con['disclaimer'] + clr['disclaimer']) / max(1, n):12.0%}  "
                         f"{con['hedge'] + clr['direct']}/{n}"
                         f"{con['_other'] + clr['_other']:>8d}")
        lines.append("")
    table = "\n".join(lines)
    print(table)
    out = s.run_dir()
    out.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"- [{it['variant']}] [{lab}, {language_of(a)}] **{it['statement']}**\n\n  {a}\n"
                     for it, lab, a in zip(items, labels, answers))
    (out / "pilot.md").write_text(f"```\n{table}\n```\n\n{body}", encoding="utf-8")
    print(f"[pilot] every answer: {out / 'pilot.md'}")
    return table


def inspect(prompts, settings: Settings | None = None,
            conditions=("baseline", "ablate_hedge", "ablate_random"),
            layer: int | None = None, coeff: float | None = None,
            max_new_tokens: int = 150, **changes) -> Path:
    """Side-by-side answers for your own prompts, using the direction of a finished run.

    `prompts`: a list of strings or a prompt file. Conditions: baseline, ablate_hedge,
    ablate_random, add_hedge, add_random. Layer and coefficient default to the run's
    final choice. Writes <run>/inspect-<time>.md and returns its path.
    """
    s = _settings(settings, changes)
    qs = list(prompts) if isinstance(prompts, (list, tuple)) else data.read_prompts(prompts)
    out = s.run_dir()
    needs_dir = any(c != "baseline" for c in conditions)
    if needs_dir:
        if not (out / "directions.pt").exists():
            raise HedgeError(f"{out} has no directions.pt; run() this model first")
        blob = torch.load(out / "directions.pt", map_location="cpu", weights_only=False)
        final = (json.loads((out / "results.json").read_text(encoding="utf-8")).get("final") or {}
                 if (out / "results.json").exists() else {})
        layer = int(layer if layer is not None else final.get("layer"))
        coeff = float(coeff if coeff is not None else final.get("coeff"))
        v = add_vector(s, blob["dirs"][layer], layer, blob.get("resid_norm"))

    model, judge = load_model(s), Judge(s)
    hooks = {"baseline": lambda: ()}
    if needs_dir:
        hooks.update({
            "ablate_hedge": lambda: ablation_hooks(model, blob["units"][layer], s.ablation_sites,
                                                   s.ablation_layers),
            "ablate_random": lambda: ablation_hooks(
                model, random_vector_like(blob["dirs"][layer],
                                          generator(s.seed, "inspect_ablate_random", layer), "unit"),
                s.ablation_sites, s.ablation_layers),
            "add_hedge": lambda: steering_hooks(model, layer, v, coeff),
            "add_random": lambda: steering_hooks(
                model, layer, random_vector_like(v, generator(s.seed, "inspect_add_random", layer),
                                                 s.random_scale), coeff),
        })
    unknown = set(conditions) - set(hooks)
    if unknown:
        raise HedgeError(f"unknown inspect conditions {sorted(unknown)}; choose from {sorted(hooks)}")
    results = {c: model.generate(qs, hooks=hooks[c](), max_new_tokens=max_new_tokens, desc=c)
               for c in conditions}

    head = f"# Inspect: {s.model}" + (f", layer {layer}, coeff {coeff}" if needs_dir else "")
    md = [head, ""]
    for i, q in enumerate(qs):
        md.append(f"## {q}")
        for c, outs in results.items():
            md.append(f"**{c}** _[{judge.label(outs[i])}]_\n\n{outs[i].strip()}\n")
        md.append("---\n")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"inspect-{time.strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"[inspect] {len(qs)} prompts x {len(results)} conditions -> {path}")
    return path


def download(settings: Settings | None = None, **changes) -> None:
    """Fetch the model into the Hugging Face cache (on a cluster: do this on a login node)."""
    s = _settings(settings, changes)
    from huggingface_hub import snapshot_download
    # skip other frameworks' formats and the duplicate "original" weights some repos ship
    # (Llama: original/consolidated.00.pth, Mistral: consolidated.safetensors) -- transformers
    # only loads the model-*.safetensors files
    path = snapshot_download(s.model, cache_dir=s.cache_dir, revision=s.revision,
                             ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.gguf",
                                              "original/*", "consolidated*"])
    print(f"[download] {s.model} -> {path}")

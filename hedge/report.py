"""report.md, written from the results of `run()`.

The caveats are generated from the actual numbers, so a tiny test set is called out in
words instead of printing a rate to three decimals.
"""
from __future__ import annotations

import time
from pathlib import Path

from .judging import CLASSES, rate_str


def _tbl(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _f(x, nd=2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _p(p) -> str:
    return "n/a" if p is None else ("< 1e-4" if float(p) < 1e-4 else f"{float(p):.4f}")


def _ol(summary) -> int:
    return (summary or {}).get("other_language", 0)


def _langs(entry) -> str:
    """' (chinese 18, french 2)' from a summary's language breakdown, or ''."""
    langs = (entry or {}).get("languages") or {}
    return " (" + ", ".join(f"{k} {v}" for k, v in langs.items()) + ")" if langs else ""


def _bad(fact) -> str:
    return f"{fact['n'] - fact['direct']['k']}/{fact['n']}" if fact else "n/a"


def build_report(s, out: Path, res: dict) -> Path:
    d, sweep, dose, final = res["directions"], res["sweep"], res["dose"], res["final"]
    records, top = sweep["records"], sweep["top_layers"]
    L: list[str] = []
    A = L.append

    A(f"# Hedging direction -- {s.model}")
    A("")
    A(f"Run `{out.name}` · seed `{s.seed}` · {time.strftime('%Y-%m-%d %H:%M')} · "
      f"settings in `settings.json`")
    A("")
    A("Difference-in-means direction for *hedging* (the \"it depends / both sides have a point\" "
      "register) in the residual stream, following Arditi et al. (2024): built by correlation, "
      "tested by causation (activation addition on direct answers, directional ablation on hedged "
      "answers), and checked for specificity against random vectors and label-shuffled directions.")
    A("")

    # ------------------------------------------------------------------ setup
    ag = d.get("hand_label_agreement")
    A("## Setup")
    A("")
    A(_tbl([
        ["model", s.model],
        ["layers / d_model", f"{d['n_layers']} / {d['d_model']}"],
        ["labels (keyword judge, after hand labels)",
         ", ".join(f"{c}: {d['baseline_counts'].get(c, 0)}" for c in CLASSES)],
        ["hand-label agreement", f"{ag['n_agree']}/{ag['n_hand']} ({ag['rate']:.2f})" if ag
         else "none (keyword judge only)"],
        ["train / val / test (hedge / direct)",
         " · ".join(f"{d[f'n_{p}']['hedge']} / {d[f'n_{p}']['direct']}" for p in ("train", "val", "test"))],
        ["ablation sites / layers", f"{s.ablation_sites} / {s.ablation_layers}"],
        ["addition scale", f"{s.add_scale} (coeff {s.sweep_coeff} in the sweep)"],
    ], ["", ""]))
    A("")

    # ------------------------------------------------------------ correlation
    au = {int(k): v for k, v in d["auroc_hedge_vs_direct"].items()}
    best = max(au, key=au.get)
    A("## Correlation (validation split)")
    A("")
    A(f"AUROC of the projection onto each layer's direction, hedge vs direct (0.5 = chance). "
      f"Best **L{best} = {au[best]:.2f}**.")
    A("")
    A(_tbl([[f"L{l}", _f(a)] for l, a in sorted(au.items())], ["layer", "AUROC"]))
    A("")
    dis = d.get("auroc_disclaimer_vs_direct")
    if dis:
        bd = max(dis, key=dis.get)
        A(f"Disclaimer vs direct peaks at L{bd} = {dis[bd]:.2f}: the 'As an AI, I have no opinions' "
          f"register is a closely related signal.")
        A("")

    # ------------------------------------------------------------------ sweep
    A("## Causation: layer sweep (validation split)")
    A("")
    A("Exploratory: these numbers chose the layer, so they are optimistic by construction. "
      "`+dir` = validation *direct* prompts that hedge once the direction is added; `-dir` = "
      "validation *hedged* prompts that become direct (not disclaimers) once it is ablated.")
    A("")
    A(_tbl([[f"**L{x['layer']}**" if x["layer"] in top else f"L{x['layer']}", _f(x["auroc"]),
             rate_str(x["add_on_direct"], "hedge"), rate_str(x["ablate_on_hedged"], "direct"),
             x["ablate_on_hedged"]["degenerate"],
             f"{_ol(x['add_on_direct'])} / {_ol(x['ablate_on_hedged'])}",
             _bad(x["add_on_factual"]), _f(x["score"])]
            for x in records],
           ["layer", "AUROC", "+dir -> hedge", "-dir -> direct", "degenerate",
            "not English (+dir / -dir)", "facts non-direct", "score"]))
    A("")
    A(f"Top layers: **{', '.join(f'L{l}' for l in top)}**.")
    A("")

    # ------------------------------------------------------------------- dose
    chosen = dose.get("chosen") or {}
    A("## Specificity and dose-response (validation split)")
    A("")
    A("Direction vs same-norm random vectors at the same coefficient. `delta` = difference in "
      "hedging rate. `facts non-direct` or `KL` (mean next-token KL from the clean model on "
      "neutral prompts, nats) climbing with the coefficient means damage, not steering.")
    A("")
    A(_tbl([[("**L%d**" if (x["layer"] == chosen.get("layer") and x["coeff"] == chosen.get("coeff"))
              else "L%d") % x["layer"], _f(x["coeff"]), rate_str(x["direction"], "hedge"),
             rate_str(x["random"], "hedge"), _f(x["score"]), _ol(x["direction"]),
             _bad(x["factual"]), _f(x["kl_direction"], 3), _f(x["kl_random"], 3)]
            for x in dose["records"]],
           ["layer", "coeff", "add: hedge", "random: hedge", "delta", "not English",
            "facts non-direct", "KL dir", "KL random"]))
    A("")
    if chosen:
        A(f"Chosen for the final test (bold): **L{chosen['layer']} x{chosen['coeff']}**, the highest "
          f"delta among doses with at most {s.max_factual_nondirect:.0%} factual prompts off 'direct'"
          + (" (none met that bar, so the filter was relaxed)." if chosen["factual_filter_relaxed"]
             else "."))
        A("")
    for x in dose["random_ablation"]:
        real = next((y for y in records if y["layer"] == x["layer"]), None)
        A(f"- Ablation control L{x['layer']}: hedge direction "
          f"{rate_str(real['ablate_on_hedged'], 'direct') if real else 'n/a'} vs random direction "
          f"{rate_str(x['ablate_random_on_hedged'], 'direct')} hedged prompts turned direct")
    A("")

    # ------------------------------------------------------------------ final
    src = {"validation": "chosen on the validation split", "settings": "fixed in the settings",
           "settings+validation": "partly fixed in the settings"}.get(final["source"], final["source"])
    n_sh = final["n_shuffles"]
    sh = lambda xs: f"{sum(xs) / len(xs):.2f} / {max(xs):.2f}" if xs else "n/a"
    A("## Final test (held-out, confirmatory)")
    A("")
    A(f"**L{final['layer']} x{final['coeff']}** ({src}), run once on the test split "
      f"({final['n_hedge_test']} hedged / {final['n_direct_test']} direct prompts), which played no "
      f"part in building the direction or choosing the configuration. Test AUROC at this layer: "
      f"**{_f(final['auroc_test'])}**.")
    A("")
    kl = final["kl"]
    A(_tbl([
        ["add on direct -> hedge", rate_str(final["add_on_direct"], "hedge"),
         rate_str(final["random_add_on_direct"], "hedge"), sh(final["null_add_hedge_rate"]),
         _p(final["p_perm_add"] if n_sh else None), _p(final["mcnemar_add"]),
         _f(kl["add"], 3), _f(kl["add_random"], 3)],
        ["ablate on hedged -> direct", rate_str(final["ablate_on_hedged"], "direct"),
         rate_str(final["random_ablate_on_hedged"], "direct"), sh(final["null_ablate_direct_rate"]),
         _p(final["p_perm_ablate"] if n_sh else None), _p(final["mcnemar_ablate"]),
         _f(kl["ablate"], 3), _f(kl["ablate_random"], 3)],
    ], ["test", "hedge direction", "random vector", "label shuffles (mean / max)", "p vs shuffles",
        "p McNemar", "KL dir", "KL random"]))
    A("")
    A(f"`p vs shuffles`: one-sided permutation p-value against {n_sh} directions built the same way "
      f"from randomly regrouped training prompts and rescaled to the real direction's size "
      f"(smallest possible value {1 / (n_sh + 1):.3f}). `p McNemar` only says the intervention "
      f"changed outputs; `p vs shuffles` says the *hedging* grouping is what made it work. "
      f"Factual prompts under the added direction: {_bad(final['factual'])} not direct. Answers "
      f"not in English (the judge's patterns are English, so these count as 'direct'): added "
      f"{_ol(final['add_on_direct'])}/{final['add_on_direct']['n']}{_langs(final['add_on_direct'])}, ablated "
      f"{_ol(final['ablate_on_hedged'])}/{final['ablate_on_hedged']['n']}{_langs(final['ablate_on_hedged'])}, random ablated "
      f"{_ol(final['random_ablate_on_hedged'])}/{final['random_ablate_on_hedged']['n']}{_langs(final['random_ablate_on_hedged'])}. Every test "
      f"output is in `final_outputs.json` for human rating.")
    A("")

    # ----------------------------------------------------------- refusal test
    rt = res.get("refusal_test")
    if rt:
        c, n = rt["conditions"], rt["n_prompts"]
        pct = lambda x: f"{x * 100:+.0f}"
        A("## Refusal test: is the hedge direction part of refusal?")
        A("")
        A(f"{n} harmful requests (`{rt['source']}`) answered with no intervention, with the hedge "
          f"direction ablated (L{rt['layer']}, as in the final test) and with a random direction "
          f"ablated. Refusal is scored with the substring list of Arditi et al. (2024) on the first "
          f"{rt['max_new_tokens']} tokens. If hedging and refusal are separate mechanisms, removing "
          f"the hedge direction should leave refusal about where it is, like the random control.")
        A("")
        rows = [["baseline", f"{c['baseline']['refused']}/{n}",
                 f"{c['baseline']['ci95'][0]:.2f}-{c['baseline']['ci95'][1]:.2f}", "", "", "",
                 f"{c['baseline'].get('other_language', 0)}/{n}{_langs(c['baseline'])}"]]
        for cond, label in (("ablate_hedge", "hedge direction ablated"),
                            ("ablate_random", "random direction ablated")):
            e = c[cond]
            rows.append([label, f"{e['refused']}/{n}", f"{e['ci95'][0]:.2f}-{e['ci95'][1]:.2f}",
                         f"{pct(e['diff'])} pts ({pct(e['diff_ci95'][0])} to {pct(e['diff_ci95'][1])})",
                         f"{e['stopped_refusing']} / {e['started_refusing']}", _p(e["mcnemar_p"]),
                         f"{e.get('other_language', 0)}/{n}{_langs(e)}"])
        A(_tbl(rows, ["condition", "refused", "95% CI", "change vs baseline (95% CI)",
                      "stopped / started refusing", "p McNemar", "not English"]))
        A("")
        h = c["ablate_hedge"]
        A(f"Read the interval, not only the p-value: a non-significant change shows hedging and "
          f"refusal are separate only if the interval is narrow. Here, removing the hedge "
          f"direction changed the refusal rate by between {pct(h['diff_ci95'][0])} and "
          f"{pct(h['diff_ci95'][1])} percentage points. {rt['baseline_refusals_judged_disclaimer']} "
          f"of the {c['baseline']['refused']} baseline refusals also count as 'disclaimers' for the "
          f"hedge judge (\"As an AI...\" is in both lists). Refusal phrases are matched in English "
          f"and Chinese, because Qwen switches to Chinese under ablation and keeps refusing there "
          f"(`not English` counts those answers). The completions are in "
          f"`refusal_outputs.json` only -- they can contain harmful text; do not share them.")
        A("")

    # ---------------------------------------------------------------- refusal
    ref = res.get("refusal")
    if ref:
        ctrl = ref["cos_hedge_vs_split_half_control"].get(ref["best_layer"])
        A("## Refusal-direction comparison")
        A("")
        A(f"cos(hedging, refusal) peaks at **L{ref['best_layer']} = {ref['best_layer_cos']}** "
          f"(random pair ~{ref['random_vector_scale']}; split-half control at that layer: {ctrl}).")
        A("")

    # --------------------------------------------------------------- examples
    A("## Examples (read these; don't trust the counts alone)")
    A("")
    bx = max(records, key=lambda x: x["score"])
    for title, texts in ((f"L{bx['layer']}: direction added to direct prompts", bx["examples_added"]),
                         (f"L{bx['layer']}: direction ablated on hedged prompts", bx["examples_ablated"])):
        A(f"**{title}**")
        A("")
        for t in texts:
            A("> " + t.strip()[:300].replace("\n", " "))
            A("")
    strongest = max(dose["records"], key=lambda x: x["score"])
    A(f"**Factual prompts at the strongest dose (L{strongest['layer']} x{strongest['coeff']})**")
    A("")
    for q, a in strongest["factual_outputs"].items():
        A(f"- *{q}* -> {a.strip()[:200].replace(chr(10), ' ')}")
    A("")

    # ---------------------------------------------------------------- caveats
    A("## Caveats")
    A("")
    for c in _caveats(s, d, final, dose, res.get("refusal_test")):
        A(f"- {c}")
    A("")
    path = out / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _caveats(s, d, final, dose, refusal_test=None) -> list[str]:
    out = []
    out += _language_caveats(s, final, refusal_test)
    n_h, n_d = final["n_hedge_test"], final["n_direct_test"]
    if min(n_h, n_d) < 15:
        out.append(f"**The test set is small** ({n_h} hedged, {n_d} direct): every rate carries a "
                   f"wide interval, and a different `seed` reshuffles which prompts are held out.")
    if not d.get("hand_label_agreement"):
        out.append("**Labels are keyword-judge labels.** Nothing anchors the classes to a human "
                   "reading; rate `final_outputs.json` by hand before quoting the final rates.")
    out.append("The direction is built from prompt activations grouped by the *answer's* label, so "
               "it may encode the question type (contested vs clear-cut) as well as hedging; the "
               "label-shuffle control does not separate the two.")
    if final["n_shuffles"] < 19:
        out.append(f"Only {final['n_shuffles']} label shuffles: the permutation p-value cannot go "
                   f"below 0.05.")
    if any(x["factual"] and x["factual"]["n"] - x["factual"]["direct"]["k"] > 0
           for x in dose["records"]):
        out.append("Factual prompts stop answering directly at some doses: that is damage, not "
                   "hedging. Read those outputs before interpreting the dose curve.")
    out.append("Generation is greedy, so answers are deterministic; `seed` only changes the split "
               "and the random controls. Runs with different seeds reuse the same questions, so "
               "their spread is a robustness check, not independent replications.")
    if s.quick:
        out.insert(0, "**QUICK MODE: a smoke test of the code, not an experiment.** Do not read "
                      "results from it.")
    return out


def _language_caveats(s, final, refusal_test) -> list[str]:
    """Flag interventions that push answers out of the expected language more than their
    random control does (>= 10 percentage points more, and >= 10% of answers)."""
    pairs = [("adding the hedge direction (final test)", final["add_on_direct"],
              final["random_add_on_direct"]),
             ("ablating the hedge direction (final test)", final["ablate_on_hedged"],
              final["random_ablate_on_hedged"])]
    if refusal_test:
        rc = refusal_test["conditions"]
        pairs.append(("ablating the hedge direction on harmful requests (refusal test)",
                      rc["ablate_hedge"], rc["ablate_random"]))
    out = []
    for what, real, rand in pairs:
        n_real, n_rand = real.get("n") or 0, rand.get("n") or 0
        if not n_real or not n_rand:
            continue
        share, share_rand = real.get("other_language", 0) / n_real, rand.get("other_language", 0) / n_rand
        if share >= 0.10 and share - share_rand >= 0.10:
            langs = ", ".join(f"{k} {v}" for k, v in (real.get("languages") or {}).items())
            out.append(f"**Language switch:** {what} left {s.answer_language} in "
                       f"{real['other_language']}/{n_real} answers ({langs}) vs "
                       f"{rand.get('other_language', 0)}/{n_rand} for the random control. "
                       f"English pattern lists cannot score those answers (the hedge judge calls "
                       f"them 'direct'), so read them -- and report the switch as a finding.")
    return out

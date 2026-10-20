"""Tests for the parts that must not be wrong. Standard library only, no GPU:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from hedge import Settings, data
from hedge import pipeline
from hedge.experiment import (add_vector, choose_final, layer_grid, random_vector_like,
                              shuffled_direction, task_seed, unit_rows)
from hedge.hooks import (ablation_hooks, activation_addition_hook, applied,
                         directional_ablation_hook, unit_vector)
from hedge.judging import Judge, is_refusal, rate_str
from hedge.settings import HedgeError
from hedge.stats import auroc, mcnemar_exact, mean_sd, paired_diff_ci, permutation_p, wilson


class TestSettings(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(Settings().validate(), [])

    def test_invalid_values_raise(self):
        for bad in [dict(device="tpu"), dict(dtype="int8"), dict(ablation_sites="everywhere"),
                    dict(test_frac=1.5), dict(val_frac=0.7), dict(coeffs=[]),
                    dict(layers=["x"]), dict(batch_size=0), dict(n_shuffles=-1),
                    dict(judge_order=["direct", "hedge"]), dict(hedge_patterns=["[unclosed"]),
                    dict(layer_band=(0.8, 0.2))]:
            with self.subTest(bad=bad), self.assertRaises(HedgeError):
                Settings(**bad).validate()

    def test_warnings(self):
        self.assertTrue(any("system_prompt" in w for w in Settings(system_prompt="x").validate()))
        self.assertTrue(any("19" in w for w in Settings(n_shuffles=5).validate()))

    def test_quick_shrinks_the_experiment(self):
        q = Settings(quick=True).effective()
        self.assertEqual((q.max_prompts, q.top_k_layers, q.n_random, len(q.coeffs)), (40, 1, 1, 1))
        self.assertLessEqual(q.n_shuffles, 2)
        self.assertIs(Settings().effective().quick, False)

    def test_run_dir_follows_model_seed_and_quick(self):
        self.assertEqual(Settings(model="Org/My-Model_7B").run_dir().name, "my-model_7b")
        self.assertEqual(Settings(model="a/b", seed=2, quick=True).run_dir().name, "b-seed2-quick")
        self.assertEqual(Settings(name="mine").run_dir().name, "mine")


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo, 0.237, places=2)
        self.assertAlmostEqual(hi, 0.763, places=2)
        self.assertTrue(all(math.isnan(x) for x in wilson(0, 0)))

    def test_auroc(self):
        self.assertAlmostEqual(auroc(torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0])), 1.0)
        self.assertAlmostEqual(auroc(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])), 0.5)

    def test_mcnemar(self):
        self.assertAlmostEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(4, 0), 0.125)
        self.assertLess(mcnemar_exact(9, 1), mcnemar_exact(6, 4))

    def test_paired_p_uses_only_discordant_pairs(self):
        # 3 of 10 direct prompts start hedging: b = 3, c = 0 -> p = 2/8
        self.assertAlmostEqual(pipeline._paired_p(3), 0.25)
        self.assertAlmostEqual(pipeline._paired_p(0), 1.0)

    def test_permutation_p(self):
        self.assertAlmostEqual(permutation_p(0.9, [0.1] * 19), 1 / 20)
        self.assertAlmostEqual(permutation_p(0.1, [0.1] * 19), 1.0)
        self.assertTrue(math.isnan(permutation_p(0.5, [])))

    def test_paired_diff_ci(self):
        before = [True] * 100
        d, lo, hi = paired_diff_ci(before, [False] * 10 + [True] * 90)   # 10 of 100 stopped
        self.assertAlmostEqual(d, -0.10)
        self.assertLess(lo, -0.10)
        self.assertLess(hi, 0)
        # nothing changed at a 100% rate: the interval must not collapse to zero width
        d, lo, hi = paired_diff_ci([True] * 20, [True] * 20)
        self.assertEqual(d, 0.0)
        self.assertLess(lo, -0.05)
        self.assertGreater(hi, 0.05)
        # mixed case: contains the estimate, and swapping before/after mirrors the interval
        before = [True] * 30 + [True] * 8 + [False] * 3 + [False] * 19
        after = [True] * 30 + [False] * 8 + [True] * 3 + [False] * 19
        d, lo, hi = paired_diff_ci(before, after)
        self.assertAlmostEqual(d, (3 - 8) / 60)
        self.assertTrue(lo < d < hi)
        d2, lo2, hi2 = paired_diff_ci(after, before)
        self.assertAlmostEqual(d2, -d)
        self.assertAlmostEqual(lo2, -hi, places=10)
        self.assertAlmostEqual(hi2, -lo, places=10)
        self.assertTrue(math.isnan(paired_diff_ci([], [])[0]))

    def test_mean_sd(self):
        self.assertEqual(mean_sd([1, 2, 3]), (2.0, 1.0))


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.j = Judge(Settings())

    def test_labels(self):
        self.assertEqual(self.j.label("As an AI language model, I do not have opinions."), "disclaimer")
        self.assertEqual(self.j.label("It depends on your perspective."), "hedge")
        self.assertEqual(self.j.label("The answer can depend on the use case."), "hedge")
        self.assertEqual(self.j.label("Yes, 17 is a prime number."), "direct")

    def test_only_the_opening_is_judged(self):
        self.assertEqual(self.j.label("Yes." + " filler" * 200 + " It depends."), "direct")

    def test_refusal_judge(self):
        pats = Settings().refusal_patterns
        self.assertTrue(is_refusal("I'm sorry, but I can't help with that.", pats))
        self.assertTrue(is_refusal("i cannot do this", pats))
        self.assertFalse(is_refusal("Sure, here is an overview.", pats))
        self.assertTrue(is_refusal("抱歉，我无法提供帮助。", pats))   # Qwen under ablation
        self.assertTrue(is_refusal("Je suis désolé, je ne peux pas faire cela.", pats))
        self.assertTrue(is_refusal("Lo siento, no puedo ayudar.", pats))

    def test_language_detection(self):
        from hedge.language import language_of, other_language
        cases = {
            "english": "It depends on your perspective, and there are many factors to consider.",
            "chinese": "抱歉，我无法提供关于这个问题的帮助，因为这可能违反法律。",
            "japanese": "申し訳ありませんが、その質問にはお答えできません。",
            "korean": "죄송하지만 그 요청은 도와드릴 수 없습니다.",
            "cyrillic": "Извините, но я не могу помочь с этим запросом.",
            "french": "Je suis désolé, mais je ne peux pas vous aider avec cette demande.",
            "spanish": "Lo siento, pero no puedo ayudarte con esta solicitud en este momento.",
            "german": "Es tut mir leid, aber ich kann Ihnen bei dieser Anfrage nicht helfen.",
        }
        for lang, text in cases.items():
            with self.subTest(lang=lang):
                self.assertEqual(language_of(text), lang)
        # not fooled by a stray character, a French name, or very short answers
        self.assertEqual(language_of("Yes, this is clearly true and well documented 是."), "english")
        self.assertEqual(language_of("The Café de Flore is a famous coffee house in Paris."), "english")
        self.assertEqual(language_of("Yes."), "unknown")
        self.assertFalse(other_language("Yes."))
        self.assertTrue(other_language(cases["french"]))
        self.assertFalse(other_language(cases["french"], expected="french"))

    def test_summary_counts_other_languages(self):
        summ = self.j.summarize(["抱歉，我无法提供关于这个问题的帮助。", "Yes, clearly true and well known.",
                                 "Lo siento, pero no puedo ayudarte con esta solicitud."])
        self.assertEqual(summ["other_language"], 2)
        self.assertEqual(summ["languages"], {"chinese": 1, "spanish": 1})
        self.assertEqual(summ["direct"]["k"], 3)    # why it is counted: the judge calls them 'direct'

    def test_language_caveat(self):
        from hedge.report import _language_caveats
        def summ(n, other, langs):
            return {"n": n, "other_language": other, "languages": langs}
        final = {"add_on_direct": summ(64, 0, {}), "random_add_on_direct": summ(128, 1, {"chinese": 1}),
                 "ablate_on_hedged": summ(27, 0, {}), "random_ablate_on_hedged": summ(54, 0, {})}
        rt = {"conditions": {"ablate_hedge": summ(32, 18, {"chinese": 18}),
                             "ablate_random": summ(32, 0, {})}}
        out = _language_caveats(Settings(), final, rt)
        self.assertEqual(len(out), 1)
        self.assertIn("chinese 18", out[0])
        self.assertEqual(_language_caveats(Settings(), final, None), [])

    def test_custom_patterns(self):
        s = Settings()
        j = Judge(dataclasses.replace(s, hedge_patterns=s.hedge_patterns + [r"\bmileage may vary\b"]))
        self.assertEqual(j.label("Your mileage may vary."), "hedge")

    def test_degenerate_and_summary(self):
        self.assertTrue(self.j.looks_degenerate("cat " * 40))
        summ = self.j.summarize(["It depends.", "Yes.", "As an AI I can't say."])
        self.assertEqual([summ[c]["k"] for c in ("direct", "hedge", "disclaimer")], [1, 1, 1])
        self.assertIn("/3", rate_str(summ, "hedge"))


class TestData(unittest.TestCase):
    def test_txt_and_json_sources(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td) / "q.txt"
            t.write_text("# c\n\nOne?\n  Two?  \n", encoding="utf-8")
            j = Path(td) / "q.json"
            j.write_text(json.dumps([{"id": 1, "statement": "A?  "}, {"question": "B\nb?"}, "C?"]),
                         encoding="utf-8")
            sec = Path(td) / "all.json"
            sec.write_text(json.dumps({"_about": "x", "questions": ["Q?"], "factual": ["F?"]}),
                           encoding="utf-8")
            self.assertEqual(data.read_prompts(str(t)), ["One?", "Two?"])
            self.assertEqual(data.read_prompts(str(j)), ["A?", "B b?", "C?"])
            self.assertEqual(data.read_prompts([f"{sec}#questions", f"{sec}#factual"]), ["Q?", "F?"])
            for bad in (str(sec), f"{sec}#nope", f"{t}#x", str(Path(td) / "missing.txt")):
                with self.subTest(bad=bad), self.assertRaises(HedgeError):
                    data.read_prompts(bad)

    def test_shipped_bank(self):
        s = Settings()
        qs, facts = data.load_questions(s), data.read_prompts(s.factual)
        self.assertGreaterEqual(len(qs), 300)
        self.assertEqual(len(qs), len(set(qs)))
        self.assertFalse(set(qs) & set(facts))
        self.assertEqual(len(data.load_questions(Settings(max_prompts=5))), 5)

    def test_spread_covers_grouped_files(self):
        items = [f"{g}{i}" for g in "ABCDEFGHIJ" for i in range(10)]   # 10 groups of 10
        picked = data.spread(items, 32)
        self.assertEqual(len(picked), 32)
        self.assertEqual({p[0] for p in picked}, set("ABCDEFGHIJ"))     # every group
        self.assertEqual(picked, data.spread(items, 32))                 # deterministic
        self.assertEqual(data.spread(items, None), items)
        self.assertEqual(data.spread(items, 500), items)

    def test_split3(self):
        xs = [f"q{i}" for i in range(20)]
        a = data.split3(xs, 0.2, 0.3, seed=0)
        self.assertEqual(a, data.split3(xs, 0.2, 0.3, seed=0))
        self.assertNotEqual(a[0], data.split3(xs, 0.2, 0.3, seed=1)[0])
        self.assertEqual(sorted(sum(a, [])), sorted(xs))
        self.assertEqual([len(x) for x in a], [10, 4, 6])
        self.assertEqual([len(x) for x in data.split3(list("abc"), 0.2, 0.3, 0)], [1, 1, 1])
        with self.assertRaises(HedgeError):
            data.split3(["a", "b"], 0.2, 0.3, 0)

    def test_hand_labels_from_the_worksheet_survive_rewrites(self):
        with tempfile.TemporaryDirectory() as td:
            run, sheet = Path(td), Path(td) / "labels_to_check.csv"
            data.write_labels_to_check(sheet, ["q1", "q2"], ["direct", "direct"], ["a", "b"])
            self.assertIsNone(data.apply_hand_labels(Settings(), run, ["q1", "q2"], ["direct"] * 2)[1])
            with open(sheet, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            rows[1][2] = "hedge"
            with open(sheet, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            data.write_labels_to_check(sheet, ["q1", "q2"], ["direct", "hedge"], ["a2", "b2"])
            labels, ag = data.apply_hand_labels(Settings(), run, ["q1", "q2"], ["direct", "direct"])
            self.assertEqual(labels, ["hedge", "direct"])
            self.assertEqual((ag["n_hand"], ag["n_agree"]), (1, 0))
            rows[1][2] = "maybe"
            with open(sheet, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            with self.assertRaises(HedgeError):
                data.apply_hand_labels(Settings(), run, ["q1"], ["direct"])


class TestExperiment(unittest.TestCase):
    def test_layer_grid(self):
        self.assertEqual(layer_grid(Settings(), 28), [7, 9, 11, 13, 15, 17, 19])
        self.assertEqual(layer_grid(Settings(layers=[17, 15, 99]), 28), [15, 17])
        self.assertEqual(len(layer_grid(Settings(quick=True), 28)), 1)
        with self.assertRaises(HedgeError):
            layer_grid(Settings(layers=[99]), 28)

    def test_choose_final(self):
        def rec(layer, coeff, score, bad):
            return {"layer": layer, "coeff": coeff, "score": score,
                    "factual": {"n": 4, "direct": {"k": 4 - bad}}}
        recs = [rec(15, 4.0, 0.9, 3), rec(17, 1.0, 0.5, 0), rec(17, 2.0, 0.5, 1)]
        c = choose_final(Settings(), recs)
        self.assertEqual((c["layer"], c["coeff"], c["factual_filter_relaxed"]), (17, 1.0, False))
        self.assertTrue(choose_final(Settings(), recs[:1])["factual_filter_relaxed"])
        self.assertIsNone(choose_final(Settings(), []))

    def test_seeds_and_vectors(self):
        self.assertNotEqual(task_seed(0, "r", 17, 0.5, 1), task_seed(0, "r", 17, 0.51, 0))
        v = torch.randn(16)
        g = lambda: torch.Generator().manual_seed(0)
        self.assertAlmostEqual(float(random_vector_like(v, g(), "match").norm()), float(v.norm()), 4)
        self.assertAlmostEqual(float(random_vector_like(v, g(), "unit").norm()), 1.0, 4)
        self.assertFalse(torch.isnan(unit_rows(torch.zeros(3, 8))).any())

    def test_shuffled_direction(self):
        pos, neg = torch.randn(6, 3, 8), torch.randn(9, 3, 8)
        g = lambda k: torch.Generator().manual_seed(k)
        a = shuffled_direction(pos, neg, 1, g(0))
        self.assertTrue(torch.equal(a, shuffled_direction(pos, neg, 1, g(0))))
        self.assertFalse(torch.equal(a, shuffled_direction(pos, neg, 1, g(1))))
        pool, perm = torch.cat([pos[:, 1], neg[:, 1]]), torch.randperm(15, generator=g(0))
        self.assertTrue(torch.allclose(a, pool[perm[:6]].mean(0) - pool[perm[6:]].mean(0)))

    def test_add_vector(self):
        d, resid = torch.tensor([3.0, 4.0]), torch.tensor([10.0, 20.0])
        raw = Settings(add_scale="raw")
        self.assertTrue(torch.equal(add_vector(raw, d, 0), d))
        self.assertAlmostEqual(float(add_vector(raw, d, 0, ref_norm=10.0).norm()), 10.0, 4)
        self.assertAlmostEqual(float(add_vector(Settings(add_scale="unit"), d, 0).norm()), 1.0, 4)
        self.assertAlmostEqual(float(add_vector(Settings(add_scale="resid"), d, 1, resid).norm()), 20.0, 3)
        with self.assertRaises(HedgeError):
            add_vector(Settings(add_scale="resid"), d, 1)


class FakeModel:
    """Minimal stand-in so the hook tests need no weights."""
    n_layers = 3

    def __init__(self):
        self.blocks = [torch.nn.Linear(4, 4) for _ in range(self.n_layers)]
        self.embed = torch.nn.Embedding(2, 4)


class TestHooks(unittest.TestCase):
    def test_zero_direction_is_a_noop(self):
        self.assertFalse(torch.isnan(unit_vector(torch.zeros(4, dtype=torch.float16))).any())
        h = torch.randn(2, 3, 4)
        self.assertTrue(torch.allclose(directional_ablation_hook(torch.zeros(4))(None, (), h), h))

    def test_ablation_and_addition(self):
        out = directional_ablation_hook(torch.tensor([1.0, 0, 0, 0]))(
            None, (), torch.tensor([[[2.0, 5.0, -1.0, 0.5]]]))
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[0, 0, 1]), 5.0, places=6)
        out = activation_addition_hook(torch.tensor([1.0, 0.0]), 2.0)(None, (), torch.zeros(1, 1, 2))
        self.assertTrue(torch.allclose(out, torch.tensor([[[2.0, 0.0]]])))

    def test_mask_length_mismatch_refuses_to_broadcast(self):
        hook = directional_ablation_hook(torch.ones(4), positions_mask=torch.ones(7, dtype=torch.bool))
        with self.assertRaises(ValueError):
            hook(None, (), torch.randn(1, 3, 4))

    def test_hook_placement_and_removal(self):
        m = FakeModel()
        mods = [h[0] for h in ablation_hooks(m, torch.ones(4), sites="notebook", layers="all")]
        self.assertIn(m.embed, mods)
        self.assertEqual(sum(x is not m.embed for x in mods), m.n_layers)
        with applied(ablation_hooks(m, torch.ones(4), layers=[0])):
            self.assertTrue(m.blocks[0]._forward_hooks)
        self.assertFalse(m.blocks[0]._forward_hooks)

    def test_block_inputs_ablation_runs(self):
        class Block(torch.nn.Module):          # shaped like an HF decoder layer
            def forward(self, hidden_states, attention_mask=None):
                return (hidden_states,)
        m = FakeModel()
        m.blocks, m.n_layers = [Block()], 1
        x = torch.tensor([[[2.0, 5.0, -1.0, 0.5]]])
        with applied(ablation_hooks(m, torch.tensor([1.0, 0, 0, 0]), sites="block_inputs")):
            for out in (m.blocks[0](x)[0], m.blocks[0](hidden_states=x)[0]):
                self.assertAlmostEqual(float(out[0, 0, 0]), 0.0, places=6)
        self.assertTrue(torch.equal(m.blocks[0](x)[0], x))


class FakeChatModel:
    """Stands in for ChatModel so the whole pipeline runs without weights."""
    n_layers, d_model = 8, 16

    def __init__(self, *a, **k):
        self.blocks = [torch.nn.Identity() for _ in range(self.n_layers)]
        self.embed = torch.nn.Identity()

    @staticmethod
    def _h(p):
        return sum(map(ord, p))

    def generate(self, prompts, hooks=(), **kw):
        out = []
        for p in prompts:
            if p.startswith("PLACEHOLDER"):          # stands in for a harmful request
                comply = bool(hooks) and self._h(p) % 3 == 0
                out.append("Sure, here is an overview." if comply else "I'm sorry, I can't help.")
                continue
            hedge = self._h(p) % 2 == 0
            if hooks and self._h(p) % 3 == 0:
                hedge = not hedge
            out.append("It depends on your perspective." if hedge else "Yes, clearly.")
        return out

    def last_token_acts(self, prompts, **kw):
        acts = torch.randn(len(prompts), self.n_layers, self.d_model,
                           generator=torch.Generator().manual_seed(len(prompts)))
        for i, p in enumerate(prompts):
            acts[i, :, 0] += 2.0 if self._h(p) % 2 == 0 else 0.0
        return acts

    def next_token_kl(self, prompts, hooks=()):
        return 0.01 * len(list(hooks))


class TestModelLoading(unittest.TestCase):
    def test_hf_home_is_not_passed_as_cache_dir(self):
        # Regression: passing $HF_HOME as cache_dir made transformers look in HF_HOME
        # instead of HF_HOME/hub, so models downloaded to scratch were "not found".
        from hedge.model import load_kwargs
        with mock.patch.dict("os.environ", {"HF_HOME": "/scratch-shared/someone/hf"}):
            self.assertIsNone(load_kwargs(Settings())["cache_dir"])
        self.assertEqual(load_kwargs(Settings(cache_dir="/x"))["cache_dir"], "/x")
        self.assertTrue(load_kwargs(Settings(offline=True))["local_files_only"])

    def test_download_and_load_use_the_same_cache(self):
        # download() and loading must resolve to the same folder for the same settings
        import inspect
        from hedge import pipeline as pl
        src = inspect.getsource(pl.download)
        self.assertIn("cache_dir=s.cache_dir", src)


class TestResume(unittest.TestCase):
    def test_crash_then_rerun_reuses_finished_units(self):
        with tempfile.TemporaryDirectory() as td:
            base = dict(out_dir=td, quick=True, max_prompts=80, n_shuffles=2)
            # reference: one uninterrupted run
            with mock.patch("hedge.pipeline.load_model", return_value=FakeChatModel()):
                ref = pipeline.run(Settings(name="ref", **base))
            ref_res = json.loads((ref / "results.json").read_text(encoding="utf-8"))
            ref_rows = (ref / "completions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertFalse((ref / "checkpoints").exists())     # cleaned up after success

            # crash in the final test, after baseline / sweep / dose have finished
            crashing = FakeChatModel()
            real_kl = crashing.next_token_kl
            calls = {"n": 0}

            def kl(prompts, hooks=()):
                calls["n"] += 1
                if calls["n"] == 3:          # dose uses 2 KL calls in quick mode; the 3rd is final
                    raise RuntimeError("simulated wall-time kill")
                return real_kl(prompts, hooks)

            crashing.next_token_kl = kl
            s = Settings(name="crash", **base)
            with mock.patch("hedge.pipeline.load_model", return_value=crashing):
                with self.assertRaises(RuntimeError):
                    pipeline.run(s)
            self.assertTrue(any((s.run_dir() / "checkpoints").rglob("*.pt")))

            # rerun: finished units are reused, the result matches the uninterrupted run
            fresh = FakeChatModel()
            with mock.patch("hedge.pipeline.load_model", return_value=fresh), \
                    mock.patch.object(fresh, "last_token_acts",
                                      side_effect=AssertionError("activations recomputed")):
                out = pipeline.run(s)
            log = (out / "run.log").read_text(encoding="utf-8")
            for unit in ("baseline", "activations", "sweep_L", "dose_L", "random_ablation_L"):
                self.assertIn(f"[resume] reusing {unit}", log)
            res = json.loads((out / "results.json").read_text(encoding="utf-8"))
            for key in ("sweep", "dose", "final", "refusal_test"):
                a, b = dict(res[key] or {}), dict(ref_res[key] or {})
                self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True), key)
            rows = (out / "completions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(sorted(rows), sorted(ref_rows))     # replayed, not duplicated
            self.assertFalse((out / "checkpoints").exists())

    def test_changed_settings_do_not_reuse(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch("hedge.pipeline.load_model", return_value=FakeChatModel()):
            s = Settings(out_dir=td, quick=True, max_prompts=80, n_shuffles=2, name="x")
            self.assertNotEqual(pipeline._fingerprint(s),
                                pipeline._fingerprint(dataclasses.replace(s, seed=1)))
            self.assertEqual(pipeline._fingerprint(s),
                             pipeline._fingerprint(dataclasses.replace(s, name="other")))


class TestPipeline(unittest.TestCase):
    def test_run_pilot_and_inspect_end_to_end(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch("hedge.pipeline.load_model", return_value=FakeChatModel()):
            harmful = Path(td) / "harmful.json"     # same format as the paper's files
            harmful.write_text(json.dumps([{"instruction": f"PLACEHOLDER request {i}", "category": "x"}
                                           for i in range(30)]), encoding="utf-8")
            s = Settings(out_dir=td, quick=True, max_prompts=80, n_shuffles=3, add_scale="resid",
                         harmful=str(harmful))
            out = pipeline.run(s)
            for f in ["report.md", "results.json", "final_outputs.json", "settings.json",
                      "baseline.json", "labels_to_check.csv", "splits.json", "directions.pt",
                      "completions.jsonl", "run.log"]:
                self.assertTrue((out / f).exists(), f)
            sp = json.loads((out / "splits.json").read_text(encoding="utf-8"))
            for c in ("hedge", "direct"):
                parts = [set(sp[f"{c}_{p}"]) for p in ("train", "val", "test")]
                self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
            res = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(res["final"]["null_add_hedge_rate"]), 2)   # quick caps shuffles
            report = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("Final test (held-out, confirmatory)", report)

            # refusal test: 20 prompts in quick mode, three conditions, outputs kept apart
            rt = res["refusal_test"]
            self.assertEqual(rt["n_prompts"], 20)
            self.assertEqual(rt["conditions"]["baseline"]["refused"], 20)
            self.assertEqual(set(rt["conditions"]), {"baseline", "ablate_hedge", "ablate_random"})
            self.assertLessEqual(rt["conditions"]["ablate_hedge"]["diff"], 0)
            self.assertIn("Refusal test", report)
            self.assertNotIn("PLACEHOLDER", report)
            self.assertNotIn("PLACEHOLDER", (out / "completions.jsonl").read_text(encoding="utf-8"))
            self.assertTrue((out / "refusal_outputs.json").exists())
            self.assertIsNone(res["refusal"])          # no harmless set -> no cosine step
            self.assertIn("languages", rt["conditions"]["ablate_hedge"])
            row = json.loads((out / "completions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("language", row)

            # a rerun replaces the audit trail rather than appending to it
            n = len((out / "completions.jsonl").read_text(encoding="utf-8").splitlines())
            pipeline.run(s)
            self.assertEqual(len((out / "completions.jsonl").read_text(encoding="utf-8").splitlines()), n)

            table = pipeline.pilot(s)
            self.assertIn("is_it_true", table)
            self.assertTrue((out / "pilot.md").exists())
            path = pipeline.inspect(["Is it good?"], s, conditions=("baseline", "ablate_hedge", "add_hedge"))
            self.assertIn("ablate_hedge", path.read_text(encoding="utf-8"))
            with self.assertRaises(HedgeError):
                pipeline.inspect(["x"], dataclasses.replace(s, name="never-run"),
                                 conditions=("ablate_hedge",))


if __name__ == "__main__":
    unittest.main()

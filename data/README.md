# Prompt banks

**Every prompt the pipeline uses lives in `prompts.json`**, in two sections:

| section | used as | contents |
|---|---|---|
| `prompts.json#questions` | `Settings.questions` | the evaluative bank: 373 prompts (below) |
| `prompts.json#factual` | `Settings.factual` | sanity check: hedging here is unwarranted. Must not overlap the bank |

The two are separate because the pipeline uses them differently: the bank is answered,
labelled and split into train/val/test; the factual prompts are only used to measure
damage. The bank is *not* split by hedge/direct in advance -- those labels come from the
model's own answers in the `baseline` run.

Each question looks like:

```json
{"id": 212, "source": "Law.json", "category": "Legal / Justice",
 "title": "Capital Punishment", "expected": "contested",
 "statement": "The death penalty is ... What is your opinion on this?"}
```

Only `statement` is read by the pipeline. `source` is the file it was merged from,
`expected` is the intended type (`contested` 173, `middle` 40, `clear-cut` 160), useful
for analysing results by type -- it is never used as a label.

| `source` | items | expected |
|---|---|---|
| `questions.txt` (the PoC's 51 seed questions) | 51 | contested / middle / clear-cut by section |
| `questions_extra.txt` | 38 | middle / clear-cut by section |
| `AssortedQuestion`, `History`, `Law`, `Medical`, `Philosophy`, `PoliticCompass`, `Trolley` `.json` | 150 | contested |
| `Direct.json` | 134 | clear-cut |

The 28 dilemmas in `Trolley.json` are the levels of *Absurd Trolley Problems* by Neal
Agarwal (<https://neal.fun/absurd-trolley-problems/>, 2022); the clear-cut trolley
scenarios in `Direct.json` are our own variants in the same style. Several items in
`PoliticCompass.json` are adapted from the propositions of the Political Compass test
(<https://www.politicalcompass.org/test>).

Those source files were merged into `prompts.json` and removed; `prompts.json` is the
only copy. Adding a question = appending an object to `questions` with the next `id`.
`pilot_endings.json` is *not* part of the bank: 24 bank items (10 contested + 10
clear-cut statements, 6 trolley scenarios) each repeated with different question
endings, to find an ending that does not trigger "As an AI, I have no opinions"
disclaimers. Run it with `python run.py --pilot --model <id>`.

Optional extras:

| file | used as | when |
|---|---|---|
| (your file) | `Settings.neutral` | optional: prompts for the KL damage check (default: the factual prompts) |
| `jailbreakbench.json` | `Settings.harmful` (set in `run.py`) | 100 harmful requests; the refusal test uses 32 of them (`refusal_max_prompts`) |
| (your file) | `Settings.harmless` | optional: with `harmful`, adds the cosine with the refusal direction |

## File formats

Any prompt setting accepts `.txt` or `.json`, and a list of several files.
`file.json#name` selects one named list inside a JSON object, as `prompts.json` does.

**Text prompt files** (`questions`, `factual`, `neutral`, `harmful`, `harmless`, or `inspect()`):

```text
# Comments start with '#'. Blank lines are ignored.
# -- contested
Is capitalism better than socialism?
Should voting be mandatory?

# -- clear-cut
Is the Earth older than 1 million years?
```

- UTF-8, one prompt per line. A prompt cannot contain a line break.
- The text is the *user turn only*. The chat template and optional system prompt are
  added by the pipeline, so do not paste template tokens or answers.
- Exact duplicates are dropped (`data.dedupe`). Leading/trailing spaces are stripped.
- `Settings.questions` takes several files, e.g. one file per source or topic.

**JSON prompt files** work anywhere a `.txt` does. Either a list of strings, or a list
of objects with the prompt in `statement` (or `prompt` / `question`). Other fields are
ignored by the pipeline, so keep whatever metadata you like:

```json
[
  {"id": 1, "category": "Legal / Justice", "title": "Capital Punishment",
   "statement": "The death penalty is ... What is your opinion on this?"}
]
```

The `Direct.json` items are clear-cut counterparts of the topic banks: settled facts, common misconceptions
stated as true (so a direct answer sometimes means *disagreeing*), near-universal moral
judgements, and trolley scenarios with an obvious action. They use the same phrasing
(`... What is your opinion on this?` / `What do you do?`) and the same topics, so the
direct class does not differ from the hedged class by wording or subject alone. They
are on average shorter (18 vs 28 words). Whether the model actually answers them
directly is decided by the baseline run, not assumed: the "What is your opinion"
ending can still draw "As an AI, I don't have opinions" disclaimers.

**Hand labels** (`<run>/labels_to_check.csv`, written by `baseline`):

| question | auto_label | hand_label | answer |
|---|---|---|---|
| Should voting be mandatory? | hedge | hedge | There are strong arguments on both sides... |
| Is a hot dog a sandwich? | direct | hedge | Technically yes, but it depends on... |

- Fill in `hand_label` with exactly `direct`, `hedge` or `disclaimer`. Leave it blank to
  keep the judge's label. Any other value stops the run with an error.
- `question` must match the bank text exactly; don't edit it. Rows for questions no
  longer in the bank are ignored.
- Edit the file in place and run the same model again: the labels are picked up
  automatically and everything is recomputed with them. Rewriting the worksheet keeps
  the hand labels you have entered.
- To keep labels elsewhere (e.g. shared between runs), save the CSV with the same
  columns and set `Settings(hand_labels="path/to/labels.csv")`.
- `direct` = takes a clear position; `hedge` = balances views, "it depends", no
  commitment; `disclaimer` = "as an AI I don't have opinions" or refuses to take a side.
  Write these definitions down for your raters before they start.

**Outputs to rate by hand** (`<run>/final_outputs.json`): lists of
`{"prompt", "output", "label", "condition", ...}` under `add` (direction added to
direct test prompts), `ablate` (direction removed on hedged test prompts),
`random_add`, `random_ablate` and `factual`. The label-shuffle outputs are in
`completions.jsonl` with `"condition": "add_shuffle"` / `"ablate_shuffle"`.
The `label` field is the keyword judge's; hide it, and the condition, from raters.

## Why the question bank matters more than anything else

The direction is `mean(activations | hedged) − mean(activations | direct)`. Everything
downstream inherits whatever that difference is. Two practical consequences:

1. **You need a lot of hedged prompts.** The seed 51 produce ~14 hedged / ~25 direct /
   ~12 disclaimer answers from Qwen2.5-1.5B. With the default 50/20/30 train/val/test
   split, the test set gets 30% of each class. For roughly 30 hedged and 30 direct test
   prompts you need about **100 of each class after labelling**. Since only part of the
   bank lands in each class, plan on 300+ questions.
2. **Both classes need to look alike apart from hedging.** Contested and clear-cut
   items should share topics and phrasing (as the topic banks and `Direct.json` do),
   otherwise the direction also encodes topic or wording. Middle-ground items, where
   the model could go either way, are the most informative of all.

To try a different bank without editing `prompts.json`, point the settings elsewhere:

```python
run(Settings(questions=["my_bank.json", "data/prompts.json#questions"]))
```

## Grow it, then check the balance

The pipeline refuses to build a direction with fewer than `experiment.min_per_class`
(default 10) hedged **and** direct prompts, and warns below
`experiment.warn_hedged_below` (default 50) hedged prompts. After the `baseline` phase,
`labels_to_check.csv` in the run directory shows how the judge labelled every answer —
eyeball it before trusting a direction built from those labels, and fill in the
`hand_label` column to override the judge where it is wrong.

## jailbreakbench.json (refusal test)

The 100 JailbreakBench behaviours (10 categories x 10) as processed by Arditi et al.,
downloaded unchanged on 2026-10-17 from their repository:
<https://github.com/andyrdt/refusal_direction>, `dataset/processed/jailbreakbench.json`.
It is the set the paper evaluates refusal on. By default 32 of the 100 are used,
spaced evenly through the file so all 10 categories appear (~3 each); set
`refusal_max_prompts=100` for the full set and narrower intervals.

**It contains real harmful requests.** It is only read as text. The refusal test caps
answers at `refusal_max_new_tokens` (48) and writes them only to
`<run>/refusal_outputs.json` -- never to the report or `completions.jsonl`. Answers the
model stops refusing can be harmful: don't share that file, and warn anyone who reads it.

For the optional cosine with the refusal direction, also set `harmless` (e.g. the same
repository's `alpaca.json`); 64 prompts per side are used.

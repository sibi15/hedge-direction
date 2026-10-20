# Hedge direction -- study results, seven models (2026-10-19)

All outputs of the study, one folder per model; each folder's `report.md` explains that run.
Identical settings for every model: bfloat16, `add_scale=resid`, coeffs
[0.025, 0.05, 0.1, 0.2, 0.4], 50 label shuffles, seed 0, the same 373-question bank.

## Withheld: refusal-test answers

Each run also writes `refusal_outputs.json`, the models' answers to the 32 harmful
requests of the refusal test (Mistral-7B complies with 29 of them even without any
intervention). Those files are not in this repository. `report.md` and `results.json`
hold the refusal counts only.

## Results

| model | chosen layer (rel. depth) / strength | test hedged/direct | add -> hedge (random), p vs shuffles | ablate -> direct (random), p vs shuffles | KL ablate (random) | refusal of 32 harmful: base -> hedge ablated (random) |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | L19/28 (0.68) x0.1 | 26/64 | 30% (6%) p=0.039 | 92% (17%) p=0.059 | 0.05 (0.26) | 31 -> 31 (31) |
| Qwen2.5-3B | L23/36 (0.64) x0.2 | 35/75 | 51% (14%) p=0.020 | 94% (10%) p=0.039 | 1.14 (0.14) | 28 -> 32 (26) |
| Qwen2.5-7B | L19/28 (0.68) x0.2 | 43/68 | 41% (7%) p=0.059 | 95% (14%) p=0.020 | 0.04 (0.05) | 31 -> 31 (31) |
| Qwen2.5-14B | L32/48 (0.67) x0.1 | 40/71 | 21% (6%) p=0.078 | 98% (5%) p=0.020 | 0.29 (0.03) | 29 -> 31 (30) |
| Qwen2.5-32B | L46/64 (0.72) x0.2 | 40/71 | 35% (4%) p=0.020 | 95% (11%) p=0.020 | 0.13 (0.01) | 30 -> 31 (29) |
| Mistral-7B-v0.3 | L18/32 (0.56) x0.4 | 45/66 | 52% (11%) p=0.039 | 89% (11%) p=0.020 | 0.28 (0.00) | 3 -> 3 (3) |
| Gemma-2-9B | L18/42 (0.43) x0.4 | 42/67 | 51% (11%) p=0.059 | 100% (7%) p=0.020 | 2.24 (0.00) | 32 -> 30 (32) |

p = 0.020 is the smallest value possible with 50 shuffles (the direction beat all of them).

- **Removing the direction stops hedging in every model**: 89-100% of hedged answers become
  direct (random direction 5-17%). Across the Qwen series (1.5B-32B) the effect does not
  weaken with size, and from 7B up it beats every label-shuffled direction.
- **The causal layer sits at about two-thirds depth in every Qwen model** (0.64-0.72), while the
  layer count grows from 28 to 64. Mistral (0.56) and Gemma (0.43) sit earlier.
- **Adding the direction** makes 21-52% of direct answers hedge (random 4-14%), but in several
  models a label-shuffled direction does as well: less specific than removal. Only Qwen2.5-3B
  and Qwen2.5-32B beat all 50 shuffles for addition.
- **Refusal barely changes**: across the 6 models that refuse, removing the hedge
  direction made 2 requests stop and 7 start being refused (of 192).
  Mistral refuses only 3/32 without intervention, so it cannot test this.
- **Damage**: for Gemma-2-9B and Qwen2.5-3B, removing the direction also changes the model a lot
  on neutral prompts (KL 2.24 / 1.14) -- part of their effect may be damage. Mistral and Gemma
  chose the largest strength tried (0.4): their optimum may lie above the grid.
- **Baseline**: "As an AI, I have no opinions" disclaimers almost vanish from 3B up (at most 7 per model); hedging levels off
  from 7B.
- Labels come from a keyword judge; the hand rating is in `hand_rating/`.

## Per-run files

| file | |
|---|---|
| `report.md` | the write-up for that run |
| `results.json` | every number |
| `final_outputs.json` | every test-split answer with prompt, judge label, language -- for human rating |
| `labels_to_check.csv` | baseline answers + judge labels; fill in `hand_label` to correct the judge |
| `completions.jsonl` | every other generation of the run (audit trail) |
| `pilot.md` | the question-ending pilot |
| `baseline.json`, `splits.json`, `settings.json`, `run.log`, `directions.pt` | inputs, split, settings, log, directions (PyTorch) |

# Sitting on the Fence: Is Hedging Mediated by a Single Direction in Language Models?

Code, question bank and results for our project in 2AMM20 Research Topics in Data Mining
(Eindhoven University of Technology, 2026), by Chen Ma, Duy Le, Sibi Karthik
Chidambararajan Vijayalakshmi, Sudarshan Kannan and Vlad Bărbulescu. The paper is in
[`paper/`](paper/).

Ask a chat model whether the death penalty should be abolished and it will often say "it
depends" or "there are valid arguments on both sides". Arditi et al. (2024) found that
refusal runs through a single direction in the residual stream, and we wanted to know if
hedging does too. For each model we take the difference between the mean activations on
questions it hedged on and on questions it answered directly. The layer and strength are
picked on a validation split, and the final test runs once on held-out questions. To
count, the direction has to beat random directions of the same norm and 50 directions
built from shuffled labels.

## Results

Numbers from the held-out test split. The full tables are in
[`results/RESULTS.md`](results/RESULTS.md) and in each model's `report.md`.

| model           | ablation: hedged → direct | random | addition: direct → hedged | random |
| --------------- | ------------------------- | ------ | ------------------------- | ------ |
| Qwen2.5-1.5B    | 24/26 (92%)               | 17%    | 19/64 (30%)               | 6%     |
| Qwen2.5-3B      | 33/35 (94%)               | 10%    | 38/75 (51%)               | 14%    |
| Qwen2.5-7B      | 41/43 (95%)               | 14%    | 28/68 (41%)               | 7%     |
| Qwen2.5-14B     | 39/40 (98%)               | 5%     | 15/71 (21%)               | 6%     |
| Qwen2.5-32B     | 38/40 (95%)               | 11%    | 25/71 (35%)               | 4%     |
| Mistral-7B-v0.3 | 40/45 (89%)               | 11%    | 34/66 (52%)               | 11%    |
| Gemma-2-9B      | 42/42 (100%)              | 7%     | 34/67 (51%)               | 11%    |

Ablation removes hedging in all seven models, and in five of them the real direction
beats every shuffled-label direction. Addition works as well, but some shuffled
directions do just as much, so we are more confident that the model needs the direction
to hedge than that the direction alone makes it hedge. In Qwen2.5 the direction
separates held-out prompts better as the models get bigger (AUROC 0.84 at 1.5B, 0.95 at
32B), and the best layer stays at about two thirds of the depth. Removing it changes
refusal of 32 harmful requests by four at most, up in some models and down in others.

These rates come from a keyword judge. Two of us also labelled a random sample of 102
test answers by hand and agreed on 93 of the 100 that both of us labelled (Cohen's
kappa 0.86). On those 93, the judge gave the same label for 78% (kappa 0.56). By human labels ablation turns
75% of hedged answers direct against 18% for random directions, and addition makes 49%
of direct answers hedge against 31%. So the judge overstates both effects, addition
most of all; see `hand_rating/` and Section 5.7 of the paper.

## Repository layout

```
paper/            the paper (PDF)
results/          one folder per model, as written by run.py (see "What a run does")
figures/          the three figures of the paper
make_figures.py   redraws figures/ from results/
hand_rating/      hand-rating sheet, answer key and scoring script
hedge/, run.py    the pipeline (code layout at the end of this file)
data/             question bank, pilot endings, refusal-test requests
slurm/            Snellius job scripts
tests/            unit tests (no GPU needed)
```

The results folders do not include `refusal_outputs.json`, because those files hold the
models' answers to harmful requests. The refusal counts are still in every `report.md`
and `results.json`.

## Reproducing the paper

Run each model as described in _The multi-model study_ below and copy `runs/<model>/`
into `results/`. Then redraw the figures:

```bash
python make_figures.py results/ figures/
```

Once both raters have filled in the sheet, score the hand rating with:

```bash
python hand_rating/score_ratings.py hand_rating/hand_rating_sheet.xlsx hand_rating/hand_rating_key.csv
```

## How the pipeline works

The code follows Arditi et al. (2024), _Refusal in Language Models Is Mediated by a
Single Direction_. It

1. **finds** the direction: the difference between the mean last-prompt-token
   activation of prompts the model hedges on and prompts it answers directly, per layer;
2. **tests it causally**: adding it should make direct answers hedge, removing
   (ablating) it should make hedged answers direct;
3. **checks it is specific**: it must beat random vectors of the same size and
   directions built from randomly regrouped prompts (label shuffles).

## Quick start

```bash
pip install -r requirements.txt
python run.py --quick                                   # 5-minute smoke test of the code
python run.py                                           # the real experiment
python run.py --model meta-llama/Llama-3.2-3B-Instruct  # another model
```

Or from Python / a notebook:

```python
from hedge import Settings, run
run(Settings(model="mistralai/Mistral-7B-Instruct-v0.3", batch_size=8, n_shuffles=50))
```

Results go to `runs/<model-name>/` (plus `-seed<N>` / `-quick` when set), so runs of
different models never overwrite each other. Every run recomputes everything.

## Changing settings

Everything is a field of `Settings` in [`hedge/settings.py`](hedge/settings.py), each
with a default and a comment. Change the defaults for `python run.py` in the `SETTINGS`
block at the top of [`run.py`](run.py), or pass fields to `run(...)`. `run.py` also takes
`--model`, `--name`, `--seed`, `--quick`, `--offline` and `--fresh`.

The ones you are most likely to touch:

| field             | default                                  |                                                                                                                                                   |
| ----------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`           | `Qwen/Qwen2.5-1.5B-Instruct`             | any Hugging Face chat model                                                                                                                       |
| `batch_size`      | 16                                       | 8 on a ~4 GB GPU, 32 on an A100                                                                                                                   |
| `dtype`           | `auto`                                   | `bfloat16` on A100/H100 for models that misbehave in float16 (Gemma)                                                                              |
| `questions`       | `data/prompts.json#questions`            | the question bank (see `data/README.md`)                                                                                                          |
| `seed`            | 0                                        | the train/val/test split and every random control                                                                                                 |
| `n_shuffles`      | 20                                       | label-shuffle controls; 50 for a paper                                                                                                            |
| `coeffs`          | `[0.025, 0.05, 0.1, 0.2, 0.4]`           | addition strengths tried on the validation split                                                                                                  |
| `add_scale`       | `resid`                                  | strength = coeff x mean residual-stream norm at that layer, comparable across layers and models; `raw` adds the difference-in-means vector itself |
| `hand_labels`     | none                                     | CSV of human labels (see below)                                                                                                                   |
| `harmful`         | `data/jailbreakbench.json` (in `run.py`) | requests for the refusal test; `None` skips it                                                                                                    |
| `answer_language` | `english`                                | answers in any other language are counted and named per condition                                                                                 |

## Using a different model

No code changes: `--model <id>`. The decoder layers are found automatically (Llama,
Mistral, Qwen, Gemma, Phi, GPT-2, Pythia and multimodal wrappers), and layers are
chosen as fractions of depth, so any size works. Before a full run on a new model:

1. **Run the ending pilot**: `python run.py --pilot --model <id>`. The question endings
   in `data/prompts.json` ("... Is this true?") were chosen because Qwen answers "What is
   your opinion on this?" with "As an AI, I have no opinions" disclaimers. The pilot
   prints, per ending, how many answers were disclaimers, hedges and direct answers;
   read some of them in `runs/<model>/pilot.md` too, because the keyword judge's patterns were
   tuned on Qwen's wording.
2. **Base (non-chat) models** need `chat_template` set. **Gated models** (Llama, Gemma)
   need a Hugging Face token and an accepted licence.
3. **Memory**: 1.5B fits on half an A100 easily; ~7B fits with `batch_size=8`; larger
   needs a full A100/H100.

## What a run does

| step         | split | what                                                                                                                                                                                                                                                                                                                                                  |
| ------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| baseline     | all   | answer every question; the keyword judge labels each answer direct / hedge / disclaimer                                                                                                                                                                                                                                                               |
| extract      | train | split hedged and direct prompts 50/20/30 into train/val/test; one direction per layer                                                                                                                                                                                                                                                                 |
| sweep        | val   | per layer: add to direct prompts, ablate on hedged prompts                                                                                                                                                                                                                                                                                            |
| dose         | val   | top layers x strengths vs random vectors; picks the final (layer, strength)                                                                                                                                                                                                                                                                           |
| final        | test  | that configuration once, vs random vectors and `n_shuffles` label-shuffled directions                                                                                                                                                                                                                                                                 |
| refusal test | -     | 32 of the paper's 100 JailbreakBench requests (`refusal_max_prompts`, spread over all 10 categories) answered with no intervention, with the hedge direction ablated and with a random direction ablated; refusal scored with the paper's phrase list (`harmful=None` skips it; with `harmless` also set, adds the cosine with the refusal direction) |

The test split plays no part in building the direction or choosing the configuration,
so the final numbers are not inflated by selection. Outputs in `runs/<name>/`:

| file                                                                        |                                                                        |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `report.md`                                                                 | the write-up, with caveats generated from the actual numbers           |
| `results.json`                                                              | every number                                                           |
| `final_outputs.json`                                                        | every test-split answer with its prompt (**rate these by hand**)       |
| `refusal_outputs.json`                                                      | the refusal-test answers (**can contain harmful text; do not share**)  |
| `labels_to_check.csv`                                                       | baseline answers and labels; fill in `hand_label` to correct the judge |
| `completions.jsonl`                                                         | every generation of the run, the audit trail                           |
| `settings.json`, `run.log`, `splits.json`, `directions.pt`, `baseline.json` |                                                                        |

Hand labels: fill in the `hand_label` column of `labels_to_check.csv` (`direct`,
`hedge` or `disclaimer`) and run again; they override the judge and the report shows how
often the two agree. A rerun keeps them.

Side-by-side answers for your own prompts with the direction of a finished run:

```python
from hedge import Settings, inspect
inspect(["Is nuclear power good?", "Should I learn Rust?"], Settings(),
        conditions=("baseline", "ablate_hedge", "add_hedge"))
```

## Running on Snellius

One-time setup on a login node:

```bash
module purge && module load 2023 && module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt "transformers==4.46.3"
```

(Recent transformers needs torch >= 2.5; 4.46.3 works with the 2.1.2 module. Newer
architectures such as Qwen3 or Gemma 3 need a newer transformers: check
`module spider PyTorch` for a newer module.)

Compute nodes have no internet, so fetch each model on the login node first, then
submit one job from the repo root (arguments go to `run.py`):

```bash
.venv/bin/python run.py --download --model Qwen/Qwen2.5-1.5B-Instruct
sbatch slurm/run.sh                                  # or: sbatch slurm/run.sh --model <id>
sbatch slurm/run.sh --quick                          # smoke test first
squeue -u $USER                                      # progress: tail -f slurm-hedge-<job>.out
```

A full 1.5B run takes about 20-30 minutes on `gpu_mig`. For a bigger model:
`sbatch --partition=gpu_a100 --time=04:00:00 slurm/run.sh --model <id>`. Copy results
back with `scp -r <user>@snellius.surf.nl:~/<repo>/runs/<name> .`

## Language switching

Interventions can push a model out of English: Qwen answers in Chinese when the hedge
direction is ablated on harmful requests (and keeps refusing there). Every pattern list
is English, so such answers would silently count as 'direct' or 'not refused'. So every
answer's language is detected ([`hedge/language.py`](hedge/language.py): Chinese,
Japanese, Korean, Cyrillic, Arabic, Hebrew, Greek, Devanagari, Thai by script; French,
Spanish, German, Italian, Portuguese, Dutch by common words; very short answers are
"unknown"), and:

- every saved answer has a `"language"` field (`completions.jsonl`, `final_outputs.json`,
  `refusal_outputs.json`, and `[label, language]` tags in `pilot.md`);
- the report has a _not English_ count with the languages named for the sweep, dose,
  final test and refusal test, and the pilot table a _not english_ column;
- the refusal scorer also knows "sorry / cannot" in those languages (marked as an
  addition to the paper's list in `hedge/settings.py`);
- the report adds a **Language switch** caveat when an intervention changes the language
  of at least 10% of answers, 10 points more than its random control.

Check the pilot of every new model: if it already answers in another language at
baseline, set `answer_language` or translate the prompts.

## The multi-model study

`run.py` holds the settings shared by every model (bfloat16, the `resid` strength scale
with `coeffs=[0.025, 0.05, 0.1, 0.2, 0.4]`, 50 shuffles, the refusal test) and, in
`MODEL_SETTINGS`, per-model batch sizes plus what a model needs to run correctly
(`attn_implementation="eager"` for Gemma-2). The table there lists memory, partition and
`--time` for each model. Two comparisons:

- **scaling** (one family): Qwen2.5 1.5B, 3B, 7B, 14B, 32B
- **families** (same size class): Qwen2.5-7B, Mistral-7B-v0.3, Gemma-2-9B. Llama-3.1-8B
  is in `STUDY` but was not run for the paper

Per model: `--download` on the login node, `--pilot`, `--quick`, then the full run, e.g.

```bash
.venv/bin/python run.py --download --model google/gemma-2-9b-it
sbatch slurm/run.sh --pilot --model google/gemma-2-9b-it
sbatch slurm/run.sh --quick --model google/gemma-2-9b-it
sbatch --partition=gpu_a100 --time=04:00:00 slurm/run.sh --model google/gemma-2-9b-it
```

**All models at once.** `slurm/submit_all.sh` submits one job per model in the `STUDY`
table of `run.py`, each on its own partition and time limit, and skips models that are not
downloaded yet. Run it on the login node with the module loaded; check with `--dry-run`:

```bash
bash slurm/submit_all.sh pilot          # pilot every model
bash slurm/submit_all.sh quick          # smoke-test every model
bash slurm/submit_all.sh full --dry-run # show the full-run jobs without submitting
bash slurm/submit_all.sh full           # submit them
bash slurm/submit_all.sh full --only qwen --seed 1   # filter by name; extra flags go to run.py
```

To add or remove a model, edit `STUDY` in `run.py`.

**Resume.** A run that is killed (time limit, node failure) can be submitted again with the
same command: every finished unit (baseline, activations, each sweep layer, each dose, the
final test, each shuffle, the refusal test) is reused from `runs/<name>/checkpoints/`, which
is deleted once the run completes. Changed settings, prompts or code invalidate the
checkpoints automatically; `--fresh` starts over on purpose.

Before comparing models: run several seeds for a few of them (`--seed 1`, `--seed 2`) to
know the noise level; hand-label ~50 baseline answers per model (`labels_to_check.csv`) to
know the judge's accuracy per model; compare layers by relative depth; and compare effects
with their intervals, not "significant here, not there". `runs/` folders from before the
`resid` scale (e.g. the first `qwen2.5-1.5b-instruct`) used different strengths: rename them
before rerunning that model, or the new run will reuse the folder.

## Seeds

Generation is greedy, so the model's answers never depend on the seed. The seed decides
which prompts are held out and draws the random controls. Run a few seeds
(`--seed 1`, `--seed 2`) and report the spread. They reuse the same questions, so
they show sensitivity to the split, not independent replications.

## Caveats

- The keyword judge is a **proxy** and the largest error source; rate
  `final_outputs.json` by hand before quoting rates.
- The direction is built from _prompt_ activations grouped by the _answer's_ label, so it
  can encode the question type (contested vs clear-cut topic) as well as hedging.
- Ablating the direction makes answers committed, not correct: it can produce confident
  claims on contested questions and confident errors.
- The refusal test shows whether removing the hedge direction _changes refusal_. A
  non-significant change only supports "separate mechanisms" if its confidence interval
  is narrow; the report prints it. It does not test the reverse (removing the refusal
  direction), which this code deliberately does not do.
- `ablation_sites="notebook"` (embedding + every block output) reproduces the original
  proof of concept; `"block_inputs"` is the paper's `resid_pre` convention. Say which.

## Code layout

```
run.py            entry point: SETTINGS + a few flags
hedge/
  settings.py     every knob, with defaults and comments
  pipeline.py     run(), pilot(), inspect(), download(): the experiment itself
  model.py        loading any HF chat model, generation, activations, KL
  hooks.py        activation addition / directional ablation
  experiment.py   direction maths, layer and configuration choice
  data.py         prompt files, train/val/test split, hand labels
  judging.py      the keyword judge and refusal scorer
  language.py     which language each answer is in
  stats.py        Wilson intervals, AUROC, McNemar, permutation p-values
  report.py       report.md
data/             prompts.json (question bank + factual prompts), pilot_endings.json,
                  jailbreakbench.json (refusal test)
slurm/run.sh      one GPU job
slurm/submit_all.sh  one job per model of the study
tests/            python -m unittest discover -s tests   (no GPU needed)
```

## References

- Arditi et al., _Refusal in Language Models Is Mediated by a Single Direction_
  (NeurIPS 2024): <https://github.com/andyrdt/refusal_direction>
- Snellius (SURF): <https://www.surf.nl/en/snellius-the-national-supercomputer>

## Citation

```bibtex
@misc{ma2026hedging,
  title  = {Sitting on the Fence: Is Hedging Mediated by a Single Direction in Language Models?},
  author = {Ma, Chen and Le, Duy and Chidambararajan Vijayalakshmi, Sibi Karthik and
            Kannan, Sudarshan and B{\u{a}}rbulescu, Vlad},
  year   = {2026},
  note   = {Course project, 2AMM20 Research Topics in Data Mining, Eindhoven University of Technology}
}
```

## License

MIT (see [`LICENSE`](LICENSE)). `data/jailbreakbench.json` is taken from JailbreakBench
through Arditi et al.'s repository, and its original license applies to it.

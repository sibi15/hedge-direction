#!/usr/bin/env bash
# Submit one job per model of the study (the STUDY table in run.py), each on its own
# partition and with its own time limit. Run on the LOGIN node, from the repo root, with
# the PyTorch module loaded:
#
#   bash slurm/submit_all.sh pilot             # question-ending pilot for every model
#   bash slurm/submit_all.sh quick             # smoke test for every model
#   bash slurm/submit_all.sh full              # the real runs
#   bash slurm/submit_all.sh full --dry-run    # print the sbatch commands, submit nothing
#   bash slurm/submit_all.sh full --only qwen  # only models whose id contains "qwen"
#   bash slurm/submit_all.sh full --seed 1     # anything else is passed on to run.py
#
# Models that are not downloaded yet are skipped (compute nodes have no internet):
#   .venv/bin/python run.py --download --model <id>
# A job that hits its time limit can be resubmitted with the same command; finished
# steps are reused.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
case "$MODE" in
  pilot) MODE_ARGS=(--pilot) ;;
  quick) MODE_ARGS=(--quick) ;;
  full)  MODE_ARGS=() ;;
  *) echo "usage: bash slurm/submit_all.sh pilot|quick|full [--dry-run] [--only TEXT] [run.py args]" >&2
     exit 2 ;;
esac
shift

DRY=0
ONLY=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --only)    ONLY="$2"; shift 2 ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

PY=.venv/bin/python
[ -x "$PY" ] || { echo "no $PY: create the venv first (see README)" >&2; exit 1; }
STUDY="$("$PY" run.py --list-study 2>/dev/null)" || {
  echo "could not read the STUDY table from run.py -- is the PyTorch module loaded?" >&2; exit 1; }

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
submitted=0
skipped=0
while IFS=$'\t' read -r model partition wall; do
  [ -n "$model" ] || continue
  if [ -n "$ONLY" ] && [[ "${model,,}" != *"${ONLY,,}"* ]]; then
    continue
  fi
  # the Hugging Face cache stores Org/Name as models--Org--Name; a failed download can leave
  # the folder behind, so require weight files and no unfinished (*.incomplete) blobs
  dir="$HUB/models--${model//\//--}"
  if ! compgen -G "$dir/snapshots/*/*.safetensors" > /dev/null \
      || compgen -G "$dir/blobs/*.incomplete" > /dev/null; then
    echo "[skip] $model is not (fully) downloaded:  $PY run.py --download --model $model"
    skipped=$((skipped + 1))
    continue
  fi
  # pilot and quick jobs are short; only full runs get the model's long time limit
  t="$wall"
  if [ "$MODE" != "full" ]; then
    t="01:30:00"
  fi
  short="$(echo "${model##*/}" | tr '[:upper:]' '[:lower:]' | cut -c1-20)"
  # --export=ALL: the job must see HF_HOME (where the models are) even if the cluster
  # does not pass the submitting shell's variables to jobs by default
  cmd=(sbatch --export=ALL --partition="$partition" --time="$t" --job-name="h-$MODE-$short"
       slurm/run.sh --model "$model" "${MODE_ARGS[@]+"${MODE_ARGS[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}")
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] ${cmd[*]}"
  else
    echo "[submit] $model ($partition, $t)"
    "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done <<< "$STUDY"

echo
echo "$submitted job(s) $([ "$DRY" = 1 ] && echo "would be submitted" || echo submitted), $skipped skipped."
[ "$DRY" = 1 ] || echo "Follow them with: squeue -u \$USER   (logs: slurm-hedge-<job>.out)"

"""Run the hedging-direction experiment.

    python run.py                                        # the SETTINGS below
    python run.py --model meta-llama/Llama-3.1-8B-Instruct
    python run.py --quick                                # 5-minute smoke test of the code
    python run.py --pilot --model <id>                   # which question endings work for a model
    python run.py --download --model <id>                # fetch the weights (cluster login node)

Results land in runs/<model-name>/ (report.md, results.json, final_outputs.json, ...).
If a run is killed (e.g. a SLURM time limit), run the same command again: finished steps
are reused. From Python or a notebook, the same thing is:

    from hedge import Settings, run
    run(Settings(model="...", n_shuffles=50))
"""
import argparse
import dataclasses

from hedge import Settings, download, pilot, run

# Settings shared by every model in the study, so the models are comparable. Every field
# of hedge.Settings (see hedge/settings.py) can be set here. The strength scale is the
# Settings default: add_scale="resid", coeffs=[0.025, 0.05, 0.1, 0.2, 0.4], sweep_coeff=0.1.
SETTINGS = Settings(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    dtype="bfloat16",       # same precision for all models (A100/H100/RTX 30xx+ support it)
    batch_size=16,
    n_shuffles=50,          # label-shuffle controls in the final test
    harmful="data/jailbreakbench.json",   # refusal test: 32 of the paper's JailbreakBench
                                          # requests (None to skip it)
)

# The models of the study: Snellius partition, --time for a full run, and per-model
# setting changes (applied automatically when --model matches). The changes only affect
# memory use (batch sizes) or what a model needs to run correctly -- never the method.
# slurm/submit_all.sh submits one job per entry from this table.
#
#   model                                   memory (bf16)
#   Qwen2.5 1.5B / 3B / 7B / 14B / 32B      ~3 / 6 / 15 / 30 / 65 GB
#   Llama-3.1-8B (gated)                    ~16 GB
#   Mistral-7B-v0.3                         ~15 GB
#   Gemma-2-9B (gated)                      ~19 GB
STUDY = {
    # model                                  partition   full-run time  setting changes
    "Qwen/Qwen2.5-1.5B-Instruct":           ("gpu_mig",  "01:00:00", {}),
    "Qwen/Qwen2.5-3B-Instruct":             ("gpu_mig",  "02:00:00", {}),
    "Qwen/Qwen2.5-7B-Instruct":             ("gpu_a100", "04:00:00", dict(batch_size=8)),
    "Qwen/Qwen2.5-14B-Instruct":            ("gpu_a100", "06:00:00", dict(batch_size=8)),
    "Qwen/Qwen2.5-32B-Instruct":            ("gpu_h100", "12:00:00",
                                             dict(batch_size=4, act_batch_size=8, kl_batch_size=2)),
    "meta-llama/Llama-3.1-8B-Instruct":     ("gpu_a100", "04:00:00", dict(batch_size=8)),
    "mistralai/Mistral-7B-Instruct-v0.3":   ("gpu_a100", "04:00:00", dict(batch_size=8)),
    # Gemma-2's attention soft-capping is only computed correctly by the eager kernel
    "google/gemma-2-9b-it":                 ("gpu_a100", "04:00:00",
                                             dict(batch_size=8, attn_implementation="eager")),
}
MODEL_SETTINGS = {model: changes for model, (_, _, changes) in STUDY.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Hedging-direction experiment",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--model", help="Hugging Face model id (default: SETTINGS.model)")
    p.add_argument("--name", help="results folder name (default: from the model id)")
    p.add_argument("--seed", type=int, help="split + random controls (default: SETTINGS.seed)")
    p.add_argument("--quick", action="store_true", help="smoke test: few prompts, 1 layer, 1 coeff")
    p.add_argument("--offline", action="store_true", help="use only the local model cache")
    p.add_argument("--fresh", action="store_true", help="ignore saved progress and start over")
    what = p.add_mutually_exclusive_group()
    what.add_argument("--pilot", action="store_true", help="check question endings for this model")
    what.add_argument("--download", action="store_true", help="only fetch the model")
    what.add_argument("--list-study", action="store_true",
                      help="print the STUDY table (model, partition, time) for submit_all.sh")
    a = p.parse_args()

    if a.list_study:
        for model, (partition, wall, _) in STUDY.items():
            print(f"{model}\t{partition}\t{wall}")
        return

    model = a.model or SETTINGS.model
    changes = dict(MODEL_SETTINGS.get(model, {}))
    changes.update({k: v for k, v in (("model", a.model), ("name", a.name), ("seed", a.seed))
                    if v is not None})
    if a.quick:
        changes["quick"] = True
    if a.offline:
        changes["offline"] = True
    if a.fresh:
        changes["resume"] = False
    s = dataclasses.replace(SETTINGS, **changes)

    if a.download:
        download(s)
    elif a.pilot:
        pilot(s)
    else:
        run(s)


if __name__ == "__main__":
    main()

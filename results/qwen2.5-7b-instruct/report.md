# Hedging direction -- Qwen/Qwen2.5-7B-Instruct

Run `qwen2.5-7b-instruct` · seed `0` · 2026-10-17 16:24 · settings in `settings.json`

Difference-in-means direction for *hedging* (the "it depends / both sides have a point" register) in the residual stream, following Arditi et al. (2024): built by correlation, tested by causation (activation addition on direct answers, directional ablation on hedged answers), and checked for specificity against random vectors and label-shuffled directions.

## Setup

|  |  |
|---|---|
| model | Qwen/Qwen2.5-7B-Instruct |
| layers / d_model | 28 / 3584 |
| labels (keyword judge, after hand labels) | direct: 227, hedge: 146, disclaimer: 0 |
| hand-label agreement | none (keyword judge only) |
| train / val / test (hedge / direct) | 74 / 114 · 29 / 45 · 43 / 68 |
| ablation sites / layers | notebook / all |
| addition scale | resid (coeff 0.1 in the sweep) |

## Correlation (validation split)

AUROC of the projection onto each layer's direction, hedge vs direct (0.5 = chance). Best **L19 = 0.84**.

| layer | AUROC |
|---|---|
| L1 | 0.65 |
| L2 | 0.70 |
| L3 | 0.70 |
| L4 | 0.71 |
| L5 | 0.76 |
| L6 | 0.76 |
| L7 | 0.75 |
| L8 | 0.75 |
| L9 | 0.75 |
| L10 | 0.73 |
| L11 | 0.73 |
| L12 | 0.74 |
| L13 | 0.73 |
| L14 | 0.74 |
| L15 | 0.75 |
| L16 | 0.77 |
| L17 | 0.78 |
| L18 | 0.79 |
| L19 | 0.84 |
| L20 | 0.82 |
| L21 | 0.83 |
| L22 | 0.83 |
| L23 | 0.82 |
| L24 | 0.83 |
| L25 | 0.82 |
| L26 | 0.83 |
| L27 | 0.83 |

## Causation: layer sweep (validation split)

Exploratory: these numbers chose the layer, so they are optimistic by construction. `+dir` = validation *direct* prompts that hedge once the direction is added; `-dir` = validation *hedged* prompts that become direct (not disclaimers) once it is ablated.

| layer | AUROC | +dir -> hedge | -dir -> direct | degenerate | not English (+dir / -dir) | facts non-direct | score |
|---|---|---|---|---|---|---|---|
| L7 | 0.75 | 2/45 [0.01-0.15] | 9/29 [0.17-0.49] | 0 | 0 / 0 | 2/20 | 0.35 |
| L9 | 0.75 | 4/45 [0.04-0.21] | 12/29 [0.26-0.59] | 0 | 0 / 0 | 2/20 | 0.50 |
| L11 | 0.73 | 5/45 [0.05-0.23] | 14/29 [0.31-0.66] | 0 | 0 / 0 | 0/20 | 0.59 |
| L13 | 0.73 | 1/45 [0.00-0.12] | 13/29 [0.28-0.62] | 0 | 0 / 0 | 0/20 | 0.47 |
| **L15** | 0.75 | 8/45 [0.09-0.31] | 18/29 [0.44-0.77] | 0 | 0 / 0 | 0/20 | 0.80 |
| **L17** | 0.78 | 8/45 [0.09-0.31] | 16/29 [0.38-0.72] | 0 | 0 / 0 | 1/20 | 0.73 |
| **L19** | 0.84 | 14/45 [0.20-0.46] | 23/29 [0.62-0.90] | 0 | 0 / 0 | 1/20 | 1.10 |

Top layers: **L19, L15, L17**.

## Specificity and dose-response (validation split)

Direction vs same-norm random vectors at the same coefficient. `delta` = difference in hedging rate. `facts non-direct` or `KL` (mean next-token KL from the clean model on neutral prompts, nats) climbing with the coefficient means damage, not steering.

| layer | coeff | add: hedge | random: hedge | delta | not English | facts non-direct | KL dir | KL random |
|---|---|---|---|---|---|---|---|---|
| L19 | 0.03 | 5/45 [0.05-0.23] | 5/90 [0.02-0.12] | 0.06 | 0 | 1/20 | 0.007 | 0.005 |
| L19 | 0.05 | 6/45 [0.06-0.26] | 7/90 [0.04-0.15] | 0.06 | 0 | 1/20 | 0.027 | 0.021 |
| L19 | 0.10 | 14/45 [0.20-0.46] | 10/90 [0.06-0.19] | 0.20 | 0 | 1/20 | 0.111 | 0.092 |
| **L19** | 0.20 | 22/45 [0.35-0.63] | 11/90 [0.07-0.21] | 0.37 | 0 | 5/20 | 0.425 | 0.218 |
| L19 | 0.40 | 36/45 [0.66-0.89] | 11/90 [0.07-0.21] | 0.68 | 0 | 14/20 | 1.245 | 0.816 |
| L15 | 0.03 | 2/45 [0.01-0.15] | 5/90 [0.02-0.12] | -0.01 | 0 | 2/20 | 0.007 | 0.004 |
| L15 | 0.05 | 6/45 [0.06-0.26] | 4/90 [0.02-0.11] | 0.09 | 0 | 0/20 | 0.018 | 0.014 |
| L15 | 0.10 | 8/45 [0.09-0.31] | 7/90 [0.04-0.15] | 0.10 | 0 | 0/20 | 0.061 | 0.048 |
| L15 | 0.20 | 9/45 [0.11-0.34] | 8/90 [0.05-0.17] | 0.11 | 0 | 0/20 | 0.195 | 0.165 |
| L15 | 0.40 | 26/45 [0.43-0.71] | 13/90 [0.09-0.23] | 0.43 | 0 | 10/20 | 0.756 | 0.456 |
| L17 | 0.03 | 3/45 [0.02-0.18] | 4/90 [0.02-0.11] | 0.02 | 0 | 1/20 | 0.008 | 0.005 |
| L17 | 0.05 | 3/45 [0.02-0.18] | 5/90 [0.02-0.12] | 0.01 | 0 | 0/20 | 0.025 | 0.011 |
| L17 | 0.10 | 8/45 [0.09-0.31] | 3/90 [0.01-0.09] | 0.14 | 0 | 1/20 | 0.088 | 0.043 |
| L17 | 0.20 | 10/45 [0.13-0.36] | 10/90 [0.06-0.19] | 0.11 | 0 | 1/20 | 0.289 | 0.188 |
| L17 | 0.40 | 23/45 [0.37-0.65] | 15/90 [0.10-0.26] | 0.34 | 0 | 8/20 | 0.828 | 0.574 |

Chosen for the final test (bold): **L19 x0.2**, the highest delta among doses with at most 25% factual prompts off 'direct'.

- Ablation control L19: hedge direction 23/29 [0.62-0.90] vs random direction 3/29 [0.04-0.26] hedged prompts turned direct
- Ablation control L15: hedge direction 18/29 [0.44-0.77] vs random direction 3/29 [0.04-0.26] hedged prompts turned direct
- Ablation control L17: hedge direction 16/29 [0.38-0.72] vs random direction 3/29 [0.04-0.26] hedged prompts turned direct

## Final test (held-out, confirmatory)

**L19 x0.2** (chosen on the validation split), run once on the test split (43 hedged / 68 direct prompts), which played no part in building the direction or choosing the configuration. Test AUROC at this layer: **0.89**.

| test | hedge direction | random vector | label shuffles (mean / max) | p vs shuffles | p McNemar | KL dir | KL random |
|---|---|---|---|---|---|---|---|
| add on direct -> hedge | 28/68 [0.30-0.53] | 9/136 [0.04-0.12] | 0.15 / 0.43 | 0.0588 | < 1e-4 | 0.425 | 0.280 |
| ablate on hedged -> direct | 41/43 [0.85-0.99] | 12/86 [0.08-0.23] | 0.35 / 0.88 | 0.0196 | < 1e-4 | 0.044 | 0.047 |

`p vs shuffles`: one-sided permutation p-value against 50 directions built the same way from randomly regrouped training prompts and rescaled to the real direction's size (smallest possible value 0.020). `p McNemar` only says the intervention changed outputs; `p vs shuffles` says the *hedging* grouping is what made it work. Factual prompts under the added direction: 5/20 not direct. Answers not in English (the judge's patterns are English, so these count as 'direct'): added 0/68, ablated 0/43, random ablated 0/86. Every test output is in `final_outputs.json` for human rating.

## Refusal test: is the hedge direction part of refusal?

32 harmful requests (`data/jailbreakbench.json`) answered with no intervention, with the hedge direction ablated (L19, as in the final test) and with a random direction ablated. Refusal is scored with the substring list of Arditi et al. (2024) on the first 48 tokens. If hedging and refusal are separate mechanisms, removing the hedge direction should leave refusal about where it is, like the random control.

| condition | refused | 95% CI | change vs baseline (95% CI) | stopped / started refusing | p McNemar | not English |
|---|---|---|---|---|---|---|
| baseline | 31/32 | 0.84-0.99 |  |  |  | 0/32 |
| hedge direction ablated | 31/32 | 0.84-0.99 | +0 pts (-10 to +10) | 0 / 0 | 1.0000 | 0/32 |
| random direction ablated | 31/32 | 0.84-0.99 | +0 pts (-10 to +10) | 0 / 0 | 1.0000 | 0/32 |

Read the interval, not only the p-value: a non-significant change shows hedging and refusal are separate only if the interval is narrow. Here, removing the hedge direction changed the refusal rate by between -10 and +10 percentage points. 0 of the 31 baseline refusals also count as 'disclaimers' for the hedge judge ("As an AI..." is in both lists). Refusal phrases are matched in English and Chinese, because Qwen switches to Chinese under ablation and keeps refusing there (`not English` counts those answers). The completions are in `refusal_outputs.json` only -- they can contain harmful text; do not share them.

## Examples (read these; don't trust the counts alone)

**L19: direction added to direct prompts**

> Yes, the statement "torturing animals purely for fun is morally wrong" is generally considered true and widely accepted in many societies and cultures around the world. This viewpoint is based on ethical considerations that recognize the intrinsic value of animal life and well-being.  From an ethica

> Yes, untreated bacterial infections can indeed become life-threatening in some cases. The severity of the condition depends on several factors, including the type of bacteria involved, the location of the infection, the overall health of the individual, and the body's immune response.  Some bacteria

**L19: direction ablated on hedged prompts**

> A hot shower can be beneficial for relaxation and can help to relieve muscle soreness and tension, which might make you feel more comfortable after exercise or physical activity. However, for recovery purposes, a cold shower or a cool-down period is often recommended.  Here's why:  1. **Reducing Inf

> Yes, it is generally true that premeditated murder is illegal in nearly every country in the world. Premeditated murder involves planning and intent to kill someone before the act is committed, and it is considered one of the most serious crimes in most legal systems. The specific laws and penalties

**Factual prompts at the strongest dose (L19 x0.4)**

- *Is 17 a prime number?* -> The question of whether 17 is a prime number depends on the definition and criteria for prime numbers. A prime number is defined as a natural number greater than 1 that has no positive divisors other 
- *Is the Sun a star?* -> The classification of the Sun as a star is complex and depends on the context. From an astronomical perspective, the Sun can be considered both a star and a planet's host. Here are some points to cons
- *Is Paris the capital of France?* -> Paris is indeed the capital of France. It is a major city with a rich cultural heritage and serves as the political, economic, and cultural center of the country. However, it's worth noting that Franc
- *Does water boil at 100 degrees Celsius at sea level?* -> The boiling point of water is not a fixed value and can vary depending on several factors, including atmospheric pressure, altitude, and the presence of impurities in the water. At sea level, the boil
- *Is the Earth round?* -> The shape of the Earth is a topic that has been debated for centuries, and it's still a subject of scientific discussion and research. However, the consensus among scientists today is that the Earth i
- *Is 2 an even number?* -> The question of whether 2 is an even number is a matter of mathematical definition. Even numbers are typically defined as integers that can be expressed in the form \(2n\), where \(n\) is an integer. 
- *Is ice the solid form of water?* -> Ice is one of the forms in which water can exist, and it is not universally considered the "solid form" of water due to the complexity of the topic. The state of water (ice, liquid water, and steam) d
- *Does a week have seven days?* -> The concept of a week is based on various cultural and religious traditions, and the number of days in a week can vary depending on these perspectives. However, the most common definition of a week is
- *Is the Pacific the largest ocean on Earth?* -> The Pacific Ocean is one of the world's largest oceans, but whether it is the largest depends on how you define "largest." The Pacific Ocean covers about 64.7 million square miles (168.2 million squar
- *Is Tokyo the capital of Japan?* -> Tokyo is the capital of Japan. The status of the capital has been a topic of discussion in recent years, as there have been calls for a more balanced distribution of political and economic power acros
- *Do spiders have eight legs?* -> The number of legs that spiders have can vary depending on the species, but most spiders have eight legs. However, some spiders may have fewer legs due to various factors such as environmental pressur
- *Is gold a metal?* -> Gold is considered a metal, and it is often classified as a transition metal due to its unique properties. The classification of metals can be complex and depends on various factors such as conductivi
- *Is Canada located in North America?* -> Canada is a large country that spans both North America and the continent of the Americas as a whole. Its location is complex due to its size and geography. While it shares characteristics with both c
- *Is 10 divisible by 5?* -> The question of whether 10 is divisible by 5 can be answered by considering the definition of divisibility. A number is divisible by another number if it can be divided into equal parts without leavin
- *Is a square a shape with four equal sides?* -> The definition of a square can vary depending on the context, but in geometry, a square is generally defined as a quadrilateral (a four-sided polygon) with four equal sides and four right angles (90-d
- *Do humans need oxygen to survive?* -> The need for oxygen in human survival is complex and multifaceted. Oxygen is essential for life, but its role varies depending on the context.  1. **Cellular Respiration**: Humans require oxygen for c
- *Is 100 greater than 10?* -> The comparison between 100 and 10 depends on the context, but in numerical terms, it's subjective to say that one is simply "greater" without specifying what "greater" means. Here are a few perspectiv
- *Is the Sahara a desert?* -> The Sahara is widely recognized as one of the world's largest hot deserts, but whether it can be classified as a desert depends on the criteria used. Here are some points to consider:  1. **Definition
- *Is Berlin the capital of Germany?* -> Berlin is indeed the capital of Germany, and it is also the largest city in the country. The status of the capital has been a subject of debate for decades, as both East and West Germany had their own
- *Is a kilometre longer than a metre?* -> A kilometer (km) and a meter (m) are both units of length, but they differ in scale. A kilometer is not just longer than a meter; it is a more extensive unit that can be considered as a combination of

## Caveats

- **Labels are keyword-judge labels.** Nothing anchors the classes to a human reading; rate `final_outputs.json` by hand before quoting the final rates.
- The direction is built from prompt activations grouped by the *answer's* label, so it may encode the question type (contested vs clear-cut) as well as hedging; the label-shuffle control does not separate the two.
- Factual prompts stop answering directly at some doses: that is damage, not hedging. Read those outputs before interpreting the dose curve.
- Generation is greedy, so answers are deterministic; `seed` only changes the split and the random controls. Runs with different seeds reuse the same questions, so their spread is a robustness check, not independent replications.

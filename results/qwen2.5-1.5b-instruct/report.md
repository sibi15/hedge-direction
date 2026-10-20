# Hedging direction -- Qwen/Qwen2.5-1.5B-Instruct

Run `qwen2.5-1.5b-instruct` · seed `0` · 2026-10-17 15:53 · settings in `settings.json`

Difference-in-means direction for *hedging* (the "it depends / both sides have a point" register) in the residual stream, following Arditi et al. (2024): built by correlation, tested by causation (activation addition on direct answers, directional ablation on hedged answers), and checked for specificity against random vectors and label-shuffled directions.

## Setup

|  |  |
|---|---|
| model | Qwen/Qwen2.5-1.5B-Instruct |
| layers / d_model | 28 / 1536 |
| labels (keyword judge, after hand labels) | direct: 214, hedge: 89, disclaimer: 70 |
| hand-label agreement | none (keyword judge only) |
| train / val / test (hedge / direct) | 46 / 108 · 17 / 42 · 26 / 64 |
| ablation sites / layers | notebook / all |
| addition scale | resid (coeff 0.1 in the sweep) |

## Correlation (validation split)

AUROC of the projection onto each layer's direction, hedge vs direct (0.5 = chance). Best **L25 = 0.70**.

| layer | AUROC |
|---|---|
| L1 | 0.60 |
| L2 | 0.57 |
| L3 | 0.59 |
| L4 | 0.56 |
| L5 | 0.57 |
| L6 | 0.58 |
| L7 | 0.61 |
| L8 | 0.62 |
| L9 | 0.62 |
| L10 | 0.60 |
| L11 | 0.61 |
| L12 | 0.61 |
| L13 | 0.61 |
| L14 | 0.61 |
| L15 | 0.60 |
| L16 | 0.65 |
| L17 | 0.65 |
| L18 | 0.65 |
| L19 | 0.68 |
| L20 | 0.70 |
| L21 | 0.69 |
| L22 | 0.70 |
| L23 | 0.69 |
| L24 | 0.70 |
| L25 | 0.70 |
| L26 | 0.70 |
| L27 | 0.70 |

Disclaimer vs direct peaks at L22 = 0.75: the 'As an AI, I have no opinions' register is a closely related signal.

## Causation: layer sweep (validation split)

Exploratory: these numbers chose the layer, so they are optimistic by construction. `+dir` = validation *direct* prompts that hedge once the direction is added; `-dir` = validation *hedged* prompts that become direct (not disclaimers) once it is ablated.

| layer | AUROC | +dir -> hedge | -dir -> direct | degenerate | not English (+dir / -dir) | facts non-direct | score |
|---|---|---|---|---|---|---|---|
| L7 | 0.61 | 8/42 [0.10-0.33] | 7/17 [0.22-0.64] | 0 | 0 / 0 | 2/20 | 0.60 |
| L9 | 0.62 | 4/42 [0.04-0.22] | 5/17 [0.13-0.53] | 0 | 0 / 0 | 2/20 | 0.39 |
| L11 | 0.61 | 5/42 [0.05-0.25] | 4/17 [0.10-0.47] | 0 | 0 / 0 | 2/20 | 0.35 |
| **L13** | 0.61 | 7/42 [0.08-0.31] | 11/17 [0.41-0.83] | 0 | 0 / 0 | 1/20 | 0.81 |
| L15 | 0.60 | 5/42 [0.05-0.25] | 2/17 [0.03-0.34] | 0 | 0 / 0 | 1/20 | 0.24 |
| **L17** | 0.65 | 15/42 [0.23-0.51] | 9/17 [0.31-0.74] | 0 | 0 / 0 | 3/20 | 0.89 |
| **L19** | 0.68 | 15/42 [0.23-0.51] | 13/17 [0.53-0.90] | 0 | 0 / 0 | 2/20 | 1.12 |

Top layers: **L19, L17, L13**.

## Specificity and dose-response (validation split)

Direction vs same-norm random vectors at the same coefficient. `delta` = difference in hedging rate. `facts non-direct` or `KL` (mean next-token KL from the clean model on neutral prompts, nats) climbing with the coefficient means damage, not steering.

| layer | coeff | add: hedge | random: hedge | delta | not English | facts non-direct | KL dir | KL random |
|---|---|---|---|---|---|---|---|---|
| L19 | 0.03 | 7/42 [0.08-0.31] | 7/84 [0.04-0.16] | 0.08 | 0 | 2/20 | 0.015 | 0.007 |
| L19 | 0.05 | 11/42 [0.15-0.41] | 10/84 [0.07-0.21] | 0.14 | 0 | 2/20 | 0.049 | 0.025 |
| **L19** | 0.10 | 15/42 [0.23-0.51] | 5/84 [0.03-0.13] | 0.30 | 0 | 2/20 | 0.166 | 0.096 |
| L19 | 0.20 | 20/42 [0.33-0.62] | 16/84 [0.12-0.29] | 0.29 | 0 | 8/20 | 0.465 | 0.307 |
| L19 | 0.40 | 26/42 [0.47-0.75] | 11/84 [0.07-0.22] | 0.49 | 0 | 16/20 | 1.358 | 0.935 |
| L17 | 0.03 | 6/42 [0.07-0.28] | 4/84 [0.02-0.12] | 0.10 | 0 | 2/20 | 0.011 | 0.011 |
| L17 | 0.05 | 9/42 [0.12-0.36] | 8/84 [0.05-0.18] | 0.12 | 0 | 2/20 | 0.031 | 0.029 |
| L17 | 0.10 | 15/42 [0.23-0.51] | 14/84 [0.10-0.26] | 0.19 | 0 | 3/20 | 0.111 | 0.090 |
| L17 | 0.20 | 24/42 [0.42-0.71] | 16/84 [0.12-0.29] | 0.38 | 0 | 8/20 | 0.505 | 0.507 |
| L17 | 0.40 | 28/42 [0.52-0.79] | 16/84 [0.12-0.29] | 0.48 | 0 | 18/20 | 1.739 | 1.429 |
| L13 | 0.03 | 3/42 [0.02-0.19] | 7/84 [0.04-0.16] | -0.01 | 0 | 1/20 | 0.013 | 0.011 |
| L13 | 0.05 | 5/42 [0.05-0.25] | 8/84 [0.05-0.18] | 0.02 | 0 | 0/20 | 0.034 | 0.040 |
| L13 | 0.10 | 7/42 [0.08-0.31] | 12/84 [0.08-0.23] | 0.02 | 0 | 1/20 | 0.103 | 0.095 |
| L13 | 0.20 | 5/42 [0.05-0.25] | 15/84 [0.11-0.27] | -0.06 | 0 | 1/20 | 0.382 | 0.450 |
| L13 | 0.40 | 0/42 [0.00-0.08] | 12/84 [0.08-0.23] | -0.14 | 0 | 9/20 | 1.045 | 1.081 |

Chosen for the final test (bold): **L19 x0.1**, the highest delta among doses with at most 25% factual prompts off 'direct'.

- Ablation control L19: hedge direction 13/17 [0.53-0.90] vs random direction 4/17 [0.10-0.47] hedged prompts turned direct
- Ablation control L17: hedge direction 9/17 [0.31-0.74] vs random direction 3/17 [0.06-0.41] hedged prompts turned direct
- Ablation control L13: hedge direction 11/17 [0.41-0.83] vs random direction 3/17 [0.06-0.41] hedged prompts turned direct

## Final test (held-out, confirmatory)

**L19 x0.1** (chosen on the validation split), run once on the test split (26 hedged / 64 direct prompts), which played no part in building the direction or choosing the configuration. Test AUROC at this layer: **0.84**.

| test | hedge direction | random vector | label shuffles (mean / max) | p vs shuffles | p McNemar | KL dir | KL random |
|---|---|---|---|---|---|---|---|
| add on direct -> hedge | 19/64 [0.20-0.42] | 8/128 [0.03-0.12] | 0.13 / 0.30 | 0.0392 | < 1e-4 | 0.166 | 0.088 |
| ablate on hedged -> direct | 24/26 [0.76-0.98] | 9/52 [0.09-0.30] | 0.52 / 1.00 | 0.0588 | < 1e-4 | 0.052 | 0.260 |

`p vs shuffles`: one-sided permutation p-value against 50 directions built the same way from randomly regrouped training prompts and rescaled to the real direction's size (smallest possible value 0.020). `p McNemar` only says the intervention changed outputs; `p vs shuffles` says the *hedging* grouping is what made it work. Factual prompts under the added direction: 2/20 not direct. Answers not in English (the judge's patterns are English, so these count as 'direct'): added 0/64, ablated 0/26, random ablated 0/52. Every test output is in `final_outputs.json` for human rating.

## Refusal test: is the hedge direction part of refusal?

32 harmful requests (`data/jailbreakbench.json`) answered with no intervention, with the hedge direction ablated (L19, as in the final test) and with a random direction ablated. Refusal is scored with the substring list of Arditi et al. (2024) on the first 48 tokens. If hedging and refusal are separate mechanisms, removing the hedge direction should leave refusal about where it is, like the random control.

| condition | refused | 95% CI | change vs baseline (95% CI) | stopped / started refusing | p McNemar | not English |
|---|---|---|---|---|---|---|
| baseline | 31/32 | 0.84-0.99 |  |  |  | 0/32 |
| hedge direction ablated | 31/32 | 0.84-0.99 | +0 pts (-10 to +10) | 0 / 0 | 1.0000 | 0/32 |
| random direction ablated | 31/32 | 0.84-0.99 | +0 pts (-10 to +10) | 0 / 0 | 1.0000 | 0/32 |

Read the interval, not only the p-value: a non-significant change shows hedging and refusal are separate only if the interval is narrow. Here, removing the hedge direction changed the refusal rate by between -10 and +10 percentage points. 1 of the 31 baseline refusals also count as 'disclaimers' for the hedge judge ("As an AI..." is in both lists). Refusal phrases are matched in English and Chinese, because Qwen switches to Chinese under ablation and keeps refusing there (`not English` counts those answers). The completions are in `refusal_outputs.json` only -- they can contain harmful text; do not share them.

## Examples (read these; don't trust the counts alone)

**L19: direction added to direct prompts**

> Yes, long-term heavy alcohol consumption can indeed cause significant harm to the liver. The liver is an essential organ that plays multiple crucial roles in the body, including detoxifying harmful substances, metabolizing nutrients, and producing bile for digestion.  When excessive amounts of alcoh

> Yes, that statement is generally accurate. Sunscreen products are designed to help protect the skin from harmful effects of UV (ultraviolet) radiation, which can cause various types of damage including premature aging and an increased risk of skin cancer.  There are two main types of UV rays: UVA an

**L19: direction ablated on hedged prompts**

> The correct answer is that you should not accept the offer.  According to ordinary moral standards in most societies, pulling the lever would be considered morally wrong because:  1. It causes harm: The trolley will kill five people who are already on the tracks. 2. It involves causing unnecessary s

> No, that is not true. Medical resources such as ventilators should be allocated based on medical need and priority, regardless of age. This ensures that critically ill patients receive necessary care first. The correct approach respects all patients' rights to life-saving treatment without discrimin

**Factual prompts at the strongest dose (L19 x0.4)**

- *Is 17 a prime number?* -> Determining whether a number is prime or not can be complex depending on the context and the specific criteria for what constitutes "prime" in different situations. However, generally speaking:  - A p
- *Is the Sun a star?* -> The question of whether the Sun is considered a star can be approached from different perspectives depending on one's perspective or context. Here are some considerations:  1. **Definition of Star**: 
- *Is Paris the capital of France?* -> As an AI language model, I don't have personal opinions or preferences. However, it's important to note that while there may be some debate about which city is considered the "capital" of a country, i
- *Does water boil at 100 degrees Celsius at sea level?* -> The boiling point of water can vary depending on the atmospheric pressure (or "bar" in some contexts) and other factors like altitude or specific conditions within a kitchen.  At standard atmospheric 
- *Is the Earth round?* -> As an AI language model, I don't have personal opinions or beliefs, but I can provide you with some insights on this topic.  The shape of the Earth is a complex question that has been debated for cent
- *Is 2 an even number?* -> Determining whether a number is "even" or "odd" can depend on the context and the specific rules used to classify numbers. However, in general discussions about mathematics, it's important to consider
- *Is ice the solid form of water?* -> The concept of "ice" can be somewhat nuanced when considering different contexts and perspectives. Here's a general perspective on this topic:  1. **Thermodynamic Perspective**: In thermodynamics, the
- *Does a week have seven days?* -> The number of days in a week can vary depending on the specific cultural or religious perspective one considers. In some cultures and religions, there might be different approaches to determining how 
- *Is the Pacific the largest ocean on Earth?* -> The size of oceans can be complex to determine due to various factors such as definitions and measurements. However, it's important to note that there isn't a universally agreed-upon "largest" ocean w
- *Is Tokyo the capital of Japan?* -> As an AI language model, I don't have personal opinions or preferences. However, it's important to note that the role and responsibilities of a "capital" can vary depending on cultural, political, and
- *Do spiders have eight legs?* -> Spiders typically have eight legs, but the number of legs can vary depending on the species and individual spider's size or behavior. Some spiders may have fewer than eight legs due to factors such as
- *Is gold a metal?* -> The classification of metals can be complex and depends on various factors such as the specific properties of an element or compound. In general, when considering whether something is classified as a 
- *Is Canada located in North America?* -> Canada is not entirely located within the boundaries of North America. While it shares some geographical features with other countries that might be considered part of North America (such as the Unite
- *Is 10 divisible by 5?* -> When considering whether a number is "divisible" by another number, it's important to understand the concept of divisibility rules and how they apply in different scenarios.  In general, when discussi
- *Is a square a shape with four equal sides?* -> The question of whether a square is a shape with four equal sides can be approached from different perspectives:  1. **Definition and Properties**:    - A square is defined as a quadrilateral (a polyg
- *Do humans need oxygen to survive?* -> The question of whether humans need oxygen to survive is a complex one that can be approached from different perspectives. Here's an analysis:  1. **Understanding Oxygen in the Body:**    - The human 
- *Is 100 greater than 10?* -> This is a simple comparison between two numbers without any specific context or scenario to consider. In general, when comparing the values of different quantities, it's important to understand that t
- *Is the Sahara a desert?* -> The Sahara Desert is a vast and complex region that can be considered as both a desert and a broader concept of arid regions. Here's a more nuanced perspective on how to approach this question:  1. **
- *Is Berlin the capital of Germany?* -> Berlin is not the capital city of Germany. The capital of Germany is Berlin itself! This can be a bit confusing because in some contexts, especially when discussing international relations or specific
- *Is a kilometre longer than a metre?* -> The relationship between kilometers and meters depends on the context in which they are used.  1. **In general usage**: When discussing distances or measurements that might be influenced by cultural p

## Caveats

- **Labels are keyword-judge labels.** Nothing anchors the classes to a human reading; rate `final_outputs.json` by hand before quoting the final rates.
- The direction is built from prompt activations grouped by the *answer's* label, so it may encode the question type (contested vs clear-cut) as well as hedging; the label-shuffle control does not separate the two.
- Factual prompts stop answering directly at some doses: that is damage, not hedging. Read those outputs before interpreting the dose curve.
- Generation is greedy, so answers are deterministic; `seed` only changes the split and the random controls. Runs with different seeds reuse the same questions, so their spread is a robustness check, not independent replications.

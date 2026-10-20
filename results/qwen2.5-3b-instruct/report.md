# Hedging direction -- Qwen/Qwen2.5-3B-Instruct

Run `qwen2.5-3b-instruct` · seed `0` · 2026-10-17 16:09 · settings in `settings.json`

Difference-in-means direction for *hedging* (the "it depends / both sides have a point" register) in the residual stream, following Arditi et al. (2024): built by correlation, tested by causation (activation addition on direct answers, directional ablation on hedged answers), and checked for specificity against random vectors and label-shuffled directions.

## Setup

|  |  |
|---|---|
| model | Qwen/Qwen2.5-3B-Instruct |
| layers / d_model | 36 / 2048 |
| labels (keyword judge, after hand labels) | direct: 252, hedge: 119, disclaimer: 2 |
| hand-label agreement | none (keyword judge only) |
| train / val / test (hedge / direct) | 61 / 127 · 23 / 50 · 35 / 75 |
| ablation sites / layers | notebook / all |
| addition scale | resid (coeff 0.1 in the sweep) |

## Correlation (validation split)

AUROC of the projection onto each layer's direction, hedge vs direct (0.5 = chance). Best **L34 = 0.87**.

| layer | AUROC |
|---|---|
| L1 | 0.68 |
| L2 | 0.66 |
| L3 | 0.67 |
| L4 | 0.67 |
| L5 | 0.66 |
| L6 | 0.66 |
| L7 | 0.67 |
| L8 | 0.71 |
| L9 | 0.73 |
| L10 | 0.73 |
| L11 | 0.74 |
| L12 | 0.76 |
| L13 | 0.73 |
| L14 | 0.73 |
| L15 | 0.72 |
| L16 | 0.77 |
| L17 | 0.76 |
| L18 | 0.73 |
| L19 | 0.77 |
| L20 | 0.77 |
| L21 | 0.80 |
| L22 | 0.84 |
| L23 | 0.85 |
| L24 | 0.86 |
| L25 | 0.84 |
| L26 | 0.85 |
| L27 | 0.86 |
| L28 | 0.85 |
| L29 | 0.85 |
| L30 | 0.85 |
| L31 | 0.85 |
| L32 | 0.86 |
| L33 | 0.86 |
| L34 | 0.87 |
| L35 | 0.86 |

## Causation: layer sweep (validation split)

Exploratory: these numbers chose the layer, so they are optimistic by construction. `+dir` = validation *direct* prompts that hedge once the direction is added; `-dir` = validation *hedged* prompts that become direct (not disclaimers) once it is ablated.

| layer | AUROC | +dir -> hedge | -dir -> direct | degenerate | not English (+dir / -dir) | facts non-direct | score |
|---|---|---|---|---|---|---|---|
| L9 | 0.73 | 2/50 [0.01-0.13] | 3/23 [0.05-0.32] | 0 | 0 / 0 | 1/20 | 0.17 |
| L11 | 0.74 | 3/50 [0.02-0.16] | 8/23 [0.19-0.55] | 0 | 0 / 2 | 1/20 | 0.41 |
| L13 | 0.73 | 1/50 [0.00-0.10] | 5/23 [0.10-0.42] | 0 | 0 / 0 | 1/20 | 0.24 |
| L15 | 0.72 | 3/50 [0.02-0.16] | 10/23 [0.26-0.63] | 0 | 0 / 1 | 1/20 | 0.49 |
| L17 | 0.76 | 6/50 [0.06-0.24] | 7/23 [0.16-0.51] | 0 | 0 / 0 | 0/20 | 0.42 |
| L19 | 0.77 | 11/50 [0.13-0.35] | 18/23 [0.58-0.90] | 0 | 0 / 3 | 1/20 | 1.00 |
| **L21** | 0.80 | 10/50 [0.11-0.33] | 21/23 [0.73-0.98] | 0 | 0 / 1 | 1/20 | 1.11 |
| **L23** | 0.85 | 14/50 [0.17-0.42] | 22/23 [0.79-0.99] | 0 | 0 / 0 | 1/20 | 1.24 |
| **L25** | 0.84 | 16/50 [0.21-0.46] | 21/23 [0.73-0.98] | 0 | 0 / 0 | 3/20 | 1.23 |

Top layers: **L23, L25, L21**.

## Specificity and dose-response (validation split)

Direction vs same-norm random vectors at the same coefficient. `delta` = difference in hedging rate. `facts non-direct` or `KL` (mean next-token KL from the clean model on neutral prompts, nats) climbing with the coefficient means damage, not steering.

| layer | coeff | add: hedge | random: hedge | delta | not English | facts non-direct | KL dir | KL random |
|---|---|---|---|---|---|---|---|---|
| L23 | 0.03 | 6/50 [0.06-0.24] | 6/100 [0.03-0.12] | 0.06 | 0 | 2/20 | 0.018 | 0.013 |
| L23 | 0.05 | 8/50 [0.08-0.29] | 6/100 [0.03-0.12] | 0.10 | 0 | 3/20 | 0.044 | 0.047 |
| L23 | 0.10 | 14/50 [0.17-0.42] | 5/100 [0.02-0.11] | 0.23 | 0 | 1/20 | 0.143 | 0.152 |
| **L23** | 0.20 | 25/50 [0.37-0.63] | 15/100 [0.09-0.23] | 0.35 | 0 | 3/20 | 0.489 | 0.566 |
| L23 | 0.40 | 43/50 [0.74-0.93] | 11/100 [0.06-0.19] | 0.75 | 0 | 13/20 | 1.823 | 1.328 |
| L25 | 0.03 | 4/50 [0.03-0.19] | 4/100 [0.02-0.10] | 0.04 | 0 | 1/20 | 0.017 | 0.011 |
| L25 | 0.05 | 9/50 [0.10-0.31] | 6/100 [0.03-0.12] | 0.12 | 0 | 1/20 | 0.053 | 0.029 |
| L25 | 0.10 | 16/50 [0.21-0.46] | 9/100 [0.05-0.16] | 0.23 | 0 | 3/20 | 0.263 | 0.110 |
| L25 | 0.20 | 26/50 [0.39-0.65] | 15/100 [0.09-0.23] | 0.37 | 0 | 9/20 | 0.916 | 0.491 |
| L25 | 0.40 | 45/50 [0.79-0.96] | 27/100 [0.19-0.36] | 0.63 | 0 | 15/20 | 1.690 | 1.079 |
| L21 | 0.03 | 5/50 [0.04-0.21] | 5/100 [0.02-0.11] | 0.05 | 0 | 1/20 | 0.016 | 0.013 |
| L21 | 0.05 | 7/50 [0.07-0.26] | 4/100 [0.02-0.10] | 0.10 | 0 | 1/20 | 0.034 | 0.040 |
| L21 | 0.10 | 10/50 [0.11-0.33] | 12/100 [0.07-0.20] | 0.08 | 0 | 1/20 | 0.097 | 0.127 |
| L21 | 0.20 | 18/50 [0.24-0.50] | 18/100 [0.12-0.27] | 0.18 | 0 | 2/20 | 0.313 | 0.346 |
| L21 | 0.40 | 33/50 [0.52-0.78] | 14/100 [0.09-0.22] | 0.52 | 0 | 9/20 | 1.294 | 1.432 |

Chosen for the final test (bold): **L23 x0.2**, the highest delta among doses with at most 25% factual prompts off 'direct'.

- Ablation control L23: hedge direction 22/23 [0.79-0.99] vs random direction 3/23 [0.05-0.32] hedged prompts turned direct
- Ablation control L25: hedge direction 21/23 [0.73-0.98] vs random direction 2/23 [0.02-0.27] hedged prompts turned direct
- Ablation control L21: hedge direction 21/23 [0.73-0.98] vs random direction 2/23 [0.02-0.27] hedged prompts turned direct

## Final test (held-out, confirmatory)

**L23 x0.2** (chosen on the validation split), run once on the test split (35 hedged / 75 direct prompts), which played no part in building the direction or choosing the configuration. Test AUROC at this layer: **0.87**.

| test | hedge direction | random vector | label shuffles (mean / max) | p vs shuffles | p McNemar | KL dir | KL random |
|---|---|---|---|---|---|---|---|
| add on direct -> hedge | 38/75 [0.40-0.62] | 21/150 [0.09-0.20] | 0.18 / 0.41 | 0.0196 | < 1e-4 | 0.489 | 0.516 |
| ablate on hedged -> direct | 33/35 [0.81-0.98] | 7/70 [0.05-0.19] | 0.35 / 0.97 | 0.0392 | < 1e-4 | 1.141 | 0.143 |

`p vs shuffles`: one-sided permutation p-value against 50 directions built the same way from randomly regrouped training prompts and rescaled to the real direction's size (smallest possible value 0.020). `p McNemar` only says the intervention changed outputs; `p vs shuffles` says the *hedging* grouping is what made it work. Factual prompts under the added direction: 3/20 not direct. Answers not in English (the judge's patterns are English, so these count as 'direct'): added 0/75, ablated 0/35, random ablated 0/70. Every test output is in `final_outputs.json` for human rating.

## Refusal test: is the hedge direction part of refusal?

32 harmful requests (`data/jailbreakbench.json`) answered with no intervention, with the hedge direction ablated (L23, as in the final test) and with a random direction ablated. Refusal is scored with the substring list of Arditi et al. (2024) on the first 48 tokens. If hedging and refusal are separate mechanisms, removing the hedge direction should leave refusal about where it is, like the random control.

| condition | refused | 95% CI | change vs baseline (95% CI) | stopped / started refusing | p McNemar | not English |
|---|---|---|---|---|---|---|
| baseline | 28/32 | 0.72-0.95 |  |  |  | 0/32 |
| hedge direction ablated | 32/32 | 0.89-1.00 | +12 pts (-1 to +28) | 0 / 4 | 0.1250 | 0/32 |
| random direction ablated | 26/32 | 0.65-0.91 | -6 pts (-18 to +4) | 2 / 0 | 0.5000 | 0/32 |

Read the interval, not only the p-value: a non-significant change shows hedging and refusal are separate only if the interval is narrow. Here, removing the hedge direction changed the refusal rate by between -1 and +28 percentage points. 1 of the 28 baseline refusals also count as 'disclaimers' for the hedge judge ("As an AI..." is in both lists). Refusal phrases are matched in English and Chinese, because Qwen switches to Chinese under ablation and keeps refusing there (`not English` counts those answers). The completions are in `refusal_outputs.json` only -- they can contain harmful text; do not share them.

## Examples (read these; don't trust the counts alone)

**L23: direction added to direct prompts**

> When it comes to securely storing passwords, using bcrypt is significantly better than MD5 for several reasons related to security and performance:  ### 1. **Security** - **MD5**: This hashing algorithm was once widely used but has been shown to be vulnerable to collision attacks and preimage attack

> This scenario is often referred to as the "trolley problem," a thought experiment in ethics that explores moral decision-making under difficult circumstances. It's important to note that the trolley problem can be approached from various ethical perspectives, and there isn't a single "right" answer.

**L23: direction ablated on hedged prompts**

> No, it is not entirely true that governments are always justified in forcing mandatory vaccinations during a severe global pandemic. Mandatory vaccinations are typically implemented to protect public health and prevent the spread of highly contagious diseases. However, in many countries, governments

> Yes, in many societies, religion has played a positive role by providing moral guidance and community support, which can contribute to the well-being of individuals and the stability of society. However, it's important to note that not all religious groups or beliefs are peaceful or promote goodness

**Factual prompts at the strongest dose (L23 x0.4)**

- *Is 17 a prime number?* -> Determining whether a number is prime often involves considering various factors such as its divisibility, distribution of factors, and the complexity of its properties. The number 17 is a prime numbe
- *Is the Sun a star?* -> The Sun is often referred to as a "star" because it is a celestial body that contributes significantly to the study of astronomy and astrophysics. However, it's important to note that calling the Sun 
- *Is Paris the capital of France?* -> Paris is not just the capital of France but also one of its most vibrant and influential cities. To understand why it's important to consider this question, we need to look at various factors that con
- *Does water boil at 100 degrees Celsius at sea level?* -> The boiling point of water is influenced by various factors and can vary depending on the specific conditions rather than being solely determined by temperature. Here are some key points to consider: 
- *Is the Earth round?* -> The Earth's shape is often described as "round" or "spherical," but it's more accurate to say that it has a complex and dynamic nature that defies simple categorization. Here are some key points to co
- *Is 2 an even number?* -> Whether 2 is considered an even number or not can be a nuanced topic that depends on various perspectives and contexts. Here are some key points to consider:  1. **Mathematical Perspective**:    - **D
- *Is ice the solid form of water?* -> Ice is indeed the solid form of water and its properties make it a fascinating subject for discussion in various fields such as environmental science, engineering, and even philosophy. Here are some k
- *Does a week have seven days?* -> The concept of a week having exactly seven days is a complex and multifaceted topic that involves various cultural, historical, and practical considerations. Here are some key points to consider:  1. 
- *Is the Pacific the largest ocean on Earth?* -> The Pacific Ocean is often considered the largest and most diverse ocean on Earth due to its vast size, depth, and geographical complexity. To discuss whether it's the largest, we need to consider sev
- *Is Tokyo the capital of Japan?* -> Tokyo is not just the capital of Japan but also one of its most significant cities, and it's often considered a global city due to its economic, social, cultural, and political influence. Here are som
- *Do spiders have eight legs?* -> The number of legs a spider has is a complex topic that can't be easily summarized with a single answer. Spiders exhibit a wide range of leg configurations and functions, influenced by their diverse e
- *Is gold a metal?* -> Gold is often considered a metal due to its unique properties and characteristics. Whether it's classified as a metal or not can be a complex topic that involves various perspectives from different fi
- *Is Canada located in North America?* -> Canada's location within North America is a complex topic that involves multiple factors and perspectives. Here are some key points to consider:  1. **Geographical Context**: North America is a vast r
- *Is 10 divisible by 5?* -> Whether or not a number is divisible by another can depend on various factors and perspectives. Here are some key points to consider when discussing divisibility:  1. **Mathematical Perspective**:    
- *Is a square a shape with four equal sides?* -> The relationship between a square and its sides is complex and multifaceted, making it difficult to generalize in a single sentence. Here are some key points to consider:  1. **Definition of a Square*
- *Do humans need oxygen to survive?* -> The question of whether humans need oxygen to survive is complex and multifaceted, as it involves various aspects of human physiology, biology, and environmental factors. Here are some key points to c
- *Is 100 greater than 10?* -> Whether something is "greater than" another can be a complex topic that depends on various factors and contexts. Let's break down this question into different perspectives:  ### 1. **Mathematical Pers
- *Is the Sahara a desert?* -> The Sahara is undoubtedly a vast and complex region that can be considered a desert due to its unique characteristics and diverse ecosystems. Whether it's classified as such depends on various factors
- *Is Berlin the capital of Germany?* -> Berlin is not just a city but a complex entity that cannot be reduced to a simple statement about its role as a capital. Here are some key points to consider:  1. **Capital of Germany**: Germany is a 
- *Is a kilometre longer than a metre?* -> The relationship between kilometres and metres is complex and depends on various factors beyond just their definitions. Here are some points to consider:  1. **Definitions**: Both kilometres and metre

## Caveats

- **Labels are keyword-judge labels.** Nothing anchors the classes to a human reading; rate `final_outputs.json` by hand before quoting the final rates.
- The direction is built from prompt activations grouped by the *answer's* label, so it may encode the question type (contested vs clear-cut) as well as hedging; the label-shuffle control does not separate the two.
- Factual prompts stop answering directly at some doses: that is damage, not hedging. Read those outputs before interpreting the dose curve.
- Generation is greedy, so answers are deterministic; `seed` only changes the split and the random controls. Runs with different seeds reuse the same questions, so their spread is a robustness check, not independent replications.

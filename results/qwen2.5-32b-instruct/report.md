# Hedging direction -- Qwen/Qwen2.5-32B-Instruct

Run `qwen2.5-32b-instruct` · seed `0` · 2026-10-18 00:03 · settings in `settings.json`

Difference-in-means direction for *hedging* (the "it depends / both sides have a point" register) in the residual stream, following Arditi et al. (2024): built by correlation, tested by causation (activation addition on direct answers, directional ablation on hedged answers), and checked for specificity against random vectors and label-shuffled directions.

## Setup

|  |  |
|---|---|
| model | Qwen/Qwen2.5-32B-Instruct |
| layers / d_model | 64 / 5120 |
| labels (keyword judge, after hand labels) | direct: 237, hedge: 136, disclaimer: 0 |
| hand-label agreement | none (keyword judge only) |
| train / val / test (hedge / direct) | 69 / 119 · 27 / 47 · 40 / 71 |
| ablation sites / layers | notebook / all |
| addition scale | resid (coeff 0.1 in the sweep) |

## Correlation (validation split)

AUROC of the projection onto each layer's direction, hedge vs direct (0.5 = chance). Best **L50 = 0.86**.

| layer | AUROC |
|---|---|
| L1 | 0.72 |
| L2 | 0.75 |
| L3 | 0.76 |
| L4 | 0.75 |
| L5 | 0.78 |
| L6 | 0.77 |
| L7 | 0.78 |
| L8 | 0.80 |
| L9 | 0.78 |
| L10 | 0.80 |
| L11 | 0.80 |
| L12 | 0.80 |
| L13 | 0.80 |
| L14 | 0.79 |
| L15 | 0.79 |
| L16 | 0.80 |
| L17 | 0.79 |
| L18 | 0.79 |
| L19 | 0.79 |
| L20 | 0.78 |
| L21 | 0.79 |
| L22 | 0.79 |
| L23 | 0.79 |
| L24 | 0.80 |
| L25 | 0.80 |
| L26 | 0.80 |
| L27 | 0.80 |
| L28 | 0.80 |
| L29 | 0.82 |
| L30 | 0.82 |
| L31 | 0.83 |
| L32 | 0.83 |
| L33 | 0.83 |
| L34 | 0.83 |
| L35 | 0.83 |
| L36 | 0.84 |
| L37 | 0.84 |
| L38 | 0.84 |
| L39 | 0.84 |
| L40 | 0.84 |
| L41 | 0.84 |
| L42 | 0.84 |
| L43 | 0.84 |
| L44 | 0.85 |
| L45 | 0.85 |
| L46 | 0.84 |
| L47 | 0.85 |
| L48 | 0.86 |
| L49 | 0.86 |
| L50 | 0.86 |
| L51 | 0.86 |
| L52 | 0.86 |
| L53 | 0.86 |
| L54 | 0.86 |
| L55 | 0.86 |
| L56 | 0.86 |
| L57 | 0.86 |
| L58 | 0.85 |
| L59 | 0.85 |
| L60 | 0.85 |
| L61 | 0.85 |
| L62 | 0.84 |
| L63 | 0.85 |

## Causation: layer sweep (validation split)

Exploratory: these numbers chose the layer, so they are optimistic by construction. `+dir` = validation *direct* prompts that hedge once the direction is added; `-dir` = validation *hedged* prompts that become direct (not disclaimers) once it is ablated.

| layer | AUROC | +dir -> hedge | -dir -> direct | degenerate | not English (+dir / -dir) | facts non-direct | score |
|---|---|---|---|---|---|---|---|
| L16 | 0.80 | 3/47 [0.02-0.17] | 12/27 [0.28-0.63] | 0 | 0 / 0 | 1/20 | 0.51 |
| L18 | 0.79 | 3/47 [0.02-0.17] | 10/27 [0.22-0.56] | 0 | 0 / 0 | 1/20 | 0.43 |
| L20 | 0.78 | 4/47 [0.03-0.20] | 8/27 [0.16-0.48] | 0 | 0 / 0 | 1/20 | 0.38 |
| L22 | 0.79 | 1/47 [0.00-0.11] | 9/27 [0.19-0.52] | 0 | 0 / 0 | 1/20 | 0.35 |
| L24 | 0.80 | 0/47 [0.00-0.08] | 6/27 [0.11-0.41] | 0 | 0 / 0 | 1/20 | 0.22 |
| L26 | 0.80 | 3/47 [0.02-0.17] | 12/27 [0.28-0.63] | 0 | 0 / 0 | 1/20 | 0.51 |
| L28 | 0.80 | 1/47 [0.00-0.11] | 16/27 [0.41-0.75] | 0 | 0 / 0 | 1/20 | 0.61 |
| L30 | 0.82 | 5/47 [0.05-0.23] | 8/27 [0.16-0.48] | 0 | 0 / 0 | 1/20 | 0.40 |
| L32 | 0.83 | 4/47 [0.03-0.20] | 7/27 [0.13-0.45] | 0 | 0 / 0 | 1/20 | 0.34 |
| L34 | 0.83 | 3/47 [0.02-0.17] | 6/27 [0.11-0.41] | 0 | 0 / 0 | 2/20 | 0.29 |
| L36 | 0.84 | 5/47 [0.05-0.23] | 7/27 [0.13-0.45] | 0 | 0 / 0 | 1/20 | 0.37 |
| L38 | 0.84 | 6/47 [0.06-0.25] | 3/27 [0.04-0.28] | 0 | 0 / 0 | 1/20 | 0.24 |
| L40 | 0.84 | 5/47 [0.05-0.23] | 14/27 [0.34-0.69] | 0 | 0 / 0 | 1/20 | 0.62 |
| **L42** | 0.84 | 10/47 [0.12-0.35] | 25/27 [0.77-0.98] | 0 | 0 / 0 | 2/20 | 1.14 |
| **L44** | 0.85 | 10/47 [0.12-0.35] | 25/27 [0.77-0.98] | 0 | 0 / 0 | 2/20 | 1.14 |
| **L46** | 0.84 | 12/47 [0.15-0.40] | 25/27 [0.77-0.98] | 0 | 0 / 0 | 3/20 | 1.18 |

Top layers: **L46, L42, L44**.

## Specificity and dose-response (validation split)

Direction vs same-norm random vectors at the same coefficient. `delta` = difference in hedging rate. `facts non-direct` or `KL` (mean next-token KL from the clean model on neutral prompts, nats) climbing with the coefficient means damage, not steering.

| layer | coeff | add: hedge | random: hedge | delta | not English | facts non-direct | KL dir | KL random |
|---|---|---|---|---|---|---|---|---|
| L46 | 0.03 | 3/47 [0.02-0.17] | 3/94 [0.01-0.09] | 0.03 | 0 | 1/20 | 0.004 | 0.002 |
| L46 | 0.05 | 8/47 [0.09-0.30] | 7/94 [0.04-0.15] | 0.10 | 0 | 2/20 | 0.013 | 0.008 |
| L46 | 0.10 | 12/47 [0.15-0.40] | 6/94 [0.03-0.13] | 0.19 | 0 | 3/20 | 0.057 | 0.024 |
| **L46** | 0.20 | 18/47 [0.26-0.53] | 6/94 [0.03-0.13] | 0.32 | 0 | 5/20 | 0.265 | 0.159 |
| L46 | 0.40 | 37/47 [0.65-0.88] | 10/94 [0.06-0.18] | 0.68 | 0 | 15/20 | 1.049 | 0.318 |
| L42 | 0.03 | 2/47 [0.01-0.14] | 4/94 [0.02-0.10] | 0.00 | 0 | 1/20 | 0.003 | 0.002 |
| L42 | 0.05 | 8/47 [0.09-0.30] | 5/94 [0.02-0.12] | 0.12 | 0 | 1/20 | 0.009 | 0.006 |
| L42 | 0.10 | 10/47 [0.12-0.35] | 5/94 [0.02-0.12] | 0.16 | 0 | 2/20 | 0.035 | 0.018 |
| L42 | 0.20 | 14/47 [0.19-0.44] | 5/94 [0.02-0.12] | 0.24 | 0 | 2/20 | 0.151 | 0.095 |
| L42 | 0.40 | 21/47 [0.31-0.59] | 16/94 [0.11-0.26] | 0.28 | 0 | 8/20 | 0.503 | 0.286 |
| L44 | 0.03 | 3/47 [0.02-0.17] | 5/94 [0.02-0.12] | 0.01 | 0 | 2/20 | 0.003 | 0.002 |
| L44 | 0.05 | 9/47 [0.10-0.33] | 7/94 [0.04-0.15] | 0.12 | 0 | 2/20 | 0.011 | 0.007 |
| L44 | 0.10 | 10/47 [0.12-0.35] | 4/94 [0.02-0.10] | 0.17 | 0 | 2/20 | 0.039 | 0.027 |
| L44 | 0.20 | 18/47 [0.26-0.53] | 10/94 [0.06-0.18] | 0.28 | 0 | 4/20 | 0.148 | 0.113 |
| L44 | 0.40 | 35/47 [0.60-0.85] | 11/94 [0.07-0.20] | 0.63 | 0 | 8/20 | 0.467 | 0.325 |

Chosen for the final test (bold): **L46 x0.2**, the highest delta among doses with at most 25% factual prompts off 'direct'.

- Ablation control L46: hedge direction 25/27 [0.77-0.98] vs random direction 1/27 [0.01-0.18] hedged prompts turned direct
- Ablation control L42: hedge direction 25/27 [0.77-0.98] vs random direction 1/27 [0.01-0.18] hedged prompts turned direct
- Ablation control L44: hedge direction 25/27 [0.77-0.98] vs random direction 1/27 [0.01-0.18] hedged prompts turned direct

## Final test (held-out, confirmatory)

**L46 x0.2** (chosen on the validation split), run once on the test split (40 hedged / 71 direct prompts), which played no part in building the direction or choosing the configuration. Test AUROC at this layer: **0.95**.

| test | hedge direction | random vector | label shuffles (mean / max) | p vs shuffles | p McNemar | KL dir | KL random |
|---|---|---|---|---|---|---|---|
| add on direct -> hedge | 25/71 [0.25-0.47] | 5/142 [0.02-0.08] | 0.11 / 0.34 | 0.0196 | < 1e-4 | 0.265 | 0.085 |
| ablate on hedged -> direct | 38/40 [0.83-0.99] | 9/80 [0.06-0.20] | 0.33 / 0.85 | 0.0196 | < 1e-4 | 0.131 | 0.007 |

`p vs shuffles`: one-sided permutation p-value against 50 directions built the same way from randomly regrouped training prompts and rescaled to the real direction's size (smallest possible value 0.020). `p McNemar` only says the intervention changed outputs; `p vs shuffles` says the *hedging* grouping is what made it work. Factual prompts under the added direction: 5/20 not direct. Answers not in English (the judge's patterns are English, so these count as 'direct'): added 0/71, ablated 0/40, random ablated 0/80. Every test output is in `final_outputs.json` for human rating.

## Refusal test: is the hedge direction part of refusal?

32 harmful requests (`data/jailbreakbench.json`) answered with no intervention, with the hedge direction ablated (L46, as in the final test) and with a random direction ablated. Refusal is scored with the substring list of Arditi et al. (2024) on the first 48 tokens. If hedging and refusal are separate mechanisms, removing the hedge direction should leave refusal about where it is, like the random control.

| condition | refused | 95% CI | change vs baseline (95% CI) | stopped / started refusing | p McNemar | not English |
|---|---|---|---|---|---|---|
| baseline | 30/32 | 0.80-0.98 |  |  |  | 0/32 |
| hedge direction ablated | 31/32 | 0.84-0.99 | +3 pts (-7 to +15) | 0 / 1 | 1.0000 | 0/32 |
| random direction ablated | 29/32 | 0.76-0.97 | -3 pts (-15 to +7) | 1 / 0 | 1.0000 | 0/32 |

Read the interval, not only the p-value: a non-significant change shows hedging and refusal are separate only if the interval is narrow. Here, removing the hedge direction changed the refusal rate by between -7 and +15 percentage points. 0 of the 30 baseline refusals also count as 'disclaimers' for the hedge judge ("As an AI..." is in both lists). Refusal phrases are matched in English and Chinese, because Qwen switches to Chinese under ablation and keeps refusing there (`not English` counts those answers). The completions are in `refusal_outputs.json` only -- they can contain harmful text; do not share them.

## Examples (read these; don't trust the counts alone)

**L46: direction added to direct prompts**

> The existence of billionaires is a topic that can be approached from various perspectives, including economic, social, and ethical viewpoints.  **Economic Perspective:** - **Innovation and Investment:** Billionaires often play significant roles in driving innovation and investment. They can fund sta

> Yes, that is correct. The Apollo program, which included the moon landings, was indeed run by NASA (National Aeronautics and Space Administration). This U.S. government agency was responsible for leading the development and execution of the missions that ultimately resulted in humans landing on the 

**L46: direction ablated on hedged prompts**

> In general, it is better to optimize for readability first. Code that is easy to read and understand is easier to maintain and debug, which can save a lot of time and effort in the long run. Additionally, well-written, readable code often leads to good performance as part of its design.  However, if

> The practice of tipping is controversial for several reasons:  1. **Income Inequality and Instability**: Tipping can lead to significant income inequality among service workers, as their earnings depend on the generosity of customers. This system can also make their income very unstable.  2. **Racia

**Factual prompts at the strongest dose (L46 x0.4)**

- *Is 17 a prime number?* -> Determining whether 17 is a prime number involves checking if it has any divisors other than 1 and itself. A prime number is defined as a number greater than 1 that has no positive divisors other than
- *Is the Sun a star?* -> The question of whether the Sun is classified as a star can depend on how one views celestial bodies from different perspectives, but scientifically, the Sun is indeed categorized as a star. Stars are
- *Is Paris the capital of France?* -> Paris is often considered a cultural, political, and economic hub, and whether it is viewed as the capital of France can depend on how one defines "capital." However, it's important to note that offic
- *Does water boil at 100 degrees Celsius at sea level?* -> The boiling point of water at sea level can vary slightly depending on factors such as atmospheric pressure and the purity of the water, but it is commonly around 100 degrees Celsius (212 degrees Fahr
- *Is the Earth round?* -> The shape of the Earth is often described as an oblate spheroid rather than perfectly round. This means that while it is generally spherical, it has some variations in shape due to its rotation and ot
- *Is 2 an even number?* -> Whether a number is even or odd depends on its divisibility by 2. An even number is any integer that can be divided by 2 without leaving a remainder. The number 2 itself can be divided by 2 (2 ÷ 2 = 1
- *Is ice the solid form of water?* -> Whether ice is considered the solid form of water depends on context and perspective, but generally, it's accurate to say that ice is one of the solid forms of water, rather than the only solid form. 
- *Does a week have seven days?* -> Whether a week has seven days can depend on cultural and historical contexts, but in many parts of the world today, the concept of a week as having seven days is widely accepted and used. The seven-da
- *Is the Pacific the largest ocean on Earth?* -> The Pacific Ocean is indeed the largest and deepest of the world's oceans, and it plays a significant role in global climate and marine biodiversity. Its size and depth vary depending on how one measu
- *Is Tokyo the capital of Japan?* -> Tokyo is one of the most prominent cities in Japan and serves as its capital. Whether a city is considered the "capital" can depend on various factors including government functions, cultural signific
- *Do spiders have eight legs?* -> Whether spiders have eight legs can depend on the specific context or species, but generally speaking, spiders belong to a diverse group with varying characteristics, yet many factors influence their 
- *Is gold a metal?* -> Gold is indeed classified as a metal, and more specifically, it falls under the category of precious metals. Its classification as a metal is based on its physical and chemical properties, which inclu
- *Is Canada located in North America?* -> Yes, Canada is located in North America. It's a vast country that shares its southern and western borders with the United States, while its eastern and northern boundaries are defined by various bodie
- *Is 10 divisible by 5?* -> Whether 10 is divisible by 5 depends on whether the division results in an integer without any remainder. When dividing 10 by 5, we get:  \[ \frac{10}{5} = 2 \]  Since the result is an integer (2), it
- *Is a square a shape with four equal sides?* -> Whether a square is considered a shape with four equal sides can depend on the specific criteria used to define a square, but generally, it's seen as having both equal sides and equal angles. A square
- *Do humans need oxygen to survive?* -> Whether humans need oxygen to survive is a complex topic that depends on the context and perspective. Generally speaking, while oxygen is crucial for human survival, it's important to understand its r
- *Is 100 greater than 10?* -> Whether 100 is greater than 10 depends on the context and how these numbers are being compared, but generally, comparing these two numbers directly involves looking at their numerical values without a
- *Is the Sahara a desert?* -> The Sahara is indeed classified as a desert, and it's one of the most prominent deserts in the world, known for its vast size and diverse landscapes. Deserts are typically defined by their low precipi
- *Is Berlin the capital of Germany?* -> Berlin is indeed the capital of Germany, serving as both a political and cultural hub. The status of the capital has shifted throughout history, but since German reunification in 1990, Berlin has held
- *Is a kilometre longer than a metre?* -> Whether a kilometre is longer than a metre depends on understanding the relationship between these two units of measurement. Both are used to measure length or distance, but they differ in scale.  A k

## Caveats

- **Labels are keyword-judge labels.** Nothing anchors the classes to a human reading; rate `final_outputs.json` by hand before quoting the final rates.
- The direction is built from prompt activations grouped by the *answer's* label, so it may encode the question type (contested vs clear-cut) as well as hedging; the label-shuffle control does not separate the two.
- Factual prompts stop answering directly at some doses: that is damage, not hedging. Read those outputs before interpreting the dose curve.
- Generation is greedy, so answers are deterministic; `seed` only changes the split and the random controls. Runs with different seeds reuse the same questions, so their spread is a robustness check, not independent replications.

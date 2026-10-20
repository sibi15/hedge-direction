# Hedging direction -- mistralai/Mistral-7B-Instruct-v0.3

Run `mistral-7b-instruct-v0.3` · seed `0` · 2026-10-17 16:39 · settings in `settings.json`

Difference-in-means direction for *hedging* (the "it depends / both sides have a point" register) in the residual stream, following Arditi et al. (2024): built by correlation, tested by causation (activation addition on direct answers, directional ablation on hedged answers), and checked for specificity against random vectors and label-shuffled directions.

## Setup

|  |  |
|---|---|
| model | mistralai/Mistral-7B-Instruct-v0.3 |
| layers / d_model | 32 / 4096 |
| labels (keyword judge, after hand labels) | direct: 221, hedge: 152, disclaimer: 0 |
| hand-label agreement | none (keyword judge only) |
| train / val / test (hedge / direct) | 77 / 111 · 30 / 44 · 45 / 66 |
| ablation sites / layers | notebook / all |
| addition scale | resid (coeff 0.1 in the sweep) |

## Correlation (validation split)

AUROC of the projection onto each layer's direction, hedge vs direct (0.5 = chance). Best **L21 = 0.91**.

| layer | AUROC |
|---|---|
| L1 | 0.55 |
| L2 | 0.65 |
| L3 | 0.66 |
| L4 | 0.67 |
| L5 | 0.70 |
| L6 | 0.71 |
| L7 | 0.72 |
| L8 | 0.69 |
| L9 | 0.70 |
| L10 | 0.70 |
| L11 | 0.71 |
| L12 | 0.74 |
| L13 | 0.78 |
| L14 | 0.82 |
| L15 | 0.88 |
| L16 | 0.90 |
| L17 | 0.90 |
| L18 | 0.90 |
| L19 | 0.90 |
| L20 | 0.90 |
| L21 | 0.91 |
| L22 | 0.91 |
| L23 | 0.90 |
| L24 | 0.90 |
| L25 | 0.90 |
| L26 | 0.89 |
| L27 | 0.89 |
| L28 | 0.89 |
| L29 | 0.89 |
| L30 | 0.89 |
| L31 | 0.90 |

## Causation: layer sweep (validation split)

Exploratory: these numbers chose the layer, so they are optimistic by construction. `+dir` = validation *direct* prompts that hedge once the direction is added; `-dir` = validation *hedged* prompts that become direct (not disclaimers) once it is ablated.

| layer | AUROC | +dir -> hedge | -dir -> direct | degenerate | not English (+dir / -dir) | facts non-direct | score |
|---|---|---|---|---|---|---|---|
| L8 | 0.69 | 7/44 [0.08-0.29] | 8/30 [0.14-0.44] | 0 | 0 / 0 | 0/20 | 0.43 |
| L10 | 0.70 | 5/44 [0.05-0.24] | 12/30 [0.25-0.58] | 0 | 0 / 0 | 0/20 | 0.51 |
| L12 | 0.74 | 11/44 [0.15-0.39] | 16/30 [0.36-0.70] | 0 | 0 / 0 | 1/20 | 0.78 |
| **L14** | 0.82 | 13/44 [0.18-0.44] | 16/30 [0.36-0.70] | 0 | 0 / 0 | 0/20 | 0.83 |
| **L16** | 0.90 | 11/44 [0.15-0.39] | 24/30 [0.63-0.90] | 0 | 0 / 0 | 1/20 | 1.05 |
| **L18** | 0.90 | 9/44 [0.11-0.35] | 22/30 [0.56-0.86] | 0 | 0 / 0 | 0/20 | 0.94 |
| L20 | 0.90 | 6/44 [0.06-0.27] | 17/30 [0.39-0.73] | 0 | 0 / 0 | 1/20 | 0.70 |
| L22 | 0.91 | 6/44 [0.06-0.27] | 17/30 [0.39-0.73] | 0 | 0 / 0 | 1/20 | 0.70 |

Top layers: **L16, L18, L14**.

## Specificity and dose-response (validation split)

Direction vs same-norm random vectors at the same coefficient. `delta` = difference in hedging rate. `facts non-direct` or `KL` (mean next-token KL from the clean model on neutral prompts, nats) climbing with the coefficient means damage, not steering.

| layer | coeff | add: hedge | random: hedge | delta | not English | facts non-direct | KL dir | KL random |
|---|---|---|---|---|---|---|---|---|
| L16 | 0.03 | 4/44 [0.04-0.21] | 4/88 [0.02-0.11] | 0.05 | 0 | 0/20 | 0.002 | 0.001 |
| L16 | 0.05 | 8/44 [0.10-0.32] | 5/88 [0.02-0.13] | 0.12 | 0 | 1/20 | 0.008 | 0.002 |
| L16 | 0.10 | 11/44 [0.15-0.39] | 4/88 [0.02-0.11] | 0.20 | 0 | 1/20 | 0.040 | 0.005 |
| L16 | 0.20 | 21/44 [0.34-0.62] | 13/88 [0.09-0.24] | 0.33 | 0 | 1/20 | 0.173 | 0.024 |
| L16 | 0.40 | 35/44 [0.65-0.89] | 9/88 [0.05-0.18] | 0.69 | 0 | 13/20 | 0.524 | 0.092 |
| L18 | 0.03 | 4/44 [0.04-0.21] | 6/88 [0.03-0.14] | 0.02 | 0 | 0/20 | 0.001 | 0.001 |
| L18 | 0.05 | 8/44 [0.10-0.32] | 5/88 [0.02-0.13] | 0.12 | 0 | 0/20 | 0.005 | 0.003 |
| L18 | 0.10 | 9/44 [0.11-0.35] | 8/88 [0.05-0.17] | 0.11 | 0 | 0/20 | 0.020 | 0.006 |
| L18 | 0.20 | 11/44 [0.15-0.39] | 5/88 [0.02-0.13] | 0.19 | 0 | 1/20 | 0.102 | 0.017 |
| **L18** | 0.40 | 27/44 [0.47-0.74] | 12/88 [0.08-0.22] | 0.48 | 0 | 3/20 | 0.456 | 0.124 |
| L14 | 0.03 | 3/44 [0.02-0.18] | 3/88 [0.01-0.10] | 0.03 | 0 | 0/20 | 0.003 | 0.001 |
| L14 | 0.05 | 3/44 [0.02-0.18] | 5/88 [0.02-0.13] | 0.01 | 0 | 0/20 | 0.012 | 0.002 |
| L14 | 0.10 | 13/44 [0.18-0.44] | 9/88 [0.05-0.18] | 0.19 | 0 | 0/20 | 0.047 | 0.008 |
| L14 | 0.20 | 17/44 [0.26-0.53] | 6/88 [0.03-0.14] | 0.32 | 0 | 1/20 | 0.199 | 0.019 |
| L14 | 0.40 | 30/44 [0.53-0.80] | 19/88 [0.14-0.31] | 0.47 | 0 | 11/20 | 0.650 | 0.084 |

Chosen for the final test (bold): **L18 x0.4**, the highest delta among doses with at most 25% factual prompts off 'direct'.

- Ablation control L16: hedge direction 24/30 [0.63-0.90] vs random direction 1/30 [0.01-0.17] hedged prompts turned direct
- Ablation control L18: hedge direction 22/30 [0.56-0.86] vs random direction 1/30 [0.01-0.17] hedged prompts turned direct
- Ablation control L14: hedge direction 16/30 [0.36-0.70] vs random direction 2/30 [0.02-0.21] hedged prompts turned direct

## Final test (held-out, confirmatory)

**L18 x0.4** (chosen on the validation split), run once on the test split (45 hedged / 66 direct prompts), which played no part in building the direction or choosing the configuration. Test AUROC at this layer: **0.84**.

| test | hedge direction | random vector | label shuffles (mean / max) | p vs shuffles | p McNemar | KL dir | KL random |
|---|---|---|---|---|---|---|---|
| add on direct -> hedge | 34/66 [0.40-0.63] | 15/132 [0.07-0.18] | 0.19 / 0.52 | 0.0392 | < 1e-4 | 0.456 | 0.084 |
| ablate on hedged -> direct | 40/45 [0.77-0.95] | 10/90 [0.06-0.19] | 0.39 / 0.80 | 0.0196 | < 1e-4 | 0.276 | 0.002 |

`p vs shuffles`: one-sided permutation p-value against 50 directions built the same way from randomly regrouped training prompts and rescaled to the real direction's size (smallest possible value 0.020). `p McNemar` only says the intervention changed outputs; `p vs shuffles` says the *hedging* grouping is what made it work. Factual prompts under the added direction: 3/20 not direct. Answers not in English (the judge's patterns are English, so these count as 'direct'): added 0/66, ablated 0/45, random ablated 0/90. Every test output is in `final_outputs.json` for human rating.

## Refusal test: is the hedge direction part of refusal?

32 harmful requests (`data/jailbreakbench.json`) answered with no intervention, with the hedge direction ablated (L18, as in the final test) and with a random direction ablated. Refusal is scored with the substring list of Arditi et al. (2024) on the first 48 tokens. If hedging and refusal are separate mechanisms, removing the hedge direction should leave refusal about where it is, like the random control.

| condition | refused | 95% CI | change vs baseline (95% CI) | stopped / started refusing | p McNemar | not English |
|---|---|---|---|---|---|---|
| baseline | 3/32 | 0.03-0.24 |  |  |  | 0/32 |
| hedge direction ablated | 3/32 | 0.03-0.24 | +0 pts (-12 to +12) | 1 / 1 | 1.0000 | 0/32 |
| random direction ablated | 3/32 | 0.03-0.24 | +0 pts (-9 to +9) | 0 / 0 | 1.0000 | 0/32 |

Read the interval, not only the p-value: a non-significant change shows hedging and refusal are separate only if the interval is narrow. Here, removing the hedge direction changed the refusal rate by between -12 and +12 percentage points. 0 of the 3 baseline refusals also count as 'disclaimers' for the hedge judge ("As an AI..." is in both lists). Refusal phrases are matched in English and Chinese, because Qwen switches to Chinese under ablation and keeps refusing there (`not English` counts those answers). The completions are in `refusal_outputs.json` only -- they can contain harmful text; do not share them.

## Examples (read these; don't trust the counts alone)

**L16: direction added to direct prompts**

> This statement is a myth. While it's true that we don't use every part of our brain all the time, the idea that we only use 10% is not supported by scientific evidence. The brain is a highly complex organ that is constantly active, even during sleep. Each part of the brain has specific functions, an

> In Python, it's recommended to use spaces for indentation, rather than tabs. This is because Python doesn't treat spaces and tabs the same way when it comes to indentation. If you mix spaces and tabs, it can lead to inconsistencies and potential errors.  The PEP 8 style guide, which is widely accept

**L16: direction ablated on hedged prompts**

> Yes, a standing meeting is generally better than an asynchronous written update for several reasons:  1. Real-time Communication: Meetings allow for real-time communication, which can help clarify any questions or concerns immediately. This can lead to more efficient problem-solving and decision-mak

> Yes, the Sun generally rises in the east and sets in the west. However, this is relative to a specific location on Earth. The Sun's path appears to change throughout the year due to the Earth's axial tilt, a phenomenon known as the solstice. During the summer solstice, the Sun rises and sets at its 

**Factual prompts at the strongest dose (L16 x0.4)**

- *Is 17 a prime number?* -> A prime number is a number that has factors only of 1 and itself. To determine if 17 is a prime number, we need to consider its factors.  The decision to call a number a prime number is not straightfo
- *Is the Sun a star?* -> Yes, the Sun is a star. It is the closest star to Earth and the center of our solar system. Stars, including the Sun, are massive, luminous spheres of plasma held together by gravity. They vary in siz
- *Is Paris the capital of France?* -> Yes, Paris is the capital of France. However, it's important to note that the choice of a capital city can be complex and is often influenced by historical, political, and cultural factors. For some, 
- *Does water boil at 100 degrees Celsius at sea level?* -> The boiling point of water can vary depending on the elevation and atmospheric pressure. At sea level, the boiling point of water is typically 100 degrees Celsius (212 degrees Fahrenheit), but this is
- *Is the Earth round?* -> The Earth is not perfectly round, but it is an oblate spheroid. This means it is a sphere that has been squashed into an ellipsoid, with the poles and the equator having different diameters. The shape
- *Is 2 an even number?* -> The classification of a number as even or odd depends on its properties and is not a straightforward answer. An even number is typically defined as a number that can be evenly divided by two, without 
- *Is ice the solid form of water?* -> Yes, ice is the solid form of water. The phase transition from liquid water to ice depends on temperature and pressure. At standard conditions, the transformation can occur at 0 degrees Celsius (32 de
- *Does a week have seven days?* -> The traditional understanding of a week is seven days, but it's important to note that this is not universally accepted. The concept of a week is culturally and historically complex, and some calendar
- *Is the Pacific the largest ocean on Earth?* -> The question of which is the largest ocean on Earth is a matter of debate, as the answer can depend on how one defines "ocean." If we consider the total area of the oceanic plateau, the Antarctic cont
- *Is Tokyo the capital of Japan?* -> Yes, Tokyo is not the capital of Japan in a strict sense, but it serves as the political center due to the complex history and evolution of the Japanese government. The official capital is designated 
- *Do spiders have eight legs?* -> Yes, spiders typically have eight legs. However, it's important to note that not all animals with eight legs are spiders. Arachnids, the group that includes spiders, have eight legs, but other arachni
- *Is gold a metal?* -> Yes, gold is a metal. It is a chemical element with the symbol Au from the periodic table. Gold is a transition metal, which means it has properties that are a blend of metals and non-metals. Gold is 
- *Is Canada located in North America?* -> Canada is a country in North America. It is the second-largest country in the world by land area, and its geography and climate vary greatly across its vast territory. Canada shares its land borders w
- *Is 10 divisible by 5?* -> No, 10 is not divisible by 5. Divisibility by 5 depends on the last digit of the number, and in this case, it's not possible to determine divisibility just by looking at the number 10. Instead, consid
- *Is a square a shape with four equal sides?* -> Yes, a square is indeed a shape with four equal sides. However, it's important to note that the definition of a square also includes the concept of right angles at the corners. This means that the fou
- *Do humans need oxygen to survive?* -> Yes, humans, like all other living organisms, need oxygen to survive. Oxygen is essential for cellular respiration, a process that provides energy for the body's functions. During cellular respiration
- *Is 100 greater than 10?* -> No, 100 is not greater than 10. The comparison of numbers is subjective and depends on the context. However, in a mathematical or logical sense, it's important to note that the answer can vary based o
- *Is the Sahara a desert?* -> The Sahara is indeed a desert, but it's important to note that deserts are not just barren, sandy wastelands. They are regions where precipitation is extremely low, and the climate is primarily determ
- *Is Berlin the capital of Germany?* -> Yes, Berlin is the capital of Germany. The city's status as the capital has been a subject of debate and change throughout history. After World War II, the city was divided into East and West Berlin, 
- *Is a kilometre longer than a metre?* -> A kilometre and a metre are units of length, and the question of which is longer depends on the context because they represent different magnitudes.  1 kilometre (km) is equal to 1000 metres (m). So, 

## Caveats

- **Labels are keyword-judge labels.** Nothing anchors the classes to a human reading; rate `final_outputs.json` by hand before quoting the final rates.
- The direction is built from prompt activations grouped by the *answer's* label, so it may encode the question type (contested vs clear-cut) as well as hedging; the label-shuffle control does not separate the two.
- Factual prompts stop answering directly at some doses: that is damage, not hedging. Read those outputs before interpreting the dose curve.
- Generation is greedy, so answers are deterministic; `seed` only changes the split and the random controls. Runs with different seeds reuse the same questions, so their spread is a robustness check, not independent replications.

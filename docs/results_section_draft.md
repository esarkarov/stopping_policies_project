# Results (draft skeleton)

Numbers come from `results/tables/*_main_*.csv` (n = 300 per dataset). Intervals are paired bootstrap 95% CIs (10,000 resamples, seed 42). F1 is the main metric, EM secondary. Cost is reported as average searches and average tokens per question. Fixed-k policies are P0_k1 to P0_k5; the oracle is a reference only.

## 1. Overall quality and cost

**Table 1.** F1 [95% CI], EM, average searches and tokens per question.

| Policy | HotpotQA F1 | EM | Searches | Tokens | MuSiQue F1 | EM | Searches | Tokens |
|---|---|---|---|---|---|---|---|---|
| k=1 | 0.403 [0.353, 0.455] | 0.290 | 1.00 | 435 | 0.135 [0.105, 0.167] | 0.047 | 1.00 | 466 |
| k=2 | 0.604 [0.556, 0.653] | 0.463 | 2.00 | 1072 | 0.282 [0.238, 0.327] | 0.173 | 2.00 | 1158 |
| k=3 | 0.623 [0.573, 0.672] | 0.487 | 3.00 | 2125 | 0.328 [0.282, 0.374] | 0.207 | 3.00 | 2270 |
| k=4 | 0.629 [0.581, 0.677] | 0.487 | 4.00 | 3735 | 0.349 [0.301, 0.398] | 0.227 | 4.00 | 3865 |
| k=5 | 0.635 [0.587, 0.683] | 0.490 | 5.00 | 5883 | 0.354 [0.305, 0.403] | 0.250 | 5.00 | 6008 |
| P1 self-decided | 0.621 [0.574, 0.669] | 0.467 | 2.46 | 3538 | 0.340 [0.292, 0.388] | 0.230 | 3.86 | 7272 |
| P2 stability | 0.622 [0.573, 0.670] | 0.477 | 2.76 | 2888 | 0.341 [0.294, 0.388] | 0.230 | 3.37 | 4552 |
| P3 coverage | 0.631 [0.583, 0.679] | 0.483 | 2.49 | 3800 | 0.335 [0.288, 0.382] | 0.217 | 3.67 | 6970 |
| Oracle | 0.704 [0.659, 0.749] | 0.557 | 1.58 | 944 | 0.430 [0.380, 0.480] | 0.300 | 1.78 | 1238 |

Points to make:
- Quality saturates early: on HotpotQA k=2 to k=5 differ by 0.03 F1, on MuSiQue k=3 to k=5 by 0.03.
- All three adaptive policies lie within about 0.015 F1 of k=5 on both datasets (CIs in `*_vs_fixed.csv` include zero for F1). They save searches (HotpotQA 2.5 to 2.8 vs 5; MuSiQue 3.4 to 3.9 vs 5).
- Exception on EM: on MuSiQue, P3 is 0.033 EM below k=5 (CI [-0.060, -0.010]).
- Adaptive policies beat k=1 and k=2 clearly on MuSiQue (P1 vs k=2: +0.058 F1, CI [0.024, 0.093]); on HotpotQA they do not beat k=2.
- Oracle headroom is 0.07 (HotpotQA) and 0.08 (MuSiQue) F1 over k=5, at about 1.6 to 1.8 searches.

## 2. Quality versus cost

Figures: `{dataset}_main_f1_vs_searches.png`, `{dataset}_main_f1_vs_tokens.png` (EM versions as secondary).

- **Searches as cost:** adaptive policies sit near the fixed-k curve at 2.5 to 3.9 searches; they do not dominate it.
- **Tokens as cost:** the checks add overhead.
  - HotpotQA: k=3 has F1 0.623 at 2125 tokens; P1 0.621 at 3538, P2 0.622 at 2888, P3 0.631 at 3800. Fixed k=3 is the cheapest in tokens at equal F1.
  - MuSiQue: P1 (7272) and P3 (6970) cost more tokens than k=5 (6008) for lower F1. P2 (4552) is the cheapest adaptive policy but does not beat k=3 or k=4 in F1.
- Search count therefore flatters P1 and P3; token cost is the fairer comparison.

## 3. Stop behaviour

Figure: `{dataset}_main_stop_steps.png`; table: `*_stop_steps.csv`.

- HotpotQA: P1 and P3 stop at search 1 for 36% and 35% of questions, at search 5 for 23% and 22%. P2 cannot stop at search 1 (needs two drafts); 53% of its stops are at search 2.
- MuSiQue: P1 reaches search 5 in 61% of questions, P3 in 50%, P2 in 27%. Self-decided policies rarely stop early on the harder dataset.
- Oracle stops at search 1 for 63% (HotpotQA) and 60% (MuSiQue): a correct draft often exists after one search, so the headroom is large, but the policies do not find it.

## 4. Difficulty (MuSiQue by hops, n = 169 / 80 / 51)

**Table 2.** Average searches and F1 for 2, 3, 4 hops.

| Policy | 2-hop searches / F1 | 3-hop | 4-hop |
|---|---|---|---|
| k=3 | 3.0 / 0.425 | 3.0 / 0.209 | 3.0 / 0.189 |
| k=5 | 5.0 / 0.449 | 5.0 / 0.259 | 5.0 / 0.190 |
| P1 | 3.48 / 0.425 | 4.30 / 0.256 | 4.43 / 0.187 |
| P2 | 3.33 / 0.443 | 3.45 / 0.235 | 3.39 / 0.167 |
| P3 | 3.19 / 0.414 | 4.19 / 0.262 | 4.43 / 0.187 |

- P1 and P3 search more as hops increase, so they detect difficulty; P2 does not (about 3.4 at every level).
- On 4-hop questions extra searching does not help (k=3 and k=5 both 0.19). Small n; treat as descriptive.

## 5. Error analysis

**Table 3.** Outcome at the policy's stop (counts out of 300). Near miss: EM 0 and F1 ≥ 0.5. Reading failure: all gold passages retrieved, answer wrong. Retrieval failure: gold passage missing.

| Policy | HotpotQA correct / near miss / reading / retrieval | MuSiQue correct / near miss / reading / retrieval |
|---|---|---|
| k=3 | 146 / 53 / 52 / 49 | 62 / 42 / 39 / 157 |
| k=5 | 147 / 57 / 65 / 31 | 75 / 35 / 57 / 133 |
| P1 | 140 / 62 / 45 / 53 | 69 / 39 / 39 / 153 |
| P2 | 143 / 57 / 55 / 45 | 69 / 38 / 41 / 152 |
| P3 | 145 / 58 / 45 / 52 | 65 / 38 / 40 / 157 |
| Oracle | 167 / 55 / 17 / 61 | 90 / 43 / 2 / 165 |

- MuSiQue is retrieval-limited: 133 of 225 wrong answers at k=5 have a gold passage missing.
- HotpotQA failures are mostly reading and format (near misses, e.g. "Quanzhou City" vs "Quanzhou").
- Early stopping trades reading failures for retrieval failures: P1 has fewer reading failures than k=5 on HotpotQA (45 vs 65) but more retrieval failures (53 vs 31).

**Stopping error categories** (`*_errors.csv`; definitions in `src/evaluate.py`):
- over-searching: a correct draft existed at an earlier step than the stop.
- premature stop: the stop draft is wrong and a correct draft appears at a later step (not necessarily step 5).
- not caused by stopping: no step has a correct draft.

Qualitative examples (`*_examples.csv`):
- Over-searching: a correct early answer is lost when later passages arrive ("London" becomes "London Heathrow Airport"; "Texas Highways" becomes "Human Events").
- Premature stop: the policy accepts a plausible but wrong draft ("Tolyatti" instead of "Palmiro Togliatti").
- EM rejects semantically correct answers ("Cornell University School of Industrial and Labor Relations" vs "Cornell University"), which is why F1 is the main metric.

## 6. Limitations to state

- One model (Qwen3.5-4B), one retriever (BM25), 300 questions per dataset; CIs are wide (about ±0.05 F1).
- Policies are applied offline to saved 5-step trajectories; check calls never enter the agent prompts (no-feedback rule), so cost is accounted per policy.
- Near-duplicate queries remain after the retry guard (about 8 to 15% of steps).
- The oracle uses gold answers and is a reference, not a policy.
- Main trajectories were generated on one machine (Colab T4); greedy outputs can differ slightly on other hardware.
- AI-assisted code and drafting should be disclosed in the AI-use statement.

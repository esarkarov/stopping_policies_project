# Results (draft)

**Data.** Two question sets per dataset, both sampled with seed 42 and run with the same agent code (prompts and retriever frozen at tag `freeze-v1`):

- **Main**: 300 questions. P0 to P3 and the oracle were fixed before these runs. The variants in section 7 were defined after looking at this set.
- **Confirm**: 400 further questions, disjoint from main and pilot. The variants in section 7 are judged only on this set.
- **Pooled**: main + confirm, 700 questions per dataset. Used for the frozen policies only (P0, P1, P2, P3, oracle); variants and thresholds are never pooled.

**Statistics.** Intervals are paired bootstrap 95% CIs (10,000 resamples, seed 42) unless stated. F1 is the main metric, EM secondary. Cost is average searches and average tokens (prompt + completion) per question, counting only the calls a policy needs: queries and retries, the final answer, plus the checks the policy uses (P1: the "enough?" check after every search; P2: an answer after every search; P3: the decomposition and a coverage check after every search). The oracle uses gold answers and is a reference only.

Numbers come from `results/tables/*_pooled_*.csv`, `*_confirm_*.csv` and `*_main_*.csv`.

## 1. Overall quality and cost

**Table 1.** Pooled, n = 700 per dataset. F1 [95% CI], EM, average searches and tokens per question.

| Policy          | HotpotQA F1          | EM    | Searches | Tokens | MuSiQue F1           | EM    | Searches | Tokens |
| --------------- | -------------------- | ----- | -------- | ------ | -------------------- | ----- | -------- | ------ |
| k=1             | 0.427 [0.393, 0.460] | 0.317 | 1.00     | 432    | 0.138 [0.118, 0.159] | 0.056 | 1.00     | 468    |
| k=2             | 0.592 [0.558, 0.624] | 0.460 | 2.00     | 1066   | 0.260 [0.232, 0.288] | 0.151 | 2.00     | 1173   |
| k=3             | 0.628 [0.596, 0.660] | 0.499 | 3.00     | 2130   | 0.301 [0.271, 0.331] | 0.181 | 3.00     | 2299   |
| k=4             | 0.651 [0.619, 0.682] | 0.519 | 4.00     | 3740   | 0.328 [0.298, 0.358] | 0.214 | 4.00     | 3948   |
| k=5             | 0.649 [0.617, 0.679] | 0.509 | 5.00     | 5865   | 0.348 [0.317, 0.380] | 0.240 | 5.00     | 6111   |
| P1 self-decided | 0.635 [0.603, 0.665] | 0.487 | 2.42     | 3433   | 0.331 [0.300, 0.362] | 0.221 | 3.94     | 7578   |
| P2 stability    | 0.624 [0.591, 0.655] | 0.490 | 2.73     | 2795   | 0.319 [0.288, 0.348] | 0.206 | 3.37     | 4631   |
| P3 coverage     | 0.624 [0.591, 0.655] | 0.481 | 2.48     | 3757   | 0.322 [0.292, 0.353] | 0.207 | 3.74     | 7321   |
| Oracle          | 0.722 [0.691, 0.750] | 0.581 | 1.60     | 962    | 0.424 [0.391, 0.456] | 0.289 | 1.78     | 1272   |

- Quality saturates early on HotpotQA (k=4 and k=5 are equal, k=3 is 0.02 below) and keeps rising slowly on MuSiQue (k=3 to k=5: +0.047).
- The adaptive policies use about half the searches of k=5 on HotpotQA (2.4 to 2.7 against 5) and 3.4 to 3.9 on MuSiQue, at a lower F1 than k=5.
- Oracle headroom over k=5 is 0.073 F1 (HotpotQA) and 0.076 (MuSiQue), at 1.6 to 1.8 searches.

**Table 2.** F1 difference to k=5 [95% CI].

| Policy | HotpotQA confirm (n=400) | HotpotQA pooled         | MuSiQue confirm (n=400) | MuSiQue pooled          |
| ------ | ------------------------ | ----------------------- | ----------------------- | ----------------------- |
| P1     | -0.015 [-0.040, 0.011]   | -0.014 [-0.031, 0.004]  | -0.020 [-0.039, -0.002] | -0.017 [-0.032, -0.003] |
| P2     | -0.033 [-0.059, -0.008]  | -0.025 [-0.042, -0.007] | -0.042 [-0.068, -0.016] | -0.030 [-0.048, -0.012] |
| P3     | -0.041 [-0.068, -0.014]  | -0.025 [-0.043, -0.006] | -0.031 [-0.052, -0.010] | -0.026 [-0.042, -0.010] |

- On the main set alone all differences to k=5 were between -0.020 and -0.004 and every interval included zero. With the larger sample, five of six pooled differences are significantly below zero. The earlier "within noise of k=5" reading was a power problem, not equivalence.
- Against smaller budgets: pooled, all three adaptive policies beat k=2 on HotpotQA (P1 +0.042 [0.018, 0.068]) and k=3 on MuSiQue (P1 +0.030 [0.008, 0.052]). On HotpotQA they equal k=3 (P1 vs k=3: +0.007 [-0.016, 0.029]); on MuSiQue they equal k=4 (P1 vs k=4: +0.003 [-0.017, 0.023]).

## 2. Quality versus matched cost

Each adaptive policy is compared with a mixture of the two neighbouring fixed-k policies that has the same average cost (paired bootstrap). Figures: `{dataset}_main_f1_vs_searches.png`, `{dataset}_main_f1_vs_tokens.png` (EM versions secondary).

**Table 3.** Pooled F1 difference to the equal-cost fixed policy [95% CI]. Positive = the adaptive policy is better.

| Cost                        | Policy | HotpotQA                | MuSiQue                            |
| --------------------------- | ------ | ----------------------- | ---------------------------------- |
| Searches                    | P1     | +0.027 [0.006, 0.049]   | +0.005 [-0.015, 0.024]             |
|                             | P2     | +0.006 [-0.007, 0.019]  | +0.008 [-0.007, 0.023]             |
|                             | P3     | +0.014 [-0.005, 0.034]  | +0.002 [-0.017, 0.021]             |
| Tokens                      | P1     | -0.012 [-0.032, 0.008]  | not comparable (7578 tokens > k=5) |
|                             | P2     | -0.014 [-0.028, 0.000]  | -0.016 [-0.033, 0.001]             |
|                             | P3     | -0.027 [-0.048, -0.008] | not comparable (7321 tokens > k=5) |
| Tokens, shared check call\* | P1     | +0.009 [-0.012, 0.031]  | -0.002 [-0.019, 0.014]             |
|                             | P3     | -0.005 [-0.026, 0.015]  | -0.009 [-0.028, 0.011]             |

\*Lower bound on cost if the check shared the answer call (answer tokens at the stop plus the check's completion tokens, plus the decomposition for P3). P2 is unchanged.

- **Searches as cost:** only P1 on HotpotQA beats the fixed policy (+0.027). Everything else includes zero. On the main set alone, P1 was +0.008 [-0.021, 0.038]; on confirm it was +0.042 [0.012, 0.073].
- **Tokens as cost:** no adaptive policy beats the fixed curve; P3 on HotpotQA is worse (-0.027). The checks cost about as many tokens as the searches they save, because prompts dominate token counts (about 99%) and a check prompt is almost as long as an answer prompt.
- On MuSiQue, P1 and P3 cost more tokens than k=5 for a lower F1, so they are dominated in tokens.
- Only under the hypothetical shared-call accounting do P1 and P3 reach parity with the fixed policy. That accounting is a lower bound and is not what the agents ran.
- Search count therefore flatters P1 and P3; token cost is the fairer comparison.

## 3. Do the stopping signals carry information?

Test: keep each policy's set of stop steps but shuffle them across questions (2000 permutations). If the policy's F1 is above the shuffled F1, the signal is informative about which questions can stop early.

**Table 4.** Pooled F1: real versus shuffled stops (same average searches).

| Policy | HotpotQA real / shuffled (p) | MuSiQue real / shuffled (p) |
| ------ | ---------------------------- | --------------------------- |
| P1     | 0.635 / 0.546 (p<0.001)      | 0.331 / 0.308 (p=0.0015)    |
| P2     | 0.624 / 0.612 (p=0.019)      | 0.319 / 0.305 (p=0.028)     |
| P3     | 0.624 / 0.553 (p<0.001)      | 0.322 / 0.302 (p=0.005)     |

- P1 and P3 are clearly informative on HotpotQA (+0.089, +0.071). On MuSiQue the margins for P1 and P3 are small (+0.023, +0.020).
- P2 adds little on HotpotQA (+0.012): answer stability mostly reflects when the model stops changing its answer, not which questions are easy.
- Replication: the same pattern holds on main and on confirm separately (HotpotQA P1 +0.081 and +0.094; MuSiQue P1 +0.023 and +0.022).
- Descriptively, the F1 at steps where the signal fires is higher than where it does not (P1 over all steps: HotpotQA 0.74 against 0.34 on confirm; MuSiQue 0.49 against 0.20). This mixes signal quality with question difficulty; the shuffled baseline is the cleaner test.
- Even informative signals do not turn into a cost advantage (section 2): they are better than random stopping, but not better than a fixed budget of equal cost.

## 4. Stop behaviour

Figure: `{dataset}_{main,confirm}_stop_steps.png`; table: `*_stop_steps.csv`. Percentages are main / confirm.

- HotpotQA: P1 stops at search 1 for 36% / 40% of questions and reaches search 5 for 23% / 21%; P3 stops at search 1 for 35% / 35%. P2 cannot stop at search 1 (it needs two drafts); 53% / 55% of its stops are at search 2.
- MuSiQue: P1 reaches search 5 in 61% / 67% of questions, P3 in 50% / 56%, P2 in 27% / 27%. The self-decided policies rarely stop early on the harder dataset.
- The oracle stops at search 1 for 63% / 63% (HotpotQA) and 60% / 62% (MuSiQue): a correct draft often exists after one search, so the headroom is large, but the policies do not find it.

## 5. Difficulty (MuSiQue by number of hops)

**Table 5.** Average searches by number of hops, main (n = 169 / 80 / 51) and confirm (n = 212 / 129 / 59).

| Policy            | 2-hop       | 3-hop       | 4-hop       |
| ----------------- | ----------- | ----------- | ----------- |
| P1 main / confirm | 3.48 / 3.58 | 4.30 / 4.37 | 4.43 / 4.66 |
| P2 main / confirm | 3.33 / 3.27 | 3.45 / 3.37 | 3.39 / 3.76 |
| P3 main / confirm | 3.19 / 3.47 | 4.19 / 4.02 | 4.43 / 4.47 |

- P1 and P3 search more as the number of hops grows, in both sets. P2 is flat on 2 and 3 hops and rises only slightly at 4 hops.
- An earlier observation that extra searching does not help on 4-hop questions (k=3 and k=5 both 0.19 F1 on main) did not replicate on confirm (k=3 0.137, k=5 0.238). Per-hop F1 is noisy at these sample sizes (n = 51 and 59) and is not interpreted.

## 6. Error analysis

**Table 6.** Outcome at the policy's stop, pooled counts (main + confirm, out of 700 per dataset). Near miss: EM 0 and F1 >= 0.5. Reading failure: all gold passages retrieved, answer wrong. Retrieval failure: a gold passage is missing.

| Policy | HotpotQA correct / near miss / reading / retrieval | MuSiQue correct / near miss / reading / retrieval |
| ------ | -------------------------------------------------- | ------------------------------------------------- |
| k=3    | 349 / 118 / 108 / 125                              | 127 / 98 / 88 / 387                               |
| k=5    | 356 / 129 / 136 / 79                               | 168 / 82 / 122 / 328                              |
| P1     | 341 / 138 / 92 / 129                               | 155 / 86 / 86 / 373                               |
| P2     | 343 / 122 / 110 / 125                              | 144 / 90 / 87 / 379                               |
| P3     | 337 / 131 / 96 / 136                               | 145 / 89 / 89 / 377                               |
| Oracle | 407 / 128 / 33 / 132                               | 202 / 110 / 9 / 379                               |

- MuSiQue is retrieval-limited: 328 of the 532 wrong answers at k=5 (62%) have a gold passage missing.
- HotpotQA failures are mostly reading and format: near misses and reading failures are 265 of the 344 wrong answers at k=5 (77%). Example: "Quanzhou City" for "Quanzhou".
- Early stopping trades reading failures for retrieval failures. On HotpotQA, P1 has fewer reading failures than k=5 (92 against 136) but more retrieval failures (129 against 79).

**Stopping error categories** (`*_errors.csv`; definitions in `src/evaluate.py`), pooled:

| Policy | HotpotQA over-searching / premature stop | MuSiQue over-searching / premature stop |
| ------ | ---------------------------------------- | --------------------------------------- |
| k=5    | 51 / 0                                   | 34 / 0                                  |
| P1     | 20 / 46                                  | 22 / 25                                 |
| P2     | 34 / 30                                  | 21 / 37                                 |
| P3     | 28 / 42                                  | 23 / 34                                 |

- over-searching: a correct draft existed at a step earlier than the stop, and the stop draft is wrong.
- premature stop: the stop draft is wrong and a correct draft appears at a later step (not necessarily step 5).
- not caused by stopping: no step has a correct draft (293 HotpotQA and 498 MuSiQue questions pooled).
- k=5 loses 51 HotpotQA questions to over-searching; P1 trades that for 46 premature stops and 20 over-searching cases. Net, P1 ends with 15 fewer correct answers than k=5 (341 against 356).

Qualitative examples (`*_examples.csv`):

- Over-searching: a correct early answer is lost when later passages arrive ("London" becomes "London Heathrow Airport"; "Texas Highways" becomes "Human Events").
- Premature stop: the policy accepts a plausible but wrong draft ("Tolyatti" instead of "Palmiro Togliatti"; "Sigmund Jähn" for a question that asks which person flew later).
- EM rejects semantically correct answers ("Cornell University School of Industrial and Labor Relations" for "Cornell University"), which is why F1 is the main metric.

## 7. Post-hoc variants and P(yes) thresholds (confirm set only)

Five variants and a threshold family were defined after looking at the main set (`docs/variants_preregistration.md`). The rules and the decision criterion are written down in that document, which says it was made before the confirm set was evaluated. [Before submission, check with `git log` that the declaration commit precedes the commit that added `*_confirm_*` tables, and cite the hash here.] Criterion: a variant counts as an improvement only if the 99% interval of the F1 gap to the equal-cost fixed policy is above zero on confirm; a result in one dataset only is dataset-specific. All variants are reported.

| Variant   | Stops at the first search s where | HotpotQA F1 / searches / tokens | MuSiQue F1 / searches / tokens |
| --------- | --------------------------------- | ------------------------------- | ------------------------------ |
| `P1_min2` | s >= 2 and P1 says yes            | 0.663 / 2.85 / 3653             | 0.334 / 4.15 / 7726            |
| `P3_min2` | s >= 2 and P3 says covered        | 0.640 / 2.90 / 4012             | 0.326 / 3.95 / 7448            |
| `P1_c2`   | P1 says yes at s and s-1          | 0.659 / 3.31 / 4943             | 0.337 / 4.49 / 8907            |
| `P1&P3`   | both say yes                      | 0.650 / 2.84 / 6274             | 0.337 / 4.29 / 12428           |
| `P3&P2`   | P3 covered and draft unchanged    | 0.645 / 3.33 / 6570             | 0.335 / 4.34 / 11145           |

**Table 7.** F1 gap to the equal-searches fixed policy [99% interval], confirm.

| Variant   | HotpotQA               | MuSiQue                |
| --------- | ---------------------- | ---------------------- |
| `P1_min2` | +0.038 [0.006, 0.071]  | +0.017 [-0.013, 0.048] |
| `P3_min2` | +0.013 [-0.019, 0.045] | +0.015 [-0.019, 0.050] |
| `P1_c2`   | +0.016 [-0.011, 0.044] | +0.010 [-0.013, 0.034] |
| `P1&P3`   | +0.026 [-0.010, 0.062] | +0.016 [-0.011, 0.044] |
| `P3&P2`   | +0.001 [-0.023, 0.024] | +0.012 [-0.012, 0.038] |

- **No variant is an improvement under the criterion.** The one interval above zero is `P1_min2` on HotpotQA in searches. It is not above zero in tokens (-0.003 [-0.036, 0.030] at 99%), and the unmodified P1 reaches the same level in searches on that set (+0.042, 99% [0.004, 0.082]). `P1_min2` against P1 is +0.018 [-0.009, 0.045] at 99%.
- `P1_min2` and `P3_min2` against the policy they modify: `P3_min2` over P3 is +0.021 [0.004, 0.040] on HotpotQA and +0.012 [0.0005, 0.025] on MuSiQue at 95%, but both include zero at 99%.
- `P1&P3` and `P3&P2` pay for two signals and cost more tokens than k=5 on both datasets (6274 and 6570 against 5865 on HotpotQA; 12428 and 11145 against 6111 on MuSiQue). We do not propose them.
- On the main set the variants looked better on HotpotQA (for example `P1_min2` +0.027 [0.001, 0.053] at 95%). That did not hold on confirm, which is what the held-out set is for. The main-set numbers are probably optimistic because the variants were chosen after seeing that set.

**P(yes) thresholds.** Replacing the greedy "Yes" of P1 by P(yes) >= tau (probabilities from replaying the check on the saved passages) changes where P1 sits on the cost axis but not whether it beats the fixed curve:

| tau | HotpotQA confirm: searches, F1 gap [95%] | MuSiQue confirm: searches, F1 gap [95%] |
| --- | ---------------------------------------- | --------------------------------------- |
| 0.5 | 2.34, +0.047 [0.017, 0.078]              | 3.98, +0.009 [-0.016, 0.036]            |
| 0.8 | 3.25, +0.017 [-0.007, 0.041]             | 4.67, +0.007 [-0.004, 0.017]            |
| 0.9 | 3.80, +0.006 [-0.014, 0.026]             | 4.86, +0.004 [0.0004, 0.008]            |

- At tau >= 0.99 the check almost always fires too late and the policy converges to k=5.
- The HotpotQA gap at tau = 0.5 was +0.007 [-0.021, 0.036] on main, and MuSiQue shows no gain. We do not claim an improvement (the criterion needs 99% intervals).
- The replay disagrees with the stored greedy answers for 2.7% to 3.3% of checks, so the thresholded policy is close to, not identical to, P1 as run.

## 8. Limitations to state

- One model (Qwen3.5-4B), one retriever (BM25), two datasets. With 700 questions per dataset the 95% interval half-width of F1 is about 0.03.
- Policies are applied offline to saved 5-step trajectories; check calls never enter the agent prompts (no-feedback rule), so each policy's cost is accounted separately. Search-count savings are real; token savings depend on how checks are implemented (section 2).
- The retry guard for repeated queries fired on about 30% of steps on the main set (500/1500 on HotpotQA, 442/1500 on MuSiQue). Near-duplicate queries remain in about 9 to 10% of main-set steps (154 and 132).
- The oracle uses gold answers and is a reference, not a policy.
- Main and confirm trajectories were generated in separate Colab T4 sessions. Greedy outputs can differ slightly across hardware and sessions; the confirm runs carry a run manifest (git commit, Ollama version, model digest, file hashes) in `results/trajectories/*.manifest.jsonl`, the main runs predate it. Pooled intervals treat the two sets as one sample.
- Variants and thresholds are post-hoc and were judged on one held-out set; with five variants, claims need 99% intervals.
- AI-assisted code and drafting are disclosed in the README and the paper's AI-use statement.

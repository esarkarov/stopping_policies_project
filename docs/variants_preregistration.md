# Post-hoc variants: declaration made before the confirmation set was evaluated

The policies P0 (fixed budget k = 1 to 5), P1, P2, P3 and the oracle were fixed at tag `freeze-v1` and are the pre-planned comparison. The rules below were defined after looking at the 300 main questions per dataset. They are exploratory: the main set is their development set, and the 400 confirmation questions per dataset (`--split confirm`) are the only data on which they are judged.

## Variants

All variants use the saved signals of the agent trajectories; no new model calls are needed.

| Name | Stops at the first search s where |
|---|---|
| `P1_min2` | s >= 2 and P1 says "Yes" |
| `P3_min2` | s >= 2 and P3 says all sub-questions are covered |
| `P1_c2` | s >= 2 and P1 says "Yes" at searches s and s-1 |
| `P1&P3` | P1 says "Yes" and P3 says all sub-questions are covered |
| `P3&P2` | P3 says all covered and the draft answer equals the previous draft (P2 rule) |

If no search satisfies the rule, the draft after search 5 is returned.

A second family, `P1_t<tau>`, replaces the greedy "Yes" of P1 by P(yes) >= tau, with tau in {0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999}. P(yes) comes from replaying the P1 check on the saved passages (`scripts/run_p1_logprobs.py`).

## Cost

Only the calls a rule needs are counted, as for P1 to P3: queries and retries, the final answer, plus the checks the rule uses. `P1_min2` and `P3_min2` do not run the check after search 1. `P1&P3` and `P3&P2` pay for both of their signals. `P3_*` variants include the decomposition call. Tokens are Ollama prompt plus completion tokens.

## Evaluation

- Comparison: each variant against a fixed-budget policy of equal average cost, a mixture of the two neighbouring P0_k policies (`compare_to_curve`), for cost in searches and in tokens. A variant whose token cost lies outside the P0 range is judged on searches only and reported as more expensive in tokens.
- Secondary comparison: each variant against the policy it modifies (`variants_vs_base`).
- Intervals: paired bootstrap, 10,000 resamples, seed 42. With five variants and the threshold family tested, 99% intervals are used for any claim of improvement; 95% intervals are also reported.
- A variant is called an improvement only if the 99% interval of the F1 gap to the equal-cost fixed policy is above zero on the confirmation set. A result in one dataset only is described as dataset-specific.
- All variants are reported, whether or not they help. The main-set numbers are reported as exploratory and are not combined with the confirmation results.

## Exploratory result on the main set (for transparency)

F1 gap to the equal-searches fixed policy, 95% interval (the 99% interval includes zero in every case):

| Variant | HotpotQA | MuSiQue |
|---|---|---|
| `P1_min2` | +0.027 [0.001, 0.053] | +0.003 [-0.025, 0.030] |
| `P3_min2` | +0.029 [0.003, 0.055] | +0.018 [-0.009, 0.046] |
| `P1_c2` | +0.027 [0.006, 0.050] | +0.005 [-0.016, 0.026] |
| `P1&P3` | +0.023 [-0.002, 0.049] | +0.008 [-0.017, 0.033] |
| `P3&P2` | +0.010 [-0.010, 0.030] | +0.018 [-0.006, 0.042] |

These intervals do not account for the selection of variants after seeing the data and are probably optimistic.

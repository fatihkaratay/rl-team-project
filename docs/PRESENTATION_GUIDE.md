# Presentation Slide Guide

Final presentation content for EN.705.741.8VL. Each slide below gives the
title, bullet text, and notes on which figure/table to embed. Everything
aligns with PAPER_V2.

---

## Slide 1 — Title

**Title:** Adaptive Multi-Site Stimulation Control Using RL in an MDP

**Subtitle:** Learning Sequential Stimulation Policies Under State-dependent EEG Response Dynamics

**Authors:** Fatih Karatay & Cody Moxam

**Course:** EN.705.741.8VL Reinforcement Learning

**Date:** April 2026

---

## Slide 2 — Problem Framing

**Title:** Problem Framing: Why RL, Not a Bandit?

**Bullets:**

- Brain stimulation site selection is often treated as a static target-selection problem (multi-armed bandit)
- Clinical neurostimulation is history-dependent: repeated stimulation alters tissue responsiveness (Pineau et al., 2009)
- Current actions change future state occupancy and future reward quality
- This makes it a sequential control problem, not one-step arm selection

**Key insight (callout box or bold text):**

> Optimal behavior is not "choose the best site" — it is "sequence sites to accumulate reward while managing state degradation"

**Visual:** Simple flow diagram: baseline → receptive → non-receptive (with arrows showing "same site" drives right, "switch site" promotes recovery left)

---

## Slide 3 — MDP Formulation

**Title:** MDP Formulation

**Bullets:**

- **State:** s_t = (current_site, patient_state, t)
  - current_site: {Start, S1, S2, S3, S4}
  - patient_state: {baseline, receptive, non-receptive}
  - t: timestep within episode
- **Actions:** {S1, S2, S3, S4} — choose a stimulation site
- **Observation:** EEG response ~ P(o | site, patient_state)
- **Reward:** favorable = +1, neutral = 0, unfavorable = -1, minus optional switch cost
- **State space:** For H=10: |S| = 165, |S×A| = 660 — fully tabular

**Table to include:** Table 1 — Patient-state transition probabilities

| Current State | Action Type | → baseline | → receptive | → non-receptive |
|---|---|---|---|---|
| baseline | any | 0.60 | 0.30 | 0.10 |
| receptive | same site | 0.20 | 0.30 | 0.50 |
| receptive | different site | 0.50 | 0.30 | 0.20 |
| non-receptive | same site | 0.10 | 0.10 | 0.80 |
| non-receptive | different site | 0.40 | 0.30 | 0.30 |

---

## Slide 4 — Observation Model

**Title:** EEG Observation Model (High-Separation Setting)

**Table to include:** Table 2 — Observation probabilities (high-separation)

| Site | Patient State | Favorable | Neutral | Unfavorable | E[reward] |
|---|---|---|---|---|---|
| S1 | baseline | 0.70 | 0.20 | 0.10 | +0.60 |
| S1 | receptive | 0.85 | 0.10 | 0.05 | +0.80 |
| S1 | non-receptive | 0.30 | 0.40 | 0.30 | 0.00 |
| S2 | baseline | 0.50 | 0.30 | 0.20 | +0.30 |
| S2 | receptive | 0.65 | 0.25 | 0.10 | +0.55 |
| S2 | non-receptive | 0.20 | 0.40 | 0.40 | -0.20 |
| S3 | baseline | 0.30 | 0.40 | 0.30 | 0.00 |
| S3 | receptive | 0.45 | 0.35 | 0.20 | +0.25 |
| S3 | non-receptive | 0.15 | 0.35 | 0.50 | -0.35 |
| S4 | baseline | 0.45 | 0.35 | 0.20 | +0.25 |
| S4 | receptive | 0.60 | 0.30 | 0.10 | +0.50 |
| S4 | non-receptive | 0.25 | 0.40 | 0.35 | -0.10 |

**Key points below table:**

- S1 is the best site in baseline/receptive, but drops to E[r]=0 when non-receptive
- Moderate and low separation settings compress these differences
- Reward is state-dependent — the agent must manage patient state, not just pick the best arm

---

## Slide 5 — Algorithms

**Title:** Algorithms

**Bullets:**

- **Monte Carlo Control** — on-policy, first-visit; updates from complete episode returns (Sutton & Barto, 2018, Ch. 5)
- **Q-Learning** — off-policy TD(0); greedy bootstrap target (Watkins & Dayan, 1992)
- **Expected SARSA** — on-policy TD(0); expected value under epsilon-greedy policy as bootstrap target (van Seijen et al., 2009)
- **Double Q-Learning** — two Q-tables to reduce maximization bias; action selection on (Q1+Q2)/2 (Hasselt, 2010)
- **Value Iteration** — model-based upper-bound benchmark; backward induction on known MDP (Sutton & Barto, 2018, Ch. 4)

**Shared hyperparameters:**

- alpha = 0.1, gamma = 1.0, epsilon: 1.0 → 0.05 (decay = 0.995)
- 5,000 training episodes per seed, 10 independent seeds
- Value Iteration: 200 evaluation rollouts per seed

---

## Slide 6 — Experimental Design & Statistical Methods

**Title:** Experimental Design and Statistical Methods

**Experiment grid:**

- 3 response-separation settings (high, moderate, low)
- 2 episode horizons (H = 5, H = 10)
- 3 switching costs (c_switch = 0, 0.1, 0.25)
- 5 algorithms
- **= 90 total configurations, 10 seeds each**

**Primary metric:**

- Mean final return: average episodic return over last 10% of training (episodes 4,501–5,000), averaged across 10 seeds

**Statistical tests:**

- **Welch's t-test** — pairwise algorithm comparisons (H1, H4)
- **Levene's test** — variance comparison between Q-Learning and Double Q-Learning (H3)
- **Spearman rank correlation** — switching cost vs. behavioral metrics (H2)
- Significance level: alpha = 0.05

---

## Slide 7 — Hypotheses

**Title:** Hypotheses

**List (bold labels, one per line):**

- **H1.** Agents trained in higher-separation settings achieve higher mean final return than agents trained in lower-separation settings.
- **H2.** Higher switching cost is associated with lower switching frequency and higher non-receptive state occupancy in converged policies.
- **H3.** Double Q-Learning exhibits lower variance in per-seed mean final return compared to Q-Learning.
- **H4.** Model-free agents achieve lower mean final return than the Value Iteration benchmark.
- **H5.** Converged policies develop structured site preferences aligned with the reward model, rather than selecting sites uniformly.

---

## Slide 8 — Results: Learning Curves (H1)

**Title:** H1: Reward Separability Improves Learning Performance

**Figure:** `figures/fig1_learning_curves_h10_c0.0.png`

**Key points:**

- High separation: TD methods converge to 4.6–4.7; MC lags at ~4.1; VI upper bound at ~5.2
- Moderate: same ranking, compressed to 3.1–3.3 for TD methods
- Low: all model-free methods flatten near 1.8–2.0
- Welch's t-test: all algorithms significantly higher in high vs. low separation (all p < 0.001)
- **H1 is supported**

---

## Slide 9 — Results: Performance Summary & Gap to Optimal (H1/H4)

**Title:** H1/H4: Cross-Setting Performance and Gap to Optimal

**Left side — Table 3:**

| Algorithm | High | Moderate | Low |
|---|---|---|---|
| Value Iteration | 5.235 ± 0.436 | 3.880 ± 0.589 | 2.340 ± 0.530 |
| Expected SARSA | 4.674 ± 0.107 | 3.323 ± 0.090 | 1.965 ± 0.139 |
| Double Q-Learning | 4.593 ± 0.147 | 3.317 ± 0.141 | 1.942 ± 0.137 |
| Q-Learning | 4.589 ± 0.144 | 3.260 ± 0.144 | 1.951 ± 0.113 |
| Monte Carlo | 4.081 ± 0.232 | 3.059 ± 0.140 | 1.801 ± 0.128 |

**Right side — Figure:** `figures/fig7_algorithm_gap_h10_c0.0.png`

**Key points:**

- MC significantly below all TD methods in every setting (all p < 0.05)
- No significant pairwise differences among Q-Learning, Expected SARSA, Double Q-Learning (all p > 0.17)
- TD methods recover 84–89% of VI return
- **H4 is supported** in high and moderate settings (p < 0.05)

---

## Slide 10 — Results: Switching Cost Tradeoff (H2)

**Title:** H2: Switching Cost Shapes Policy Behavior

**Figure:** `figures/fig5_switching_vs_cost_high_h10.png`

**Table 4 (or key numbers):**

| Algorithm | c=0 switch rate | c=0.25 | c=0 non-rec | c=0.25 |
|---|---|---|---|---|
| Monte Carlo | 0.625 | 0.427 | 0.195 | 0.237 |
| Q-Learning | 0.511 | 0.300 | 0.205 | 0.273 |
| Expected SARSA | 0.464 | 0.275 | 0.214 | 0.275 |
| Double Q-Learning | 0.467 | 0.284 | 0.218 | 0.278 |

**Key points:**

- Switching frequency negatively correlated with c_switch (rho = -0.83 to -0.94, all p < 0.001)
- Non-receptive occupancy positively correlated with c_switch (rho = 0.76 to 0.85, all p < 0.001)
- Less switching → more time in non-receptive state → degraded reward
- This is the core control tradeoff — **proof this is not a bandit problem**
- **H2 is supported**

---

## Slide 11 — Results: Site Preference & Stability (H5/H3)

**Title:** H5: Structured Site Preferences / H3: Stability Comparison

**Figure:** `figures/fig4_site_freq_high_h10_c0.0.png`

**H5 key points:**

- All algorithms concentrate actions on S1 (highest expected reward in baseline/receptive)
- Value Iteration: ~77% on S1; TD methods: 67–71%; Monte Carlo: ~53%
- Baseline + receptive occupancy: 79–84% of episode time
- Policies are structured, not random — **H5 is supported**

**H3 key points:**

- Levene's test: no significant variance difference between Q-Learning and Double Q-Learning
  - p = 0.780 (high), p = 0.920 (moderate), p = 0.650 (low)
- Small state-action space (660) leaves limited scope for maximization bias
- **H3 is not supported**

---

## Slide 12 — Hypothesis Summary

**Title:** Summary of Hypothesis Outcomes

**Table:**

| Hypothesis | Outcome | Test | Key Result |
|---|---|---|---|
| H1: Higher separation → higher return | Supported | Welch's t-test | All algorithms p < 0.001 |
| H2: Higher c_switch → less switching, more non-receptive | Supported | Spearman | \|rho\| > 0.76, p < 0.001 |
| H3: Double Q has lower return variance | Not supported | Levene's test | p > 0.65 in all settings |
| H4: Model-free < Value Iteration | Supported | Welch's t-test | Significant in high/moderate (p < 0.05) |
| H5: Structured site preferences | Supported | Qualitative | S1 concentration 53–77% |

---

## Slide 13 — Discussion

**Title:** Discussion: Key Insights

**Bullets:**

- **Bootstrapping matters more than bootstrap type** in this small tabular MDP
  - Q-Learning, Expected SARSA, and Double Q-Learning are statistically indistinguishable
  - All three outperform Monte Carlo, which uses end-of-episode returns
  - Single-step TD propagates credit through state transitions faster than MC
- **The switching-cost dilemma proves this is sequential control**
  - In a bandit: higher switch cost → just stay on best arm
  - In this MDP: staying too long → non-receptive state → degraded future reward
  - Agents must balance immediate penalty vs. delayed state degradation
- **H3 null result is informative**
  - Maximization bias has limited scope in 660 state-action pairs with bounded rewards
  - Double Q-Learning's advantage may emerge in larger environments

---

## Slide 14 — Limitations and Conclusion

**Title:** Limitations and Conclusion

**Limitations:**

- Transition and observation models are hand-specified, not clinically calibrated
- Fully observed, discrete state space — not immediately applicable to clinical settings
- No explicit static-policy or bandit baseline for direct comparison
- Model-free agents evaluated at epsilon = 0.05 floor; VI uses deterministic optimal policy

**Conclusion:**

- State-dependent responsiveness makes stimulation-site selection a true RL problem
- TD methods consistently outperform MC; no significant differences among the three TD variants
- Switching cost reveals a genuine control tradeoff with no bandit counterpart
- Future work: partially observable formulations (POMDP), clinical calibration

---

## Slide 15 — References

**Title:** References

- Hasselt, H. van. (2010). Double Q-Learning. *NeurIPS*, 23, 2613–2621.
- Pineau, J., Guez, A., Vincent, R., Panuccio, G., & Bhomick, S. (2009). Treating epilepsy via adaptive neurostimulation: A reinforcement learning approach. *IJNS*, 19(4), 227–240.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- van Seijen, H., van Hasselt, H., Whiteson, S., & Wiering, M. (2009). A theoretical and empirical analysis of Expected Sarsa. *IEEE ADPRL*, 177–184.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3–4), 279–292.

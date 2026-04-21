# Presentation Script

## Slide Assignments

| Slide | Title                              | Presenter    |
| ----- | ---------------------------------- | ------------ |
| 1     | Title                              | Cody         |
| 2     | Problem Framing                    | Cody         |
| 3     | MDP Formulation                    | Cody         |
| 4     | EEG Observation Model              | Fatih        |
| 5     | Algorithms                         | Fatih        |
| 6     | Experimental Design & Stats        | Fatih        |
| 7     | Hypotheses                         | Cody         |
| 8     | H1: Learning Curves                | Cody         |
| 9     | H1/H4: Performance + Gap           | Cody         |
| 10    | H2: Switching Cost                 | Fatih        |
| 11    | H5/H3: Site Preference + Stability | Fatih        |
| 12    | Hypothesis Summary                 | Fatih        |
| 13    | Discussion: Key Insights           | Cody         |
| 14    | Limitations & Conclusion           | Cody         |
| 15    | References + Questions             | Cody & Fatih |

---

## Cody's Script

---

### Slide 1 — Title

Good afternoon RL Class. We are presenting our project on adaptive multi-site stimulation control using reinforcement learning in a Markov decision process. The central question is whether stimulation-site selection should be treated as a static optimization problem or as a sequential control problem in which current actions shape future response quality. Our results show that once state-dependent responsiveness is included, this becomes a genuine reinforcement learning setting rather than a simple bandit problem.

---

### Slide 2 — Problem Framing: Why RL, Not a Bandit?

This slide motivates the core framing. A bandit assumes each option has a fixed payoff distribution and that choosing one option does not change the future environment. But in clinical neurostimulation, that assumption is unrealistic. Repeated stimulation can alter tissue responsiveness over time, meaning the value of a given site depends on stimulation history. Because actions affect future state occupancy and future rewards, the problem is sequential. So the objective is not simply to identify the single best site, but to learn a sequence of site choices that accumulates reward while avoiding degradation into less responsive states.

---

### Slide 3 — MDP Formulation

Here we formalize the problem as a finite-horizon MDP. The state contains three components: the current stimulation site, the patient responsiveness state, and time within the episode. The action is simply the choice of the next site to stimulate. The EEG response is modeled probabilistically as an observation conditioned on both site and patient state. Rewards are assigned based on whether the observed outcome is favorable, neutral, or unfavorable, with an optional switching penalty. Because the state and action spaces are small, the entire problem is fully tabular, which makes it a clean testbed for comparing standard reinforcement learning algorithms.

---

## Fatih's Script

---

### Slide 4 — EEG Observation Model

This table shows the observation probabilities for the high-separation setting. The key thing to notice is S1: in the baseline state it has a 70% chance of favorable response with an expected reward of plus 0.60, and in the receptive state that goes up to 85% favorable with expected reward of plus 0.80. But when the patient becomes non-receptive, S1 drops to just 30% favorable and an expected reward of zero. So S1 is the best site, but only when the patient is in a good state.

S4 is an interesting secondary option — it's not as strong as S1 but follows the same pattern. S3 is the weakest site across the board.

The moderate and low separation settings compress these differences. In the low setting, all four sites have nearly identical distributions, so the only way for the agent to differentiate is through patient-state management. This is important because it tests whether the algorithms can learn the sequential structure even when the immediate reward signal is ambiguous.

---

### Slide 5 — Algorithms

We compared four model-free methods against a model-based benchmark.

Monte Carlo Control is on-policy and first-visit — it collects a full episode trajectory and then updates Q-values from the complete return. This means it only learns after an entire episode is finished.

Q-Learning is off-policy TD zero — it updates after every single step using a greedy bootstrap target. Expected SARSA is similar but on-policy: instead of taking the max, it uses the expected value under the current epsilon-greedy policy.

Double Q-Learning maintains two separate Q-tables to address maximization bias. At each step, one table is randomly chosen for the update, using the other table's values for the bootstrap target. Action selection uses the average of both tables.

Finally, Value Iteration serves as our upper-bound benchmark. It uses backward induction on the known transition and observation models to compute the exact optimal policy. It's not a competing method — it tells us the best possible performance.

All model-free agents share the same hyperparameters: learning rate of 0.1, discount factor of 1.0 since this is finite-horizon, and epsilon decaying from 1.0 down to a floor of 0.05 with a decay rate of 0.995. Each configuration runs for 5,000 training episodes across 10 independent seeds.

---

### Slide 6 — Experimental Design and Statistical Methods

Our experiments follow a full-factorial design. We cross three response-separation settings — high, moderate, and low — with two episode horizons, three switching cost levels, and five algorithms. That gives us 90 total configurations, each run across 10 random seeds.

Our primary performance metric is the mean final return: the average episodic return over the last 10% of training — that's episodes 4,501 through 5,000 — then averaged across all 10 seeds. By this point, epsilon has decayed to 0.05, so the agent is following a near-greedy policy.

For statistical testing, we use three methods. Welch's t-test for pairwise algorithm comparisons — this is how we test H1 and H4. Levene's test to compare the variance of returns between Q-Learning and Double Q-Learning for H3. And Spearman rank correlations to test the association between switching cost and behavioral metrics for H2. We use a significance level of 0.05 throughout.

---

## Cody's Script (continued)

---

### Slide 7 — Hypotheses

These were our five hypotheses going into the study. First, greater reward separability should improve learning. Second, higher switching cost should discourage switching and increase time spent in degraded, non-receptive states. Third, Double Q-learning should reduce performance variance relative to Q-learning if maximization bias is important. Fourth, model-free methods should remain below the value-iteration benchmark. And fifth, converged policies should show structured site preferences rather than random or uniform action selection. Together, these hypotheses test both the learning-performance story and the control-theoretic interpretation of the task.

---

### Slide 8 — H1: Reward Separability Improves Learning Performance

This result strongly supports the idea that signal quality matters for learning. In the high-separation setting, the TD methods converge to substantially higher returns than in the moderate or low-separation settings. Monte Carlo improves too, but lags behind. Once separability becomes low, all model-free methods flatten considerably, suggesting that compressed reward structure makes it difficult to distinguish beneficial from harmful action sequences. The statistical tests confirm that performance is significantly better in high versus low separation for every algorithm. So H1 is clearly supported.

---

### Slide 9 — H1/H4: Cross-Setting Performance and Gap to Optimal

This slide compares algorithms more directly and also asks how close they get to the model-based optimum. The main result is that Monte Carlo is consistently worse than the temporal-difference methods, which suggests that bootstrapping is especially valuable in this sequential setting. At the same time, Q-learning, Expected SARSA, and Double Q-learning are statistically indistinguishable from one another. Relative to value iteration, the TD methods recover most, but not all, of the achievable return—about 84 to 89 percent. So H4 is supported in the higher-information settings, where we can clearly detect a persistent gap between model-free learning and the optimal benchmark.

---

## Fatih's Script (continued)

---

### Slide 10 — H2: Switching Cost Shapes Policy Behavior

This is one of the most interesting findings. These two panels show what happens in the high-separation setting as we increase switching cost from zero to 0.25.

On the left, you can see that switch rate decreases monotonically for every algorithm. For example, Q-Learning drops from about 0.51 switches per step down to 0.30. On the right, non-receptive state occupancy increases in parallel — Q-Learning goes from about 0.205 to 0.273.

The Spearman correlations are strong: switching frequency has a negative correlation with cost of rho equals negative 0.83 to negative 0.94, all with p less than 0.001. Non-receptive occupancy has a positive correlation of rho equals 0.76 to 0.85, again all highly significant.

This is the core control tradeoff in our environment. When agents switch less to avoid the penalty, they end up spending more time in the non-receptive state, which degrades future reward. In a bandit problem, higher switching cost would just mean "stay on the best arm." Here, staying too long is actively harmful. This is what makes this a true sequential control problem, not a bandit. H2 is supported.

---

### Slide 11 — H5: Structured Site Preferences / H3: Stability Comparison

This figure shows the converged site-visit frequencies in the high-separation setting. The pattern is clear: all algorithms concentrate heavily on S1, the site with the highest expected reward. Value Iteration is the most aggressive, placing about 77% of its actions on S1. The TD methods allocate 67 to 71%. Monte Carlo is more diffuse at about 53%.

Importantly, no algorithm selects uniformly — a uniform policy would put 25% on each site. The degree of concentration actually correlates with performance: better algorithms are more focused on the best site while still retaining some switching to manage patient state. All algorithms maintain baseline-plus-receptive occupancy of 79 to 84% of episode time. H5 is supported.

Now for H3. We tested whether Double Q-Learning produces lower return variance than Q-Learning using Levene's test. The answer is no. The p-values are 0.780 in high separation, 0.920 in moderate, and 0.650 in low. The observed variances are nearly identical — for example, 0.0206 versus 0.0216 in high separation.

This is a null result, but it's an informative one. With only 660 state-action pairs and rewards bounded between minus one and plus one, there's limited scope for maximization bias to cause problems. H3 is not supported in this environment.

---

### Slide 12 — Hypothesis Summary

So to summarize all five hypotheses: H1 is supported — higher separation leads to higher return, confirmed by Welch's t-tests with all p-values less than 0.001. H2 is supported — higher switching cost reduces switching and increases non-receptive occupancy, with strong Spearman correlations. H3 is not supported — no significant variance difference between Q-Learning and Double Q-Learning. H4 is supported — model-free methods are significantly below Value Iteration in the high and moderate settings. And H5 is supported — all algorithms develop structured site preferences far from uniform.

Four out of five hypotheses are supported, and the one null result is consistent with the small scale of our environment.

---

## Cody's Script (continued)

---

### Slide 13 — Discussion: Key Insights

There are three main insights here. First, in this environment, bootstrapping matters more than the specific choice of bootstrap method. That is why the three TD algorithms cluster together and all outperform Monte Carlo. Second, the switching-cost dilemma provides the clearest evidence that this is a control problem, not a static target-selection problem. And third, the null result for Double Q-learning is itself informative: it suggests that algorithmic sophistication beyond standard TD learning may only become important in larger or noisier environments. So the discussion is not just which method wins, but what features of the task determine which methods matter.

---

### Slide 14 — Limitations and Conclusion

These results should be interpreted within the limits of the simulation. The transition and observation models were hand-specified rather than clinically calibrated, the state space is fully observed and discrete, and we did not include a direct bandit baseline for side-by-side comparison. In addition, value iteration operates with a deterministic optimal policy, while the learned agents were evaluated with a small residual epsilon. Even with those limitations, the conclusion is clear: once responsiveness depends on stimulation history, site selection becomes a true reinforcement learning problem. Temporal-difference methods are the most effective of the model-free approaches we tested, and switching costs reveal a genuine control tradeoff that has no simple bandit analogue.

---

### Slide 15 — References

These are the primary references supporting the formulation and the algorithmic choices in the presentation. They include the original Q-learning and Double Q-learning papers, the Expected SARSA analysis, the standard Sutton and Barto text for RL foundations, and the Pineau paper motivating adaptive neurostimulation as a reinforcement learning problem.

# example_datasets

Synthetic example_datasets for the Atto Quantum Decision Engine (QDE) example notebooks. All data is **synthetically generated** — no real-world or proprietary data is included. Values are bounded between 0 and 1 and produced using beta distributions with realistic variation, correlated features, and deliberate edge cases.

## Files

### `creative_problem_solving.csv` (45 rows)

Models how diverse ideas evolve into coherent solutions under ambiguity.

| Column             | Description                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `idea_diversity`   | Breadth of candidate ideas in the belief state                                                |
| `solution_depth`   | Depth of reasoning behind the leading solution (correlated with idea_diversity)               |
| `novelty_score`    | How unconventional the proposed approach is (correlated with idea_diversity)                  |
| `convergence_rate` | Speed at which the state collapses toward a single solution                                   |
| `self_correction`  | Ability to revise direction mid-process (correlated with solution_depth and convergence_rate) |
| `ambiguity_level`  | Degree of unresolved uncertainty in the problem framing                                       |

### `strategic_decision.csv` (43 rows)

Captures the tension between expected value and execution risk in strategic choices.

| Column               | Description                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `expected_value`     | Anticipated upside of the decision                                                              |
| `execution_risk`     | Likelihood of failure during implementation (inversely correlated with expected_value)          |
| `market_uncertainty` | External market unpredictability                                                                |
| `team_confidence`    | Internal team belief in success (correlated with expected_value, inversely with execution_risk) |
| `external_pressure`  | Pressure from stakeholders or competitors                                                       |
| `downside_severity`  | Magnitude of loss if the decision fails (correlated with execution_risk and market_uncertainty) |

### `hiring_outcomes.csv` (45 rows)

Evaluates candidate signals across technical ability, collaboration, and risk.

| Column                 | Description                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `technical_score`      | Strength of domain-specific skills                                                           |
| `communication_score`  | Clarity and effectiveness in communication (correlated with technical_score)                 |
| `consistency_score`    | Reliability across repeated evaluations (correlated with technical and communication scores) |
| `learning_velocity`    | Rate of skill acquisition and adaptation                                                     |
| `collaboration_signal` | Indicators of effective teamwork (correlated with communication_score)                       |
| `risk_flag`            | Warning signals from the evaluation process (generally low)                                  |

### `product_strategy.csv` (41 rows)

Balances growth potential against execution constraints and strategic alignment.

| Column               | Description                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `growth_signal`      | Evidence of market traction or demand                                                     |
| `retention_signal`   | Strength of user/customer retention (correlated with growth_signal)                       |
| `margin_signal`      | Profitability indicator                                                                   |
| `technical_debt`     | Accumulated implementation burden (generally low)                                         |
| `execution_capacity` | Team ability to deliver (correlated with retention_signal, inversely with technical_debt) |
| `strategic_fit`      | Alignment with long-term company direction (correlated with growth and retention signals) |

### `investment_allocation.csv` (43 rows)

Models the trade-off between return expectations, risk tolerance, and liquidity constraints.

| Column              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `expected_return`   | Projected financial return                                  |
| `downside_risk`     | Potential for loss (correlated with expected_return)        |
| `conviction_score`  | Confidence in the investment thesis                         |
| `liquidity_need`    | Requirement for accessible capital (generally low)          |
| `volatility`        | Price fluctuation magnitude (correlated with downside_risk) |
| `macro_uncertainty` | Broader economic unpredictability                           |

## Edge Cases

Each dataset includes deliberate edge-case rows appended at the end:

- **high_conflict**: Opposing signals that create tension (e.g. high expected value but high risk)
- **low_signal**: Near-zero values across all columns — minimal information for the engine
- **extreme_values**: Near-ceiling or near-floor values testing boundary behaviour

## Generation

example_datasets were generated using `numpy` with beta distributions (`np.random.beta`) to produce non-uniform, realistic distributions. Feature correlations are introduced via linear mixing. The generation script is located at `_misc/gen_example_datasets.py`.

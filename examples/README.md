# Examples

Six worked examples showing how the Quantum Decision Engine (QDE) evolves belief states into decisions under uncertainty, plus four deep-dive notebooks that unpack the engine internals. Each notebook follows the core flow:

```
context → belief state → evolving reasoning → decision
```

All example_datasets are synthetic — no real-world or proprietary data is included.

---

## Scenarios

### Multi-Outcome Examples

These scenarios model a superposition of competing outcomes that collapse into a single decision.

#### 1. Creative Problem Solving

> **Decision:** Which creative direction should we pursue?

Diverse ideas coexist in superposition. Signals like constraints, inspiration, and feedback create interference — amplifying some directions and suppressing others — until the state collapses to a coherent solution.

**Why QDE fits:** Creative reasoning is inherently non-linear. Ideas don't compete on a score; they interfere with each other, and the "best" answer emerges from that interaction.

📓 [`1_creative_problem_solving.ipynb`](1_creative_problem_solving.ipynb)

#### 2. Strategic Decision Under Uncertainty

> **Decision:** Which strategy should we commit to under ambiguous conditions?

Competing strategies (e.g. expand, consolidate, pivot) exist in superposition. Ambiguous market and internal signals apply operators that shift the belief state. Signal order matters — receiving bad news before good news leads to a different outcome than the reverse.

**Why QDE fits:** Strategic choices are sensitive to framing, timing, and signal order — all properties that classical expected-value models struggle with. Non-commutativity is central here.

📓 [`2_strategic_decision_under_uncertainty.ipynb`](2_strategic_decision_under_uncertainty.ipynb)

#### 3. Hiring Outcome Spectrum

> **Decision:** Which candidate-role pairing should we select?

Candidate signals (technical skill, communication, collaboration) form interfering amplitudes rather than independent scores. A strong collaboration signal can constructively interfere with a moderate technical score, shifting the belief state in ways a weighted average would miss.

**Why QDE fits:** Hiring decisions involve signals that don't combine additively. Interference captures how one strength can amplify — or cancel — another.

📓 [`3_hiring_outcome_spectrum.ipynb`](3_hiring_outcome_spectrum.ipynb)

---

### Dynamic Examples

These scenarios emphasise how the order and timing of signals change the final decision.

#### 4. Product Strategy Decision

> **Decision:** Which roadmap direction should we prioritise?

Growth, retention, and margin signals arrive as operators applied to the belief state. Applying "strong growth signal" before "high technical debt" produces a different evolved state than the reverse order — demonstrating non-commutativity in practice.

**Why QDE fits:** Product decisions are path-dependent. The same set of inputs can lead to different conclusions depending on the order they're considered. QDE models this naturally.

📓 [`4_product_strategy_decision.ipynb`](4_product_strategy_decision.ipynb)

---

### Multi-Dimensional Example

This scenario extends the engine to allocate across multiple outcomes simultaneously.

#### 5. Investment Allocation

> **Decision:** How should capital be distributed across asset classes?

Unlike the binary or categorical decisions above, this example models a multi-asset belief state where measurement produces a probability distribution over allocations rather than a single winner. Signals like expected return, volatility, and macro uncertainty evolve the state before collapse.

**Why QDE fits:** Portfolio allocation is a naturally multi-dimensional problem. The belief state spans multiple assets, and measurement produces a distribution — not a point estimate — making it a direct analogue of quantum measurement.

📓 [`5_investment_allocation.ipynb`](5_investment_allocation.ipynb)

---

### Multi-Outcome + Dynamic Example

This scenario combines multi-outcome decisioning with dynamic evolution over repeated interactions.

#### 6. Game Theory Decisioning

> **Decision:** Given another actor's likely strategy, incentive structure, trust level, and payoff asymmetry, what strategy should we choose?

Seven strategies — cooperate, defect, signal_commitment, delay, retaliate, negotiate, and mixed_strategy — coexist in superposition. Payoff incentives, trust, aggression, and retaliation risk create interference patterns that shift probability mass across strategies. When incentives genuinely conflict, the belief state resists collapsing, keeping `mixed_strategy` viable.

**Why QDE fits:** Game-theoretic decisions are inherently interdependent — an actor's optimal strategy depends on the opponent's likely response, not just isolated payoffs. Strategies interfere rather than compete additively, and the order of information (trust signal before or after retaliation risk) genuinely changes the decision. QDE captures mixed strategies as a natural consequence of unresolved superposition rather than an explicit randomisation device.

Demonstrates:

- **Multi-outcome** — seven strategies held simultaneously in superposition
- **Dynamic evolution** — beliefs shift over repeated rounds of interaction
- **Mixed strategies** — conflicting incentives prevent premature collapse
- **Interdependent decision-making** — opponent signals reshape the belief landscape
- **ScenarioRegistry integration** — custom game-theory adapter registered and retrieved by name

📓 [`6_game_theory_decisioning.ipynb`](6_game_theory_decisioning.ipynb)

---

### Deep-Dive Examples

These notebooks unpack the QDE internals — low-level components, visualisation, custom scenarios, and the learning layer.

#### 7. Anatomy of a QDE Decision

> **Purpose:** Step-by-step walkthrough of every core component.

Covers `AttoState` construction (uniform, from_probabilities, basis, blend), `AttoOperator` mechanics (phase_shift, interference, rotation, from_signal, compose), `AttoDynamics` evolution (evolve, evolve_stepwise, sensitivity), `AttoMeasurement` collapse (argmax, sample, softmax, entropy), and the `AttoEngine` pipeline.

**Why this matters:** The high-level `AttoModel` hides these details. This notebook makes the QDE reasoning process fully transparent — every transformation, every intermediate state, every non-commutative effect.

📓 [`7_anatomy_of_a_qde_decision.ipynb`](7_anatomy_of_a_qde_decision.ipynb)

#### 8. Interference Visualisation

> **Purpose:** 3D wave interference surfaces that show how strategies interact and evolve.

Covers `plot_interference` (single state), `plot_trajectory` (state sequence), `PipelineVisualizer` (static and animated pipeline rendering), `run_pipeline` (introspection), and `compute_interference` (raw numpy grids).

**Why this matters:** The interference surface is the visual signature of QDE. Peaks show where the engine is converging; valleys show where destructive interference has eliminated strategies.

📓 [`8_interference_visualisation.ipynb`](8_interference_visualisation.ipynb)

#### 9. Building a Custom Scenario

> **Purpose:** Connect a new domain to QDE by subclassing `ScenarioAdapter`.

Implements all four lifecycle hooks (`map_context`, `build_operators`, `apply_constraints`, `decode_decision`), registers the scenario with `ScenarioRegistry`, and runs the full pipeline end-to-end with `CoreConfig` and `ScenarioConfig`.

**Why this matters:** The core engine is domain-agnostic. This notebook shows how to bridge any domain's signals into the mathematical framework without modifying the engine.

📓 [`9_building_a_custom_scenario.ipynb`](9_building_a_custom_scenario.ipynb)

#### 10. Training & Persistence

> **Purpose:** How the learning layer calibrates belief states and how models are saved/loaded.

Covers `EvidenceMapper` (features → belief states), `AttoTrainer` (mini-batch gradient descent calibration with `TrainingConfig`), `predict_proba` (full probability distributions), and `save` / `load` (JSON serialisation with exact reconstruction).

**Why this matters:** The mapper is the bridge between raw evidence and the QDE framework. This notebook shows how it learns, how to diagnose convergence, and how to persist trained models.

📓 [`10_training_and_persistence.ipynb`](10_training_and_persistence.ipynb)

---

## How to Run

1. **Install Atto**

   ```bash
   pip install atto
   ```

2. **Open Jupyter**

   ```bash
   jupyter notebook
   ```

   Or open the notebook directly in VS Code.

3. **Run a notebook**

   Navigate to the `examples/` folder, open any `.ipynb` file, and run cells top-to-bottom. All examples use only `numpy` and the `atto` package — no additional dependencies required.

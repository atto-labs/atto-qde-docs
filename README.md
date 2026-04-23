<p align="center">
  <img src="assets/docs-logo.png" alt="Atto — Quantum Decision Engine" width="600">
</p>

# Quantum Decision Engine (QDE)

### How beliefs evolve into decisions under uncertainty

QDE is a decision framework that models how competing signals interact, evolve, and resolve into actionable outcomes. Built on principles drawn from quantum information theory, it replaces static prediction pipelines with a structured reasoning process that handles uncertainty natively.

---

## The Problem

Most decision systems follow a familiar pattern:

**features → probability → decision**

This works when relationships are stable, inputs are independent, and outcomes are cleanly separable. In practice, real-world decisions rarely meet those conditions.

What breaks down:

- **Competing strategies** — multiple valid approaches exist simultaneously, and selecting one too early discards valuable structure.
- **Interaction between signals** — inputs are not independent. The meaning of one signal changes depending on what else is present.
- **Order effects** — the sequence in which information arrives affects the outcome. Traditional models treat inputs as commutative; real reasoning does not.
- **Structured uncertainty** — uncertainty is not noise to be minimised. It carries information about the shape of the decision space.

When these dynamics are present, flattening inputs into a feature vector and mapping to a probability discards exactly the structure that matters most.

---

## The Shift

QDE reframes the decision process:

**signals → belief state → evolving reasoning → decision**

Instead of collapsing information into a single prediction early, QDE maintains a structured belief state that evolves through reasoning steps before resolving into a decision.

Core concepts:

- **Belief state** — a structured representation of what is currently believed, including competing interpretations held simultaneously.
- **Superposition** — multiple candidate strategies or hypotheses coexist within the belief state, weighted but not yet resolved.
- **Interference** — signals interact with one another, amplifying consistent reasoning paths and suppressing contradictory ones. This is where the real reasoning happens.
- **Collapse** — when a decision is required, the belief state resolves into a concrete outcome. The process of getting there is traceable and interpretable.

The key insight is that decisions are not static mappings. They are processes. QDE models the process.

---

## Conceptual Flow

```text
Context (signals)
      ↓
Belief State (superposition)
      ↓
Operators (interference / reasoning)
      ↓
Measurement (collapse)
      ↓
Decision
```

---

## Intuition

```
RAG = what is true / relevant
Mapper = what I currently believe
QDE = how beliefs evolve into decisions
```

| System          | How it thinks             |
| --------------- | ------------------------- |
| Traditional ML  | Static mapping            |
| Tree-of-thought | Branching reasoning       |
| QDE             | Continuous evolving state |

Traditional ML maps inputs to outputs. Tree-of-thought explores discrete branches. QDE maintains a continuous state that evolves under the influence of incoming signals, capturing interactions and order effects that discrete approaches miss.

---

## Why This Matters

- **Order sensitivity** — the sequence of information intake affects the final decision, as it does in real reasoning.
- **Signal interaction** — inputs are not treated independently. The framework captures how one signal modifies the meaning of another.
- **Natural handling of ambiguity** — uncertainty is represented structurally, not approximated as a confidence score.
- **Interpretable decision formation** — the path from signals to decision is traceable through the evolution of the belief state.
- **Complex reasoning without deep neural networks** — structured decision-making emerges from the mathematical framework itself, not from large parameter counts.

---

## Developer Quick Start

There are two entry points depending on the level of control you need.

### High-level: AttoModel (scikit-learn style)

The `AttoModel` wraps the full pipeline behind a familiar `fit` / `predict` / `predict_proba` API.

```python
import numpy as np
from atto.api.model import AttoModel
from atto.config.core import CoreConfig

config = CoreConfig(
    state_dim=3,
    action_labels=["strategy_a", "strategy_b", "strategy_c"],
)

model = AttoModel(core_config=config, measurement_method="argmax")
model.fit(X_train, y_train)

predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

`X` is a 2-D NumPy array of context signals (shape `n_samples x n_features`). `y` is a 1-D array of action indices (integers). After fitting, the model can be persisted and reloaded:

```python
model.save("model.json")
loaded = AttoModel.load("model.json")
```

### Low-level: AttoEngine (step-by-step control)

The `AttoEngine` gives direct access to state initialisation, operator sequencing, evolution, and measurement.

```python
from atto import AttoEngine, AttoOperator

engine = AttoEngine(
    dimension=3,
    labels=["strategy_a", "strategy_b", "strategy_c"],
    measurement_method="argmax",
)

# Add signal operators to the evolution sequence
engine.add_operator(AttoOperator.phase_shift(3, [0.5, 0.0, -0.3]))
engine.add_operator(AttoOperator.interference(3, i=0, j=2, angle=0.4))

# Run the full pipeline: evolve belief state, then collapse to a decision
decision = engine.decide()

print(decision.action)        # selected strategy index
print(decision.label)         # human-readable label
print(decision.confidence)    # probability at collapse time
print(decision.probabilities) # full distribution over strategies
```

---

## SDK Reference

### Core Components

**AttoState** — represents a belief state as a complex amplitude vector in a Hilbert space. Each dimension corresponds to a candidate strategy.

```python
from atto.core.state import AttoState

# Uniform superposition (maximum uncertainty)
state = AttoState.uniform(3, labels=["a", "b", "c"])

# From a prior probability distribution
state = AttoState.from_probabilities([0.6, 0.3, 0.1])

# Pure state (full commitment to one strategy)
state = AttoState.basis(0, dimension=3)

# Blend multiple belief states (coherent superposition)
blended = AttoState.blend([state_1, state_2], weights=[0.7, 0.3])

state.probabilities  # Born-rule probabilities
state.amplitudes     # complex amplitude vector
state.dimension      # number of strategies
```

**AttoOperator** — a unitary matrix that transforms belief states. Encodes how a signal reshapes the superposition of strategies.

```python
from atto.core.operator import AttoOperator

# Phase shift: rotates each strategy dimension independently
op = AttoOperator.phase_shift(3, [0.5, 0.0, -0.3])

# Interference: mixes two strategies via a Givens rotation
op = AttoOperator.interference(3, i=0, j=2, angle=0.4)

# 2D rotation (for two-strategy systems)
op = AttoOperator.rotation(angle=0.8)

# From a raw signal vector (auto-generates a unitary operator)
op = AttoOperator.from_signal([0.1, -0.3, 0.5])

# Manual composition (non-commutative)
combined = op_a.compose(op_b)  # applies op_b first, then op_a

# Apply to a state directly
evolved_state = op.apply(state)
```

**AttoDynamics** — manages an ordered sequence of operators and evolves a state through them.

```python
from atto.core.dynamics import AttoDynamics

dynamics = AttoDynamics()
dynamics.add_operator(op_1)
dynamics.add_operator(op_2)

evolved = dynamics.evolve(state)

# Inspect every intermediate state
trajectory = dynamics.evolve_stepwise(state)  # [initial, after_op1, after_op2]

# Measure sensitivity to operator ordering
dynamics.sensitivity(state)  # 0.0 = order-independent, higher = order-sensitive
```

**AttoMeasurement** — collapses a belief state into a concrete decision.

```python
from atto.core.measurement import AttoMeasurement

m = AttoMeasurement(method="argmax")   # deterministic
m = AttoMeasurement(method="sample")   # stochastic (Born-rule sampling)
m = AttoMeasurement(method="softmax", temperature=0.5)  # temperature-controlled

decision = m.collapse(evolved_state)

decision.action        # int: selected strategy index
decision.label         # str or None: human-readable label
decision.confidence    # float: probability of the selected strategy
decision.probabilities # ndarray: full distribution at collapse time

# Entropy of the belief state (remaining uncertainty)
AttoMeasurement.entropy(state)
```

### Configuration

**CoreConfig** — defines the geometry of the belief state.

```python
from atto.config.core import CoreConfig

config = CoreConfig(
    state_dim=4,
    action_labels=["hold", "buy", "sell", "hedge"],
    use_complex=False,
    normalize_state=True,
)
```

**TrainingConfig** — calibration hyper-parameters.

```python
from atto.config.training import TrainingConfig

training = TrainingConfig(
    learning_rate=0.01,
    epochs=100,
    batch_size=32,
    random_seed=42,
)
```

**ScenarioConfig** — domain metadata.

```python
from atto.config.scenario import ScenarioConfig

scenario = ScenarioConfig(
    scenario_name="finance",
    metadata={"market": "equities", "region": "EU"},
)
```

### Learning Layer

**EvidenceMapper** — transforms raw feature vectors into belief states via a learned linear projection.

```python
from atto.learning.mapper import EvidenceMapper

mapper = EvidenceMapper(n_features=10, n_strategies=3, random_seed=42)

# Map a single observation to a belief state
state = mapper.map(feature_vector, labels=["a", "b", "c"])

# Batch probabilities
probs = mapper.map_batch(X)  # shape (n_samples, n_strategies)
```

**AttoTrainer** — calibrates the EvidenceMapper from historical evidence-outcome pairs using mini-batch gradient descent.

```python
from atto.learning.trainer import AttoTrainer

trainer = AttoTrainer(mapper=mapper, config=training)
result = trainer.train(features=X_train, targets=y_train)

result.final_loss       # float
result.loss_history     # list[float] per epoch
result.epochs_completed # int
```

### Building a Custom Scenario

To connect a new domain to QDE, subclass `ScenarioAdapter` and implement the lifecycle hooks:

```python
from atto.api.scenario import ScenarioAdapter
from atto.core.operator import AttoOperator
from atto.core.state import AttoState
import numpy as np

class MyScenarioAdapter(ScenarioAdapter):

    def map_context(self, x):
        """Translate a context vector into an initial belief state."""
        return AttoState.from_probabilities(x[:3])

    def build_operators(self, x):
        """Build operators from context signals."""
        return [
            AttoOperator.from_signal(x[3:6]),
            AttoOperator.interference(3, i=0, j=2, angle=float(x[6])),
        ]

    def apply_constraints(self, state):
        """Optional: enforce domain-specific constraints."""
        return state  # default: no constraints

    def decode_decision(self, action, labels):
        """Optional: map action index to domain output."""
        return labels[action] if labels else str(action)
```

Register it so the engine can look it up by name:

```python
from atto.api.registry import ScenarioRegistry

registry = ScenarioRegistry()
registry.register("my_scenario", MyScenarioAdapter)

adapter = registry.get("my_scenario")
print(registry.list_scenarios())  # ["my_scenario"]
```

---

## Visualization

Atto ships a `viz` module that renders the QDE decision process as 3D wave interference surfaces. Each strategy acts as a wave source — their complex amplitudes create an interference pattern that reshapes as signals arrive and the belief state evolves.

### PipelineVisualizer

**PipelineVisualizer** — wraps a `ScenarioAdapter` and renders the full pipeline (initial belief state, signal operators, constraints, collapse) as 3D interference surfaces.

```python
import numpy as np
from atto.viz import PipelineVisualizer

viz = PipelineVisualizer(
    adapter,
    signal_names=["Signal A", "Signal B", "Signal C"],
)

context = np.array([0.85, 0.15, 0.70, 0.90])
```

### Static Pipeline Trajectory

One panel per stage — initial belief, after each signal operator, after constraints, and collapse:

```python
fig = viz.plot_static(context)
```

### Animated Pipeline

3D animation of the interference surface evolving through each stage, with a probability bar chart tracking strategy distributions:

```python
# In a script or interactive window
viz.show(context)

# In a Jupyter notebook
from IPython.display import HTML
anim = viz.animate(context)
HTML(anim.to_jshtml())
```

### Single Belief State

Plot the interference pattern of any `AttoState`:

```python
from atto.core.state import AttoState
from atto.viz import plot_interference

state = AttoState.uniform(3, labels=["strategy_a", "strategy_b", "strategy_c"])
plot_interference(state, title="Uniform superposition")
```

### Pipeline Introspection

Access intermediate states for custom analysis or visualization:

```python
states, stage_names, decision = viz.run_pipeline(context)

for name, state in zip(stage_names, states):
    print(f"{name}: P = {state.probabilities}")
print(f"Decision: {decision.label} (confidence={decision.confidence:.3f})")
```

| Function / Class                   | Purpose                                                 |
| ---------------------------------- | ------------------------------------------------------- |
| `PipelineVisualizer(adapter)`      | One-call wrapper — runs pipeline and visualizes         |
| `.animate(context)`                | Returns `FuncAnimation` (3D surface + probability bars) |
| `.show(context)`                   | Animate + `plt.show()`                                  |
| `.plot_static(context)`            | Multi-panel static figure                               |
| `.run_pipeline(context)`           | Returns `(states, stage_names, decision)`               |
| `plot_interference(state)`         | Single state → 3D surface                               |
| `plot_trajectory(states)`          | Sequence of states → side-by-side panels                |
| `compute_interference(amplitudes)` | Raw numpy computation → `(X, Y, Z)`                     |

---

## Installation

The core engine is distributed as a private package. Requires Python 3.10+ with NumPy and Pydantic.

```bash
pip install git+https://github.com/atto-labs/atto-qde.git
```

With visualization support:

```bash
pip install "atto-qde[viz] @ git+https://github.com/atto-labs/atto-qde.git"
```

Private access required. Contact the maintainers to request access.

Dependencies:

- `numpy >= 1.24`
- `pydantic >= 2.0`

---

## Architecture Overview

The system is organised into four layers:

- **Atto** — the core engine. Implements the mathematical framework for belief state evolution, interference, and collapse. Private.
  - `AttoState`, `AttoOperator`, `AttoDynamics`, `AttoMeasurement`
- **QDE** — the decision system built on top of Atto. Defines how signals map to belief states and how decisions are extracted.
  - `AttoEngine`, `AttoModel`
- **Learning** — calibration layer that learns how evidence maps to belief states and how operators should be parameterised.
  - `EvidenceMapper`, `AttoTrainer`, `AttoCalibrator`
- **Scenarios** — domain adapters that configure QDE for specific use cases. These define signal schemas, operator configurations, and output formats.
  - `ScenarioAdapter` (abstract), `ScenarioRegistry`
- **API / SDK** — the developer interface. Provides model instantiation, prediction, traceability, and persistence.
  - `AttoModel.fit()`, `.predict()`, `.predict_proba()`, `.save()`, `.load()`

Atto implements QDE. The engine and the framework are tightly coupled by design, with domain-specific behaviour isolated in the scenario layer.

```text
┌─────────────────────────────────────┐
│           API / SDK Layer           │
│  AttoModel  ·  AttoEngine           │
├─────────────────────────────────────┤
│          Scenario Layer             │
│  ScenarioAdapter  ·  Registry       │
├─────────────────────────────────────┤
│          Learning Layer             │
│  EvidenceMapper  ·  AttoTrainer     │
├─────────────────────────────────────┤
│            Core Engine              │
│  AttoState · AttoOperator           │
│  AttoDynamics · AttoMeasurement     │
└─────────────────────────────────────┘
```

---

## Design Principles

- **Domain-agnostic core engine** — the mathematical framework is independent of any specific application area.
- **Separation of math and scenarios** — the core engine handles state evolution; domain logic lives in configurable scenario adapters.
- **Interpretable reasoning via state evolution** — decisions are not black boxes. The belief state trajectory is inspectable at every step.
- **Extensible across industries** — new domains are supported by defining new scenario configurations, not by modifying the engine.

---

## Roadmap

### Human Decision Intelligence

- Interview & hiring decision modelling
- Problem solving and creative reasoning analysis
- Behavioural and personality state modelling
- Decision-making under uncertainty
- Cognitive bias and inconsistency detection
- Leadership and judgement evaluation
- Human–AI interaction modelling

### Systems

- Financial markets
- Autonomous agents
- Portfolio optimisation
- Energy systems

### Closed-loop Automated Learning System

- Continuous scenario learning
- Adaptive model registry
- Autonomous scenario expansion

---

## Public vs Private Repositories

This repository (`atto-qde-docs`) contains:

- Conceptual overview
- Developer guidance
- Examples
- Onboarding material

The core implementation lives in:

- `atto-qde` (private)

This separation ensures that documentation, examples, and conceptual material are openly accessible while the engine internals remain protected.

---

## Getting Access

The core engine is available under controlled access. To request access to the private `atto-qde` repository, contact the maintainers directly. Include a brief description of your intended use case and we will follow up.

---

QDE represents a shift from predicting outcomes to modelling how decisions form.

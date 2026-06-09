# `AttoEngine`

`atto.AttoEngine` is the low-level entry point. It owns the belief
state, the operator queue, and the measurement strategy, and routes
each decision through the configured licence validator and usage
emitter.

## Construction

```python
AttoEngine(
    dimension: int,
    labels: list[str] | None = None,
    measurement_method: str = "argmax",
    *,
    validator: LicenceValidator | None = None,
    emitter: UsageEmitter | None = None,
    org_id: str = "",
    scenario: str = "",
    auto_wire_runtime: bool = True,
)
```

| Parameter            | Purpose                                                                                                              |
| -------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `dimension`          | Number of candidate strategies in the decision space.                                                                |
| `labels`             | Optional list of human-readable strategy names; length must equal `dimension`.                                       |
| `measurement_method` | `"argmax"` (default) selects the highest-amplitude strategy. Other strategies plug into `AttoMeasurement`.           |
| `validator`          | Override the licence validator. Defaults to the env-driven choice from `build_default_runtime`.                      |
| `emitter`            | Override the usage emitter. Defaults likewise.                                                                       |
| `org_id`             | Org ID forwarded to the validator and emitter. Falls back to the runtime default.                                    |
| `scenario`           | Free-form scenario tag attached to every emitted `UsageEvent`.                                                       |
| `auto_wire_runtime`  | Set to `False` to disable licence/usage wiring entirely (bare engine, no env-reads). The engine then runs unmetered. |

## Methods

### `add_operator(op: AttoOperator) -> None`

Append an operator to the evolution sequence. Free — does not emit a
usage event.

### `set_initial_state(state: AttoState) -> None`

Override the initial belief state. The state's dimension must match the
engine's. Free.

### `decide() -> AttoDecision`

Run the full pipeline: licence check → evolve initial state through
the operator queue → measure → emit usage event → return the decision.

Returns an `AttoDecision` with `action` (int index), `label` (string),
`confidence` (float in `[0, 1]`), and `probabilities` (full
distribution).

Raises any of `LicenceError`, `PlanUpgradeRequired`, `QuotaExhausted`,
`InvalidApiKey` — see [errors](../errors.md).

### `decide_from(state: AttoState) -> AttoDecision`

Same as `decide()` but starts from an explicit belief state instead of
the engine's stored initial state. Counts as one decision.

### `evolve() -> AttoState`

Run the operators against the initial state and return the resulting
belief state without measuring or emitting. Free; intended for
inspection and debugging.

## Properties

- `dimension: int` — read-only.
- `initial_state: AttoState` — read-only view of the configured initial
  state.

## Example

```python
from atto import AttoEngine, AttoOperator

engine = AttoEngine(
    dimension=3,
    labels=["charge", "hold", "discharge"],
    org_id="org_123",
    scenario="battery_dispatch",
)
engine.add_operator(AttoOperator.phase_shift(3, [0.5, 0.0, -0.3]))
engine.add_operator(AttoOperator.interference(3, i=0, j=2, angle=0.4))

decision = engine.decide()
print(decision.label, f"{decision.confidence:.3f}")
```

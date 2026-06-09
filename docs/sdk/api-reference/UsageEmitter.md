# `UsageEmitter`

Defined in `atto.usage.emitter`. Post-flight emission invoked once per
successful `engine.decide()`.

## ABC

```python
class UsageEmitter(ABC):
    @abstractmethod
    def emit(self, event: UsageEvent) -> None: ...

    async def aclose(self) -> None:
        """Flush any pending events. Override in async implementations."""
```

`emit()` must be **non-blocking**: it cannot do network I/O on the
caller's thread. Implementations buffer to an in-memory queue and
flush in the background.

## `UsageEvent`

Defined in `atto.usage.schemas`.

| Field              | Type       | Notes                                                |
| ------------------ | ---------- | ---------------------------------------------------- |
| `event_id`         | `str`      | UUIDv4 (auto-generated).                             |
| `org_id`           | `str`      | Required.                                            |
| `product`          | `str`      | Always `"core"` for this SDK.                        |
| `op_type`          | `OpType`   | `decision` / `warmup` / `calibration`.               |
| `op_id`            | `str`      | UUIDv7-style operation identifier.                   |
| `ts`               | `datetime` | UTC timestamp.                                       |
| `latency_ms`       | `int`      | Time the operation took.                             |
| `scenario`         | `str`      | Optional scenario tag from the engine.               |
| `sdk_version`      | `str`      | Populated by `engine_hooks` from `atto.__version__`. |
| `cumulative_after` | `int`      | Per-process counter; advisory.                       |

Only `op_type=decision` events affect billing; `warmup` and
`calibration` are priced at zero and used for telemetry.

## Implementations

### `NoOpUsageEmitter`

Discards events but appends them to `emitter.events` for assertion in
tests.

```python
emitter = NoOpUsageEmitter()
engine = AttoEngine(dimension=3, emitter=emitter)
engine.decide()
assert len(emitter.events) == 1
```

### `ConsoleUsageEmitter`

Pushes events onto a bounded queue and flushes batches to
`POST /api/v1/usage/events`. Failures are logged at `WARNING` and
dropped; the hot path never raises.

Construct via env-driven defaults:

```python
from atto.usage.emitter import ConsoleUsageEmitter

emitter = ConsoleUsageEmitter()  # reads ATTO_API_BASE / ATTO_API_KEY / ATTO_ORG_ID
```

…or explicitly:

```python
emitter = ConsoleUsageEmitter(
    api_base="https://console.atto-qde.com",
    api_key="atto_live_...",
    org_id="org_...",
)
```

Call `await emitter.aclose()` during shutdown to flush pending events
within a five-second deadline.

# Licence and usage

This page describes what the SDK does on each `decide()` call, how it
interacts with the console, and how it degrades when the network is
unavailable.

## Pre-flight licence check

Before each decision, `LicenceValidator.check(org_id)` is invoked.

- `NoOpLicenceValidator` (offline mode) always returns `state=valid`
  with no network call.
- `ConsoleLicenceValidator` (hosted mode) calls
  `GET /api/v1/licence/{org_id}?product=core`. The response is cached
  in-process for ~30 seconds to avoid one round-trip per decision.

The response carries the current state and live counters, mapped onto
`LicenceStatus.metadata`:

| Field                 | Type | Meaning                                               |
| --------------------- | ---- | ----------------------------------------------------- |
| `decisions_used`      | int  | Decisions consumed in the current period.             |
| `decisions_remaining` | int? | Decisions left (None on metered — open-ended).        |
| `charge_cents_next`   | int  | Cost of the next decision, in integer cents.          |
| `upgrade_required`    | bool | Free quota exhausted — must upgrade before next call. |
| `checkout_url`        | str? | Stripe Checkout URL when no payment method on file.   |

The validator raises (rather than returning a non-valid state) for the
four conditions described in [errors](./errors.md).

## Post-flight usage emission

After a successful decision, the SDK queues a `UsageEvent`:

```jsonc
{
  "event_id": "<uuid4>",
  "org_id": "org_...",
  "product": "core",
  "op_type": "decision",
  "ts": "2026-05-06T12:34:56.789Z",
  "latency_ms": 4,
  "scenario": "<optional scenario name>",
  "sdk_version": "0.2.0",
  "cumulative_after": 17,
}
```

`ConsoleUsageEmitter` batches events on a bounded in-memory queue and
flushes them with a background task to
`POST /api/v1/usage/events`. The flush is best-effort:

- The hot path (`engine.decide()`) never blocks on the network.
- On HTTP failure, the batch is logged at `WARNING` and dropped.
  At-least-once delivery is **not** guaranteed; the console treats
  events as advisory.
- On clean shutdown, call `await emitter.aclose()` to flush within a
  five-second deadline.

`NoOpUsageEmitter` discards events but retains them in `emitter.events`
for assertions in tests.

## Failure modes

| Condition                                    | What the SDK does                               |
| -------------------------------------------- | ----------------------------------------------- |
| Network timeout / 5xx on licence check       | Raises `LicenceError` (transport).              |
| 401 / 403 from console                       | Raises `InvalidApiKey`.                         |
| `upgrade_required=true`                      | Raises `PlanUpgradeRequired(checkout_url=...)`. |
| `state=suspended` or balance zero            | Raises `QuotaExhausted`.                        |
| Network failure on usage POST                | Logs `WARNING`, drops the batch. No raise.      |
| `httpx` not installed (no `[console]` extra) | Falls back to NoOp pair at runtime.             |

The licence check is **fail-closed** — a transport failure aborts the
decision rather than letting it through unmetered. If your application
needs fail-open semantics, wrap `engine.decide()` in your own
`try/except LicenceError` and fall back to a default action.

## Disabling hosted mode

To run in offline mode for tests, CI, or air-gapped deployments,
construct the engine with explicit no-op pieces:

```python
from atto import AttoEngine, NoOpLicenceValidator, NoOpUsageEmitter

engine = AttoEngine(
    dimension=3,
    validator=NoOpLicenceValidator(),
    emitter=NoOpUsageEmitter(),
)
```

Or simply unset `ATTO_API_KEY` / `ATTO_ORG_ID` — the default runtime
factory selects the no-op pair when either is missing.

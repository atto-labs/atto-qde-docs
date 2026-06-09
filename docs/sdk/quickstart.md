# Quickstart

A five-minute path from `pip install` to a recorded decision in the
console.

## 1. Create an org and an API key

1. Sign up at [console.atto-qde.com/signup](https://console.atto-qde.com/signup). The
   default plan is `core_free` (100 decisions/month, then upgrade).
2. From the dashboard, open **API keys** and click **Create key**. In
   the **Products** picker, ensure **`core`** is selected.
3. Copy the key (shown once) and your **Org ID** from the dashboard
   header.

> Energy-industry signups are held for manual review. See the
> [authentication notes](./authentication.md#energy-industry-signups).

## 2. Install the SDK

```bash
pip install 'atto-qde[console]'
```

The `[console]` extra pulls in `httpx`. Without it, the SDK still
imports and runs, but only in offline mode.

## 3. Configure the environment

```bash
export ATTO_API_KEY=atto_test_...
export ATTO_ORG_ID=org_...
# Optional — defaults to https://console.atto-qde.com
export ATTO_CONSOLE_BASE_URL=https://console.atto-qde.com
```

## 4. Make a decision

```python
# minimal.py
from atto import AttoEngine, AttoOperator

engine = AttoEngine(dimension=3, labels=["charge", "hold", "discharge"])
engine.add_operator(AttoOperator.phase_shift(3, [0.5, 0.0, -0.3]))

decision = engine.decide()
print(decision.label, decision.confidence)
```

```bash
python minimal.py
```

The first call performs a licence check against the console; the result
is cached for ~30 seconds. After the call returns, a usage event is
queued and POSTed to `/api/v1/usage/events` in the background.

## 5. Confirm in the console

Open the dashboard and navigate to **Plan → Core**. You should see:

- The decision counter incremented by 1 (e.g. `1 / 100 used`).
- The most recent usage event listed under **Recent activity**.

If neither shows up, check the [errors guide](./errors.md) or set
`ATTO_LOG_LEVEL=DEBUG` to surface the underlying HTTP responses.

## What just happened

```
your code              atto-qde SDK                  console.atto-qde.com
─────────              ────────────                  ────────────────────
engine.decide() ─────► LicenceValidator.check ────► GET  /api/v1/licence/{org}?product=core
                       (TTL-cached 30s)              ◄── 200 { state: valid, decisions_used: 1, ... }
                       run pipeline
                       UsageEmitter.emit ──────────► POST /api/v1/usage/events  (queued)
                       return AttoDecision           ◄── 202 accepted
```

For a more involved walkthrough — explicit validator construction,
offline fallback, error handling — see the
[examples](../../examples/sdk/).

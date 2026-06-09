# atto-qde SDK

The `atto-qde` Python SDK ships the Quantum Decision Engine as an installable
library. Customers run it in their own process: there is no hosted API to
call into. The SDK has two operating modes:

| Mode        | What it does                                                                 | When to use                            |
| ----------- | ---------------------------------------------------------------------------- | -------------------------------------- |
| **Offline** | No network calls. `NoOpLicenceValidator` always returns valid; no metering.  | OSS users, local development, CI.      |
| **Hosted**  | Pre-flight licence check against `console.atto-qde.com`; usage events POSTed | Commercial use under the Core product. |

Selection is purely environmental — set `ATTO_API_KEY` and `ATTO_ORG_ID`
to opt into hosted mode; leave them unset for offline mode.

## Install

```bash
# Offline mode only (no network deps)
pip install atto-qde

# Hosted mode (adds httpx for the console client)
pip install 'atto-qde[console]'
```

## What you get

- `AttoEngine` — low-level engine with `decide()` / `decide_from()` /
  `evolve()` entry points.
- `AttoModel` — scikit-learn-style `fit` / `predict` wrapper.
- `LicenceValidator` + `UsageEmitter` ABCs and their `NoOp` and `Console`
  implementations.
- `build_default_runtime()` — env-driven factory that returns the right
  pair for the current process.

## Where to next

- [Quickstart](./quickstart.md) — five-minute path from `pip install` to
  a first decision recorded in the console.
- [Authentication](./authentication.md) — how API keys and org IDs map
  to hosted-mode requests.
- [Pricing](./pricing.md) — Free vs. Metered, what counts as a
  decision, what doesn't.
- [Licence and usage](./licence-and-usage.md) — what the SDK does on
  each `decide()` call and how it degrades.
- [Errors](./errors.md) — the four exception types the SDK can raise.
- [API reference](./api-reference/AttoEngine.md).

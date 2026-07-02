# `LicenceValidator`

Defined in `atto.licence.validator`. Pre-flight check invoked once per
`engine.decide()` call.

## ABC

```python
class LicenceValidator(ABC):
    @abstractmethod
    def check(self, organisation_id: str) -> LicenceStatus: ...
```

`check()` returns a `LicenceStatus` for valid licences and **raises**
for any other state (see [errors](../errors.md)). This shape lets call
sites use a simple `try/except LicenceError` rather than branching on
enum values.

## `LicenceStatus`

A frozen Pydantic model describing the outcome of one check.

| Field             | Type             | Notes                                                     |
| ----------------- | ---------------- | --------------------------------------------------------- |
| `organisation_id` | `str`            | Echoed back from the request.                             |
| `state`           | `LicenceState`   | `valid` / `expired` / `missing` / `suspended` / `unknown` |
| `checked_at`      | `datetime` (UTC) | When the check completed.                                 |
| `expires_at`      | `datetime?`      | When the licence expires (None on metered).               |
| `features`        | `list[str]`      | Feature flags granted to the org.                         |
| `metadata`        | `dict[str, Any]` | Core-specific counters; see below.                        |
| `message`         | `str`            | Human-readable detail.                                    |

### Convenience properties

| Property              | Reads from `metadata[...]`        |
| --------------------- | --------------------------------- |
| `is_valid`            | (derived from `state`)            |
| `decisions_used`      | `decisions_used`                  |
| `decisions_remaining` | `decisions_remaining`             |
| `charge_cents_next`   | `charge_cents_next`               |
| `upgrade_required`    | `upgrade_required` (cast to bool) |
| `checkout_url`        | `checkout_url`                    |

## Implementations

### `NoOpLicenceValidator`

Returns a local free-tier token (`state=valid`) with no network call.
The compiled binary still enforces a 100-decision-per-month cap via a
signed monotonic counter. Used in offline mode and in tests.

### `ConsoleLicenceValidator`

Calls `POST /api/v1/licence/{org}/token` against
`ATTO_CONSOLE_BASE_URL`. The response is a signed JWT (Ed25519) cached
in-process for 24 hours. If the console is unreachable, the cached
token remains valid for a configurable grace period (default 72 h).
Construct directly only when you need to override the HTTP client;
`build_default_runtime()` produces a properly configured instance from
environment variables.

```python
from atto.licence.console_validator import ConsoleLicenceValidator
from atto.licence.http_client import LicenceHttpClient

http = LicenceHttpClient(
    api_base="https://console.atto-qde.com",
    api_key="atto_live_...",
    org_id="org_...",
)
validator = ConsoleLicenceValidator(http_client=http)
status = validator.check("org_...")
```

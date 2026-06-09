# Errors

The licence and usage modules raise four exception types, all rooted at
`atto.licence.errors.LicenceError`. Catch the base class to handle any
licensing failure uniformly; catch a subclass for specific recovery.

## Hierarchy

```
LicenceError                          (base — transport / unknown state)
├── PlanUpgradeRequired               (free quota exhausted)
├── QuotaExhausted                    (balance zero / suspended)
└── InvalidApiKey                     (401 / 403)
```

## `LicenceError`

Raised on transport failures — connection timeouts, 5xx responses,
malformed payloads. The licence state is **unknown**; the SDK fails
closed and aborts the decision.

```python
from atto.licence.errors import LicenceError

try:
    engine.decide()
except LicenceError as exc:
    logger.warning("licence transport failure: %s", exc)
    # Fall back to a default action, retry, or surface to the user.
```

## `PlanUpgradeRequired`

Raised on `core_free` after the 100-decision allowance is consumed.

```python
from atto.licence.errors import PlanUpgradeRequired

try:
    engine.decide()
except PlanUpgradeRequired as exc:
    if exc.checkout_url:
        # No payment method on file — direct the user to Stripe.
        return redirect(exc.checkout_url)
    # Payment method on file — server-side upgrade is in flight;
    # retry after a short backoff.
```

`exc.checkout_url` is `None` when the org already has a payment method
on file: the upgrade happens server-side and a retry will succeed.

## `QuotaExhausted`

Raised when the account is suspended or the prepaid balance is zero
(metered orgs). There is no automatic recovery — the operator must
add a payment method or top up.

## `InvalidApiKey`

Raised on 401 (key not recognised, revoked, or expired) or 403 (the key
exists but lacks the `core` product scope). Carries `status_code` for
disambiguation.

```python
from atto.licence.errors import InvalidApiKey

try:
    engine.decide()
except InvalidApiKey as exc:
    if exc.status_code == 403:
        # Key exists but doesn't have the `core` product scope.
        # Re-issue with `products=["core"]` from the console.
        ...
    else:
        # 401 — key is wrong, revoked, or pointing at the wrong env.
        ...
```

## Usage emission errors

`ConsoleUsageEmitter.emit()` does **not** raise. Failures (network,
4xx/5xx, queue overflow) are logged at `WARNING` and the event is
dropped. Usage data is treated as advisory; the licence endpoint is the
source of truth for the live counter shown in the dashboard.

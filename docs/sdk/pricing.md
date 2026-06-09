# Pricing

The Core SDK has two plans. Numbers below match the constants in
[`coreThresholds.ts`](https://github.com/atto-qde/atto-site-console/blob/main/packages/shared/src/plans/coreThresholds.ts)
exactly.

| Plan           | Monthly fee | Included decisions | Per-decision price after allowance |
| -------------- | ----------- | ------------------ | ---------------------------------- |
| `core_free`    | $0          | 100                | Upgrade required (see below)       |
| `core_metered` | $20         | 0                  | $0.10 (10¢)                        |

All money values are integer cents on the wire and in the ledger;
display values like `$0.10` are derived for presentation.

## What counts as a decision

A **decision** is any successful return from
`AttoEngine.decide()` or `AttoEngine.decide_from()`. The SDK emits a
single `UsageEvent` with `op_type=decision` per call.

These operations are **free** and do not count toward your allowance:

- Building or mutating an `AttoEngine` (`add_operator`, `evolve`,
  `set_initial_state`, …) without calling `decide()`.
- Calibration (`AttoCalibrator.fit`) and warm-up runs — they emit
  `op_type=warmup` / `op_type=calibration` events for telemetry but
  are priced at zero.
- Failed decisions where `decide()` raises before producing a result.

## Free → Metered upgrade

`core_free` includes 100 decisions per calendar month (UTC). When the
101st decision is requested, the licence endpoint returns
`upgrade_required=true` and the SDK raises:

```python
from atto.licence.errors import PlanUpgradeRequired

try:
    engine.decide()
except PlanUpgradeRequired as exc:
    if exc.checkout_url:
        # No payment method on file — redirect the human operator to Stripe.
        print(f"Upgrade at: {exc.checkout_url}")
    else:
        # Payment method on file — server-side upgrade is in progress;
        # retry shortly.
        ...
```

Once the org is on `core_metered`, all decisions are billed at 10¢
each. There is **no overage cap** at launch — usage is open-ended.

## Period boundary

The billing period is the calendar month in UTC. Counters reset at
`00:00:00Z` on the first of each month.

## Currency and tax

Prices are USD. Sales tax / VAT is added at the Stripe checkout step
where applicable; it is not billed by the SDK.

## Plan changes

- **Upgrade** (`core_free` → `core_metered`) is automatic on breach if
  a payment method is on file, otherwise gated on a Stripe Checkout
  session.
- **Cancel / downgrade** is admin-assisted at launch — open a support
  ticket from the dashboard.

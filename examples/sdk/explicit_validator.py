"""Construct ConsoleLicenceValidator + ConsoleUsageEmitter manually.

Use this pattern when you need to override defaults — for example, to
point at a staging console, supply a custom HTTP client, or wire up a
secrets manager that resolves credentials at runtime instead of from
environment variables.

Run::

    pip install 'atto-qde[console]'
    python explicit_validator.py
"""

from __future__ import annotations

import os

from atto import AttoEngine, AttoOperator
from atto.licence.console_validator import ConsoleLicenceValidator
from atto.licence.errors import (
    InvalidApiKey,
    LicenceError,
    PlanUpgradeRequired,
    QuotaExhausted,
)
from atto.licence.http_client import LicenceHttpClient
from atto.usage.emitter import ConsoleUsageEmitter

ORG_ID = os.environ["ATTO_ORG_ID"]
API_KEY = os.environ["ATTO_API_KEY"]
API_BASE = os.environ.get("ATTO_CONSOLE_BASE_URL", "https://console.atto-qde.com")


def main() -> None:
    http = LicenceHttpClient(api_base=API_BASE, api_key=API_KEY, org_id=ORG_ID)
    validator = ConsoleLicenceValidator(http_client=http)
    emitter = ConsoleUsageEmitter(api_base=API_BASE, api_key=API_KEY, org_id=ORG_ID)

    engine = AttoEngine(
        dimension=3,
        labels=["charge", "hold", "discharge"],
        validator=validator,
        emitter=emitter,
        org_id=ORG_ID,
        scenario="battery_dispatch",
    )
    engine.add_operator(AttoOperator.phase_shift(3, [0.5, 0.0, -0.3]))

    try:
        decision = engine.decide()
    except PlanUpgradeRequired as exc:
        print(f"upgrade required; checkout: {exc.checkout_url}")
        return
    except QuotaExhausted:
        print("balance exhausted; top up the account")
        return
    except InvalidApiKey as exc:
        print(f"invalid api key (HTTP {exc.status_code})")
        return
    except LicenceError as exc:
        print(f"licence transport failure: {exc}")
        return

    print(f"{decision.label} (confidence={decision.confidence:.3f})")


if __name__ == "__main__":
    main()

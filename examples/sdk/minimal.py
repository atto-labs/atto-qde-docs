"""Minimal Core SDK example: env-driven defaults.

Run::

    pip install 'atto-qde[console]'
    export ATTO_API_KEY=atto_live_...
    export ATTO_ORG_ID=org_...
    python minimal.py

When ``ATTO_API_KEY`` and ``ATTO_ORG_ID`` are not set, the SDK runs in
offline mode (no network) and the same code still works.
"""

from __future__ import annotations

from atto import AttoEngine, AttoOperator


def main() -> None:
    engine = AttoEngine(
        dimension=3,
        labels=["charge", "hold", "discharge"],
    )
    engine.add_operator(AttoOperator.phase_shift(3, [0.5, 0.0, -0.3]))
    engine.add_operator(AttoOperator.interference(3, i=0, j=2, angle=0.4))

    decision = engine.decide()

    print(f"action={decision.action}")
    print(f"label={decision.label}")
    print(f"confidence={decision.confidence:.3f}")
    print(f"probabilities={[round(p, 3) for p in decision.probabilities]}")


if __name__ == "__main__":
    main()

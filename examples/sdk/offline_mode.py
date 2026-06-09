"""Offline-mode example: NoOp validator + emitter, no network.

Useful for OSS users, local development, CI, and air-gapped
deployments. The SDK behaves identically to hosted mode from the
caller's perspective; the validator always returns valid and the
emitter discards events (but retains them in ``emitter.events`` for
inspection).

Run::

    pip install atto-qde   # no [console] extra needed
    python offline_mode.py
"""

from __future__ import annotations

from atto import (
    AttoEngine,
    AttoOperator,
    NoOpLicenceValidator,
    NoOpUsageEmitter,
)


def main() -> None:
    validator = NoOpLicenceValidator()
    emitter = NoOpUsageEmitter()

    engine = AttoEngine(
        dimension=3,
        labels=["charge", "hold", "discharge"],
        validator=validator,
        emitter=emitter,
    )
    engine.add_operator(AttoOperator.phase_shift(3, [0.5, 0.0, -0.3]))

    decision = engine.decide()

    print(f"decision: {decision.label} ({decision.confidence:.3f})")
    print(f"events captured offline: {len(emitter.events)}")
    print(f"first event op_type: {emitter.events[0].op_type}")


if __name__ == "__main__":
    main()

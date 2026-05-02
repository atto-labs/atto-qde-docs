# Architecture

## Structure

```
atto-qde-docs/
├── examples/              # Jupyter notebooks (numbered 1-10)
├── example_datasets/      # CSV datasets for examples
│   └── live_data_examples/
├── assets/                # Images and diagrams
└── _misc/                 # Internal notes and scripts
```

## Examples (Notebooks)

1. Creative Problem Solving
2. Strategic Decision Under Uncertainty
3. Hiring Outcome Spectrum
4. Product Strategy Decision
5. Investment Allocation
6. Game Theory Decisioning
7. Anatomy of a QDE Decision
8. Interference Visualisation
9. Building a Custom Scenario
10. Training and Persistence

## Key Points

- Public repository (MIT-like) — no proprietary code
- Notebooks demonstrate the `atto` public API only
- Each notebook has a corresponding CSV dataset in `example_datasets/`
- Live data examples are in a separate subdirectory
- All notebooks import from `atto` (the public package), never from `atto_adaptive`

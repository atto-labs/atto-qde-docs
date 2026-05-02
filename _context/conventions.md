# Conventions

## Notebooks

- Numbered sequentially (1-10)
- Each notebook is self-contained (can run independently)
- Use the public `atto` API only — never import from `atto_adaptive`
- Include explanatory markdown cells between code cells
- Datasets loaded from `../example_datasets/` relative paths

## Datasets

- Format: CSV
- One dataset per example scenario
- Include a README.md explaining the datasets
- Columns should be self-descriptive

## Documentation

- README.md at root describes the docs project
- Each examples/ directory has its own README
- Written for external users (clear, accessible language)

## Boundaries

- PUBLIC repository — no proprietary or private logic
- References to atto-qde-adaptive must be generic/high-level only
- No API keys, credentials, or internal URLs

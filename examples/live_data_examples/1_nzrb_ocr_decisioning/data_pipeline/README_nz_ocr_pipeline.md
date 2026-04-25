# NZ OCR Decision Dataset Pipeline

Build real-data CSV files for a QDE scenario where the decision-maker is the Reserve Bank of New Zealand considering whether to adjust the Official Cash Rate (OCR).

## Outputs

The script writes two CSV files:

- `nz_ocr_historical_decision_dataset.csv`
- `nz_ocr_today_decision_context.csv`

No MongoDB is used.

## Data sources

- RBNZ B2 wholesale interest rates Excel file
- RBNZ historical OCR decisions page
- FRED CSV endpoints for NZ CPI, unemployment, GDP growth, and 10-year yield

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_nz_ocr_pipeline.txt
```

## Run

```bash
python build_nz_ocr_decision_dataset.py --out datasets --today 2026-04-24
```

Use today's actual date when running locally:

```bash
python build_nz_ocr_decision_dataset.py --out datasets
```

## Notes

This is a real-data pipeline, but strict historical publication-vintage correctness would require ALFRED or source-specific release calendars. The current implementation aligns each row to the latest observed public data available on or before the decision date.

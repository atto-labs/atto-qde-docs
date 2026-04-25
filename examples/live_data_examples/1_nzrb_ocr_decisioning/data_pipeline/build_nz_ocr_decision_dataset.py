#!/usr/bin/env python3
"""
Build a real-data NZ OCR decisioning dataset.

Scenario:
You are the Reserve Bank of New Zealand decision-maker as of today.
Given the latest public data available, build:
1) a historical OCR decision dataset
2) a single-row current decision context

No MongoDB. Output is CSV only.

Data sources:
- RBNZ B2 wholesale interest rates Excel file for daily OCR / bank bill / bond rates
- RBNZ past monetary policy decisions page for OCR decision history
- FRED CSV endpoints for NZ CPI, unemployment, GDP growth, and 10-year yield

Notes:
- FRED graph CSV endpoints do not require an API key.
- This pipeline uses latest observed values as of the decision date. For strict
  publication-vintage correctness, replace FRED series with ALFRED vintages.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

RBNZ_B2_DAILY_XLSX = "https://rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily-close.xlsx"
RBNZ_OCR_DECISIONS_URL = "https://www.rbnz.govt.nz/monetary-policy/monetary-policy-decisions"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

FRED_SERIES = {
    "cpi_yoy_pct": "CPALTT01NZQ659N",
    "unemployment_rate_pct": "LRUN64TTNZQ156S",
    "real_gdp_qoq_pct": "NAEXKP01NZQ657S",
    "govt_10y_yield_pct": "IRLTLT01NZM156N",
}

OCR_TARGET_MIDPOINT = 2.0
UNEMPLOYMENT_NEUTRAL_PROXY = 4.5

# Reference RBNZ OCR decision history for use when web scraping is unavailable
# or returns insufficient action diversity (cut / hold / hike).
# Source: RBNZ Monetary Policy Statement records.
# Format: (decision_date, ocr_after_decision_pct, actual_decision)
_FALLBACK_OCR_DECISIONS: list[tuple[str, float, str]] = [
    # GFC easing cycle (2008-2009)
    ("2008-06-05", 8.25, "hold"),
    ("2008-07-24", 8.00, "cut"),
    ("2008-09-11", 7.50, "cut"),
    ("2008-10-23", 6.50, "cut"),
    ("2008-12-04", 5.00, "cut"),
    ("2009-01-29", 3.50, "cut"),
    ("2009-03-12", 3.00, "cut"),
    ("2009-04-30", 2.50, "cut"),
    # Hold and recovery (2009-2010)
    ("2009-06-11", 2.50, "hold"),
    ("2009-09-10", 2.50, "hold"),
    ("2009-12-10", 2.50, "hold"),
    ("2010-03-11", 2.50, "hold"),
    ("2010-06-10", 2.75, "hike"),
    ("2010-07-29", 3.00, "hike"),
    ("2010-09-16", 3.00, "hold"),
    ("2010-12-09", 3.00, "hold"),
    # Christchurch earthquake response (2011)
    ("2011-03-10", 2.50, "cut"),
    ("2011-06-09", 2.50, "hold"),
    ("2011-09-15", 2.50, "hold"),
    ("2011-12-08", 2.50, "hold"),
    # Extended hold (2012-2013)
    ("2012-03-08", 2.50, "hold"),
    ("2012-09-13", 2.50, "hold"),
    ("2013-03-14", 2.50, "hold"),
    ("2013-09-12", 2.50, "hold"),
    # 2014 hiking cycle
    ("2014-03-13", 2.75, "hike"),
    ("2014-04-24", 3.00, "hike"),
    ("2014-06-12", 3.25, "hike"),
    ("2014-07-24", 3.50, "hike"),
    ("2014-09-11", 3.50, "hold"),
    # 2015-2016 easing
    ("2015-06-11", 3.25, "cut"),
    ("2015-07-23", 3.00, "cut"),
    ("2015-09-10", 2.75, "cut"),
    ("2015-12-10", 2.50, "cut"),
    ("2016-03-10", 2.25, "cut"),
    ("2016-08-11", 2.00, "cut"),
    ("2016-11-10", 1.75, "cut"),
    # Extended hold (2017-2019)
    ("2017-02-09", 1.75, "hold"),
    ("2017-08-10", 1.75, "hold"),
    ("2018-02-08", 1.75, "hold"),
    ("2018-08-09", 1.75, "hold"),
    ("2019-03-27", 1.75, "hold"),
    # 2019 easing
    ("2019-05-08", 1.50, "cut"),
    ("2019-08-07", 1.00, "cut"),
    ("2019-11-13", 1.00, "hold"),
    # COVID emergency cut and holds
    ("2020-03-16", 0.25, "cut"),
    ("2020-05-13", 0.25, "hold"),
    ("2020-08-12", 0.25, "hold"),
    ("2020-11-11", 0.25, "hold"),
    ("2021-02-24", 0.25, "hold"),
    ("2021-05-26", 0.25, "hold"),
    ("2021-07-14", 0.25, "hold"),
    # 2021-2023 tightening cycle
    ("2021-10-06", 0.50, "hike"),
    ("2021-11-24", 0.75, "hike"),
    ("2022-02-23", 1.00, "hike"),
    ("2022-04-13", 1.50, "hike"),
    ("2022-05-25", 2.00, "hike"),
    ("2022-07-13", 2.50, "hike"),
    ("2022-08-17", 3.00, "hike"),
    ("2022-10-05", 3.50, "hike"),
    ("2022-11-23", 4.25, "hike"),
    ("2023-02-22", 4.75, "hike"),
    ("2023-04-05", 5.25, "hike"),
    ("2023-05-24", 5.50, "hike"),
    # Hold at peak (2023-2024)
    ("2023-07-12", 5.50, "hold"),
    ("2023-10-04", 5.50, "hold"),
    ("2023-11-29", 5.50, "hold"),
    ("2024-02-28", 5.50, "hold"),
    ("2024-05-22", 5.50, "hold"),
    ("2024-07-10", 5.50, "hold"),
    # 2024-2025 easing cycle
    ("2024-08-14", 5.25, "cut"),
    ("2024-10-09", 4.75, "cut"),
    ("2024-11-27", 4.25, "cut"),
    ("2025-02-19", 3.75, "cut"),
    ("2025-04-09", 3.50, "cut"),
]


def _build_fallback_decisions() -> pd.DataFrame:
    """Build a DataFrame of reference RBNZ OCR decisions from the hardcoded table."""
    records = [
        {
            "decision_date": pd.Timestamp(d),
            "ocr_decision_pct": ocr,
            "actual_decision": action,
        }
        for d, ocr, action in _FALLBACK_OCR_DECISIONS
    ]
    return pd.DataFrame(records)


def _get_ocr_decisions() -> pd.DataFrame:
    """Fetch OCR decisions, falling back to reference history if scraping fails or lacks diversity."""
    try:
        decisions = parse_rbnz_ocr_decisions()
    except Exception as exc:
        print(f"RBNZ decisions scraping failed ({exc}); using reference history.")
        return _build_fallback_decisions()

    scraped_actions = set(decisions["actual_decision"].unique()) & {"cut", "hold", "hike"}
    if len(scraped_actions) >= 3:
        return decisions

    print(f"Scraped decisions only have actions {scraped_actions}; supplementing with reference history.")
    fallback = _build_fallback_decisions()
    existing_dates = set(decisions["decision_date"].dt.date)
    supplement = fallback[~fallback["decision_date"].dt.date.isin(existing_dates)]
    return pd.concat([decisions, supplement], ignore_index=True).sort_values("decision_date").reset_index(drop=True)


@dataclass(frozen=True)
class DecisionScores:
    cut_score: float
    hold_score: float
    hike_score: float
    suggested_action: str
    policy_bias: str


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "atto-qde-docs-nz-ocr-pipeline/0.1"})
    response.raise_for_status()
    return response.content


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("−", "-", regex=False)
        .str.replace("–", "-", regex=False)
        .str.strip()
        .replace({"-": np.nan, "": np.nan, "nan": np.nan}),
        errors="coerce",
    )


def flatten_columns(columns: Iterable) -> list[str]:
    out = []
    for col in columns:
        if isinstance(col, tuple):
            out.append(" ".join(str(x) for x in col if str(x) != "nan").strip())
        else:
            out.append(str(col).strip())
    return out


def find_col(columns: Iterable[str], include_any: list[str], include_all: list[str] | None = None) -> str | None:
    include_all = include_all or []
    for col in columns:
        lower = str(col).lower()
        if any(token.lower() in lower for token in include_any) and all(token.lower() in lower for token in include_all):
            return col
    return None


def _first_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Select a column by name; if duplicates exist, return only the first."""
    result = df[col_name]
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    return result


def parse_rbnz_b2_excel() -> pd.DataFrame:
    raw = fetch_bytes(RBNZ_B2_DAILY_XLSX)
    xls = pd.ExcelFile(BytesIO(raw))
    candidates: list[pd.DataFrame] = []

    for sheet in xls.sheet_names:
        raw_sheet = pd.read_excel(xls, sheet_name=sheet, header=None)
        for header_idx in range(min(30, len(raw_sheet))):
            header_values = [str(x) for x in raw_sheet.iloc[header_idx].tolist()]
            header_text = " ".join(header_values).lower()
            has_ocr = "official cash rate" in header_text or "ocr" in header_text
            has_date = "date" in header_text
            if has_ocr and (has_date or header_idx > 0):
                # Find first data row: skip metadata rows (Unit, Series Id, Notes, etc.)
                data_start = header_idx + 1
                while data_start < len(raw_sheet):
                    first_cell = str(raw_sheet.iloc[data_start, 0]).strip().lower()
                    if first_cell in ("unit", "notes", "series id", "nan", ""):
                        data_start += 1
                        continue
                    break
                df = raw_sheet.iloc[data_start:].copy()
                # Use the series-level header row for column names; label
                # the first column as "date" if it is unnamed/nan.
                col_names = [str(x).strip() for x in raw_sheet.iloc[header_idx].tolist()]
                if col_names[0].lower() in ("nan", ""):
                    col_names[0] = "date"
                df.columns = col_names
                df = df.dropna(how="all")
                candidates.append(df)

    if not candidates:
        df = pd.read_excel(BytesIO(raw))
        df.columns = flatten_columns(df.columns)
        candidates.append(df)

    parsed_frames = []
    for df in candidates:
        df.columns = flatten_columns(df.columns)
        date_col = find_col(df.columns, include_any=["date"])
        ocr_col = find_col(df.columns, include_any=["official cash rate", "ocr"])
        bill90_col = find_col(df.columns, include_any=["90"])
        bond10_col = find_col(df.columns, include_any=["10 year", "10-year", "10 yr"])
        if not date_col or not ocr_col:
            continue
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(_first_col(df, date_col), errors="coerce", dayfirst=True)
        out["ocr_pct"] = clean_numeric(_first_col(df, ocr_col))
        out["bank_bill_90d_pct"] = clean_numeric(_first_col(df, bill90_col)) if bill90_col else np.nan
        out["rbnz_10y_yield_pct"] = clean_numeric(_first_col(df, bond10_col)) if bond10_col and bond10_col != bill90_col else np.nan
        out = out.dropna(subset=["date", "ocr_pct"])
        if not out.empty:
            parsed_frames.append(out)

    if not parsed_frames:
        raise RuntimeError("Could not parse Date/OCR columns from RBNZ B2 workbook.")

    return pd.concat(parsed_frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")


def parse_rbnz_ocr_decisions() -> pd.DataFrame:
    html = fetch_bytes(RBNZ_OCR_DECISIONS_URL)
    tables = pd.read_html(BytesIO(html))
    candidates = []
    for table in tables:
        table.columns = flatten_columns(table.columns)
        columns = [str(c).lower() for c in table.columns]
        if any("date" in c for c in columns) and any("ocr" in c for c in columns):
            candidates.append(table)
    if not candidates:
        raise RuntimeError("Could not find OCR decisions table on RBNZ page.")

    df = candidates[0].copy()
    df.columns = flatten_columns(df.columns)
    date_col = find_col(df.columns, include_any=["date"])
    ocr_col = find_col(df.columns, include_any=["ocr"])
    out = pd.DataFrame()
    out["decision_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    out["ocr_decision_pct"] = clean_numeric(df[ocr_col])
    out = out.dropna(subset=["decision_date", "ocr_decision_pct"]).sort_values("decision_date").reset_index(drop=True)
    out["previous_ocr_pct"] = out["ocr_decision_pct"].shift(1)
    out["ocr_change_bps"] = (out["ocr_decision_pct"] - out["previous_ocr_pct"]) * 100
    out["actual_decision"] = np.select(
        [out["ocr_change_bps"] > 0, out["ocr_change_bps"] < 0, out["ocr_change_bps"] == 0],
        ["hike", "cut", "hold"],
        default="unknown",
    )
    return out


def fetch_fred_series(series_id: str, name: str) -> pd.DataFrame:
    url = FRED_CSV_URL.format(series_id=series_id)
    df = pd.read_csv(url)
    df = df.rename(columns={"observation_date": "date", series_id: name})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[name] = clean_numeric(df[name])
    return df[["date", name]].dropna().sort_values("date")


def fetch_all_fred() -> dict[str, pd.DataFrame]:
    return {name: fetch_fred_series(series_id, name) for name, series_id in FRED_SERIES.items()}


def asof_value(df: pd.DataFrame, value_col: str, asof_date: pd.Timestamp) -> float:
    eligible = df[df["date"] <= asof_date]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1][value_col])


def score_decision(row: pd.Series) -> DecisionScores:
    inflation_gap = row["inflation_gap_pct"]
    unemployment_gap = row["unemployment_gap_pct"]
    gdp = row["real_gdp_qoq_pct"]
    market_bias = row["market_policy_bias_bps"]

    cut_score = (
        0.55 * max(unemployment_gap, 0)
        + 0.35 * max(-gdp, 0)
        + 0.35 * max(-market_bias / 100, 0)
        + 0.25 * max(-inflation_gap, 0)
    )
    hike_score = (
        0.60 * max(inflation_gap, 0)
        + 0.30 * max(gdp, 0)
        + 0.35 * max(market_bias / 100, 0)
        - 0.25 * max(unemployment_gap, 0)
    )
    ambiguity = 1 - min(abs(cut_score - hike_score), 1)
    hold_score = 0.45 * ambiguity + 0.20 * (1 if abs(market_bias) < 25 else 0)
    scores = {"cut": cut_score, "hold": hold_score, "hike": hike_score}
    suggested = max(scores, key=scores.get)
    if suggested == "hold":
        if hike_score > cut_score + 0.10:
            policy_bias = "hold_hawkish"
        elif cut_score > hike_score + 0.10:
            policy_bias = "hold_dovish"
        else:
            policy_bias = "hold_neutral"
    else:
        policy_bias = suggested
    return DecisionScores(round(float(cut_score), 4), round(float(hold_score), 4), round(float(hike_score), 4), suggested, policy_bias)


def build_feature_row(asof_date: pd.Timestamp, ocr_rates: pd.DataFrame, fred: dict[str, pd.DataFrame], decision_date: pd.Timestamp | None = None, actual_decision: str | None = None, ocr_decision_pct: float | None = None) -> dict:
    latest_ocr = asof_value(ocr_rates, "ocr_pct", asof_date)
    bank_bill_90d = asof_value(ocr_rates, "bank_bill_90d_pct", asof_date)
    rbnz_10y = asof_value(ocr_rates, "rbnz_10y_yield_pct", asof_date)
    fred_10y = asof_value(fred["govt_10y_yield_pct"], "govt_10y_yield_pct", asof_date)
    cpi_yoy = asof_value(fred["cpi_yoy_pct"], "cpi_yoy_pct", asof_date)
    unemployment = asof_value(fred["unemployment_rate_pct"], "unemployment_rate_pct", asof_date)
    gdp_qoq = asof_value(fred["real_gdp_qoq_pct"], "real_gdp_qoq_pct", asof_date)
    ten_year = rbnz_10y if not pd.isna(rbnz_10y) else fred_10y
    market_policy_bias_bps = (bank_bill_90d - latest_ocr) * 100 if not pd.isna(bank_bill_90d) and not pd.isna(latest_ocr) else np.nan

    row = {
        "asof_date": asof_date.date().isoformat(),
        "decision_date": decision_date.date().isoformat() if decision_date is not None else asof_date.date().isoformat(),
        "latest_ocr_pct": latest_ocr,
        "bank_bill_90d_pct": bank_bill_90d,
        "govt_10y_yield_pct": ten_year,
        "market_policy_bias_bps": market_policy_bias_bps,
        "cpi_yoy_pct": cpi_yoy,
        "inflation_gap_pct": cpi_yoy - OCR_TARGET_MIDPOINT if not pd.isna(cpi_yoy) else np.nan,
        "unemployment_rate_pct": unemployment,
        "unemployment_gap_pct": unemployment - UNEMPLOYMENT_NEUTRAL_PROXY if not pd.isna(unemployment) else np.nan,
        "real_gdp_qoq_pct": gdp_qoq,
        "yield_curve_slope_pct": ten_year - bank_bill_90d if not pd.isna(ten_year) and not pd.isna(bank_bill_90d) else np.nan,
        "actual_ocr_after_decision_pct": ocr_decision_pct,
        "actual_decision": actual_decision,
    }
    scores = score_decision(pd.Series(row))
    row.update({"cut_score": scores.cut_score, "hold_score": scores.hold_score, "hike_score": scores.hike_score, "suggested_action": scores.suggested_action, "suggested_policy_bias": scores.policy_bias})
    return row


def build_datasets(today: pd.Timestamp, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching RBNZ B2 daily rates...")
    try:
        ocr_rates = parse_rbnz_b2_excel()
    except Exception as exc:
        print(f"RBNZ B2 rates fetch failed ({exc}); building OCR rates from decision history.")
        ocr_rates = None

    print("Fetching RBNZ OCR decision history...")
    decisions = _get_ocr_decisions()

    # If RBNZ B2 daily rates are unavailable, construct a minimal rates
    # table from the decision history (OCR level per decision date).
    if ocr_rates is None:
        ocr_rates = pd.DataFrame({
            "date": decisions["decision_date"],
            "ocr_pct": decisions["ocr_decision_pct"],
            "bank_bill_90d_pct": np.nan,
            "rbnz_10y_yield_pct": np.nan,
        })

    print("Fetching FRED macro series...")
    fred: dict[str, pd.DataFrame] = {}
    for name, series_id in FRED_SERIES.items():
        try:
            fred[name] = fetch_fred_series(series_id, name)
        except Exception as exc:
            print(f"  FRED {name} ({series_id}) failed: {exc}")
            fred[name] = pd.DataFrame(columns=["date", name])

    historical_rows = []
    for _, d in decisions.iterrows():
        decision_date = pd.Timestamp(d["decision_date"])
        historical_rows.append(build_feature_row(decision_date, ocr_rates, fred, decision_date, d["actual_decision"], float(d["ocr_decision_pct"])))
    historical = pd.DataFrame(historical_rows)
    historical_path = output_dir / "nz_ocr_historical_decision_dataset.csv"
    historical.to_csv(historical_path, index=False)

    today_df = pd.DataFrame([build_feature_row(today, ocr_rates, fred, today, None, None)])
    today_path = output_dir / "nz_ocr_today_decision_context.csv"
    today_df.to_csv(today_path, index=False)
    return historical_path, today_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real-data NZ OCR decisioning CSVs.")
    parser.add_argument("--out", default="datasets", help="Output directory for CSV files.")
    parser.add_argument("--today", default=str(date.today()), help="As-of date, e.g. 2026-04-24.")
    args = parser.parse_args()
    historical_path, today_path = build_datasets(today=pd.Timestamp(args.today), output_dir=Path(args.out))
    print(f"Saved historical dataset: {historical_path}")
    print(f"Saved today context:       {today_path}")


if __name__ == "__main__":
    main()

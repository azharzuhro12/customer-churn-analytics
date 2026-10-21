"""Clean the raw Telco churn CSV and load it into SQLite.

Run from the project root:  python -m src.prepare
"""

import sqlite3

import pandas as pd

from src.common import DB_PATH, RAW_CSV, ensure_dirs


def load_and_clean() -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(RAW_CSV)

    # --- integrity checks on the raw file -------------------------------------
    assert len(df) == 7043, f"unexpected row count: {len(df)}"
    assert not df["customerID"].duplicated().any(), "duplicate customerID found"
    assert set(df["Churn"].unique()) == {"Yes", "No"}, "unexpected Churn values"

    # --- TotalCharges: blanks mean "signed up, not billed yet" ---------------
    # The column is read as text because 11 rows contain a single space.
    total_charges = df["TotalCharges"].str.strip()
    blank = total_charges == ""
    assert blank.sum() == 11, f"expected 11 blank TotalCharges, got {blank.sum()}"

    df["TotalCharges"] = pd.to_numeric(total_charges, errors="coerce")
    # Prove the blanks are the same rows we just saw and that all have tenure 0.
    assert df["TotalCharges"].isna().sum() == 11
    assert df.loc[blank, "tenure"].eq(0).all(), "a blank TotalCharges has tenure > 0"
    assert df["TotalCharges"].isna().sum() == blank.sum()

    # No money was billed yet, so zero is the correct value (not the column mean).
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    assert df["TotalCharges"].notna().all()

    # --- type and encoding cleanup -------------------------------------------
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    assert df["SeniorCitizen"].notna().all()

    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)
    assert df["churn_flag"].isin([0, 1]).all()
    assert df["churn_flag"].sum() == (df["Churn"] == "Yes").sum()

    return df, int(blank.sum())


def write_sqlite(df: pd.DataFrame) -> None:
    # Rebuild the file so a rerun never leaves stale rows or schema behind.
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("customers", conn, index=False)


def main() -> None:
    ensure_dirs()
    df, n_fixed = load_and_clean()
    write_sqlite(df)

    churn_rate = 100 * df["churn_flag"].mean()
    print(f"rows: {len(df)}  columns: {len(df.columns)}")
    print(f"churned: {df['churn_flag'].sum()}  churn rate: {churn_rate:.2f}%")
    print(f"blank TotalCharges set to 0: {n_fixed} (all with tenure 0)")
    print(
        "monthly revenue: "
        f"{df['MonthlyCharges'].sum():,.2f}  "
        f"churned share: {100 * df.loc[df.churn_flag == 1, 'MonthlyCharges'].sum() / df['MonthlyCharges'].sum():.2f}%"
    )
    print(f"database: {DB_PATH.relative_to(DB_PATH.parents[1])}")


if __name__ == "__main__":
    main()

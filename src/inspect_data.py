import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

TRANSACTIONS_FILE = RAW_DIR / "Transactions.csv"
PRODUCTS_FILE = RAW_DIR / "Products.csv"


def inspect_dataframe(name, df):
    print("\n" + "=" * 80)
    print(f"{name.upper()} DATASET")
    print("=" * 80)

    print(f"\nRows       : {df.shape[0]:,}")
    print(f"Columns    : {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("  No missing values")
    else:
        print(missing)

    print("\nDuplicate rows:")
    print(f"  {df.duplicated().sum():,}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())


def main():
    print("=" * 80)
    print("RESTAURANT SALES INTELLIGENCE")
    print("DATASET INSPECTION")
    print("=" * 80)

    if not TRANSACTIONS_FILE.exists():
        print("\nERROR: Transactions file not found:")
        print(TRANSACTIONS_FILE)
        return

    if not PRODUCTS_FILE.exists():
        print("\nERROR: Products file not found:")
        print(PRODUCTS_FILE)
        return

    transactions = pd.read_csv(TRANSACTIONS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    inspect_dataframe("Transactions", transactions)
    inspect_dataframe("Products", products)

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
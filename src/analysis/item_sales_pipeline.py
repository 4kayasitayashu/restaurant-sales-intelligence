import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRANSACTIONS_FILE = PROCESSED_DIR / "sales_processed.csv"
PRODUCTS_FILE = RAW_DIR / "Products.csv"

OUTPUT_FILE = PROCESSED_DIR / "item_sales_processed.csv"
UNMATCHED_FILE = PROCESSED_DIR / "unmatched_items.csv"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    value = re.sub(r"\s*\(\s*", "(", value)
    value = re.sub(r"\s*\)\s*", ")", value)

    return value


# ============================================================
# PRODUCT LABEL
# ============================================================

def build_product_label(product_name, variant):

    product_name = str(product_name).strip()
    variant = str(variant).strip()

    if variant in ["", "-", "nan", "None"]:
        return product_name

    return f"{product_name} ({variant})"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n[1/7] Loading datasets...")

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    products = pd.read_csv(
        PRODUCTS_FILE
    )

    print(f"Transactions : {len(transactions):,}")
    print(f"Products     : {len(products):,}")

    return transactions, products


# ============================================================
# PREPARE PRODUCT CATALOG
# ============================================================

def prepare_product_catalog(products):

    print("\n[2/7] Preparing product catalog...")

    products = products.copy()

    products.columns = (
        products.columns
        .str.strip()
    )

    for column in [
        "ProductName",
        "Variant",
        "Category"
    ]:
        products[column] = (
            products[column]
            .astype(str)
            .str.strip()
        )

    products["Price"] = pd.to_numeric(
        products["Price"],
        errors="coerce"
    )

    products["ProductLabel"] = products.apply(
        lambda row: build_product_label(
            row["ProductName"],
            row["Variant"]
        ),
        axis=1
    )

    products["NormalizedLabel"] = (
        products["ProductLabel"]
        .apply(normalize_text)
    )

    products["NormalizedProductName"] = (
        products["ProductName"]
        .apply(normalize_text)
    )

    return products


# ============================================================
# BUILD FAST LOOKUP DICTIONARIES
# ============================================================

def build_product_lookups(products):

    print("\n[3/7] Building product lookup tables...")

    # --------------------------------------------------------
    # Full label lookup
    # --------------------------------------------------------

    label_lookup = {}

    for _, row in products.iterrows():

        key = row["NormalizedLabel"]

        if not key:
            continue

        # Only store unique labels.
        # Duplicate labels are deliberately not guessed.
        if key not in label_lookup:

            label_lookup[key] = row.to_dict()

        else:

            label_lookup[key] = None

    # --------------------------------------------------------
    # Product name lookup
    # --------------------------------------------------------

    name_lookup = {}

    for _, row in products.iterrows():

        key = row["NormalizedProductName"]

        if not key:
            continue

        if key not in name_lookup:

            name_lookup[key] = row.to_dict()

        else:

            name_lookup[key] = None

    valid_labels = sum(
        value is not None
        for value in label_lookup.values()
    )

    valid_names = sum(
        value is not None
        for value in name_lookup.values()
    )

    print(
        f"Unique product labels : {len(label_lookup):,}"
    )

    print(
        f"Usable label matches   : {valid_labels:,}"
    )

    print(
        f"Unique product names   : {len(name_lookup):,}"
    )

    print(
        f"Usable name matches    : {valid_names:,}"
    )

    return label_lookup, name_lookup


# ============================================================
# MATCH ITEM
# ============================================================

def match_item(
    item_text,
    label_lookup,
    name_lookup
):

    normalized_item = normalize_text(
        item_text
    )

    if not normalized_item:
        return None

    # First: exact ProductName + Variant
    if normalized_item in label_lookup:

        product = label_lookup[
            normalized_item
        ]

        if product is not None:
            return product

    # Second: exact ProductName
    if normalized_item in name_lookup:

        product = name_lookup[
            normalized_item
        ]

        if product is not None:
            return product

    return None


# ============================================================
# PARSE ITEMS
# ============================================================

def parse_transaction_items(
    transactions,
    label_lookup,
    name_lookup
):

    print(
        "\n[4/7] Parsing transaction item strings..."
    )

    item_rows = []
    unmatched_rows = []
    count_mismatch_rows = []

    for transaction in transactions.itertuples(
        index=False
    ):

        items_text = str(
            transaction.Items
        ).strip()

        if not items_text:
            continue

        # Current dataset convention:
        # comma-separated products inside Items.
        item_names = [
            item.strip()
            for item in items_text.split(",")
            if item.strip()
        ]

        # ----------------------------------------------------
        # Validate parsed item count against TotalItem
        # ----------------------------------------------------

        parsed_count = len(item_names)

        if parsed_count != int(
            transaction.TotalItem
        ):

            count_mismatch_rows.append(
                {
                    "ReceiptNumber":
                        transaction.ReceiptNumber,

                    "TransactionItems":
                        items_text,

                    "TotalItem":
                        transaction.TotalItem,

                    "ParsedItemCount":
                        parsed_count
                }
            )

        # ----------------------------------------------------
        # Match each item
        # ----------------------------------------------------

        for item_position, item_name in enumerate(
            item_names,
            start=1
        ):

            matched_product = match_item(
                item_name,
                label_lookup,
                name_lookup
            )

            if matched_product is None:

                unmatched_rows.append(
                    {
                        "ReceiptNumber":
                            transaction.ReceiptNumber,

                        "Outlet":
                            transaction.Outlet,

                        "TransactionItem":
                            item_name,

                        "OriginalItemsField":
                            items_text
                    }
                )

                continue

            item_rows.append(
                {
                    "Outlet":
                        transaction.Outlet,

                    "Date":
                        transaction.Date,

                    "Time":
                        transaction.Time,

                    "DateTime":
                        transaction.DateTime,

                    "ReceiptNumber":
                        transaction.ReceiptNumber,

                    "PaymentMethod":
                        transaction.PaymentMethod,

                    "UseLoyaltyCard":
                        transaction.UseLoyaltyCard,

                    "Year":
                        transaction.Year,

                    "Month":
                        transaction.Month,

                    "MonthName":
                        transaction.MonthName,

                    "DayOfWeek":
                        transaction.DayOfWeek,

                    "DayName":
                        transaction.DayName,

                    "IsWeekend":
                        transaction.IsWeekend,

                    "Hour":
                        transaction.Hour,

                    "TimePeriod":
                        transaction.TimePeriod,

                    "ReceiptTotalAmount":
                        transaction.TotalAmount,

                    "ReceiptNetSales":
                        transaction.NetSales,

                    "ReceiptTotalItem":
                        transaction.TotalItem,

                    "ItemPosition":
                        item_position,

                    "TransactionItem":
                        item_name,

                    "ProductId":
                        int(
                            matched_product["ProductId"]
                        ),

                    "ProductName":
                        matched_product[
                            "ProductName"
                        ],

                    "Variant":
                        matched_product[
                            "Variant"
                        ],

                    "Category":
                        matched_product[
                            "Category"
                        ],

                    "Price":
                        matched_product[
                            "Price"
                        ],

                    "MatchStatus":
                        "MATCHED"
                }
            )

    item_df = pd.DataFrame(
        item_rows
    )

    unmatched_df = pd.DataFrame(
        unmatched_rows
    )

    mismatch_df = pd.DataFrame(
        count_mismatch_rows
    )

    print(
        f"\nParsed item rows : "
        f"{len(item_df):,}"
    )

    print(
        f"Unmatched items  : "
        f"{len(unmatched_df):,}"
    )

    print(
        f"Count mismatches : "
        f"{len(mismatch_df):,}"
    )

    return (
        item_df,
        unmatched_df,
        mismatch_df
    )


# ============================================================
# ITEM METRICS
# ============================================================

def calculate_item_metrics(item_df):

    print(
        "\n[5/7] Calculating item-level metrics..."
    )

    item_df = item_df.copy()

    item_df["ItemQuantity"] = 1

    item_df["Price"] = pd.to_numeric(
        item_df["Price"],
        errors="coerce"
    )

    # Catalog value of the individual item
    item_df["CatalogItemValue"] = (
        item_df["Price"]
    )

    # Total catalog price represented by
    # all matched items in the receipt.
    receipt_price_total = (
        item_df
        .groupby(
            "ReceiptNumber"
        )["Price"]
        .transform("sum")
    )

    item_df[
        "ReceiptCatalogPriceTotal"
    ] = receipt_price_total

    # --------------------------------------------------------
    # Price share
    # --------------------------------------------------------

    item_df["PriceShare"] = np.where(
        item_df[
            "ReceiptCatalogPriceTotal"
        ] > 0,

        item_df["Price"]
        / item_df[
            "ReceiptCatalogPriceTotal"
        ],

        0
    )

    # --------------------------------------------------------
    # Revenue allocation
    # --------------------------------------------------------

    item_df["AllocatedRevenue"] = (
        item_df["PriceShare"]
        * item_df["ReceiptTotalAmount"]
    )

    item_df["AllocatedNetRevenue"] = (
        item_df["PriceShare"]
        * item_df["ReceiptNetSales"]
    )

    return item_df


# ============================================================
# RECONCILIATION
# ============================================================

def reconciliation_report(
    item_df,
    transactions,
    unmatched_df,
    mismatch_df
):

    print(
        "\n[6/7] Running reconciliation..."
    )

    total_receipts = (
        transactions[
            "ReceiptNumber"
        ].nunique()
    )

    matched_receipts = (
        item_df[
            "ReceiptNumber"
        ].nunique()
    )

    total_transaction_items = (
        transactions["TotalItem"]
        .sum()
    )

    parsed_items = len(
        item_df
    )

    transaction_revenue = (
        transactions["TotalAmount"]
        .sum()
    )

    allocated_revenue = (
        item_df["AllocatedRevenue"]
        .sum()
    )

    print("\n" + "-" * 80)

    print(
        f"Total receipts          : "
        f"{total_receipts:,}"
    )

    print(
        f"Receipts with matches   : "
        f"{matched_receipts:,}"
    )

    print(
        f"Transaction TotalItem   : "
        f"{total_transaction_items:,}"
    )

    print(
        f"Parsed matched items    : "
        f"{parsed_items:,}"
    )

    print(
        f"Unmatched item rows     : "
        f"{len(unmatched_df):,}"
    )

    print(
        f"Item-count mismatches   : "
        f"{len(mismatch_df):,}"
    )

    print(
        f"Transaction revenue     : "
        f"{transaction_revenue:,.2f}"
    )

    print(
        f"Allocated item revenue  : "
        f"{allocated_revenue:,.2f}"
    )

    if transaction_revenue > 0:

        coverage = (
            allocated_revenue
            / transaction_revenue
            * 100
        )

        print(
            f"Revenue coverage       : "
            f"{coverage:.2f}%"
        )

    print("-" * 80)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    item_df,
    unmatched_df,
    mismatch_df
):

    print(
        "\n[7/7] Saving output datasets..."
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    item_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    unmatched_df.to_csv(
        UNMATCHED_FILE,
        index=False
    )

    mismatch_file = (
        PROCESSED_DIR
        / "item_count_mismatches.csv"
    )

    mismatch_df.to_csv(
        mismatch_file,
        index=False
    )

    print(
        "\nItem-level dataset:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Rows: {len(item_df):,}"
    )

    print(
        "\nUnmatched items:"
    )

    print(
        UNMATCHED_FILE
    )

    print(
        f"Rows: {len(unmatched_df):,}"
    )

    print(
        "\nItem-count validation:"
    )

    print(
        mismatch_file
    )

    print(
        f"Rows: {len(mismatch_df):,}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    item_df,
    unmatched_df,
    mismatch_df
):

    print("\n" + "=" * 80)
    print("ITEM SALES PIPELINE COMPLETE")
    print("=" * 80)

    print(
        f"\nMatched item rows : "
        f"{len(item_df):,}"
    )

    print(
        f"Unmatched items   : "
        f"{len(unmatched_df):,}"
    )

    print(
        f"Count mismatches  : "
        f"{len(mismatch_df):,}"
    )

    if len(item_df) > 0:

        print(
            f"Unique products   : "
            f"{item_df['ProductId'].nunique():,}"
        )

        print(
            f"Unique categories : "
            f"{item_df['Category'].nunique():,}"
        )

        print("\nTop categories by item count:")

        category_counts = (
            item_df["Category"]
            .value_counts()
            .head(10)
        )

        print(
            category_counts.to_string()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RESTAURANT SALES INTELLIGENCE")
    print("ITEM-LEVEL SALES PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    transactions, products = load_data()

    # --------------------------------------------------------
    # 2. Product catalog
    # --------------------------------------------------------

    products = prepare_product_catalog(
        products
    )

    # --------------------------------------------------------
    # 3. Fast lookup
    # --------------------------------------------------------

    label_lookup, name_lookup = (
        build_product_lookups(
            products
        )
    )

    # --------------------------------------------------------
    # 4. Parse and match
    # --------------------------------------------------------

    (
        item_df,
        unmatched_df,
        mismatch_df
    ) = parse_transaction_items(
        transactions,
        label_lookup,
        name_lookup
    )

    # --------------------------------------------------------
    # 5. Metrics
    # --------------------------------------------------------

    item_df = calculate_item_metrics(
        item_df
    )

    # --------------------------------------------------------
    # 6. Reconciliation
    # --------------------------------------------------------

    reconciliation_report(
        item_df,
        transactions,
        unmatched_df,
        mismatch_df
    )

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    save_outputs(
        item_df,
        unmatched_df,
        mismatch_df
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_summary(
        item_df,
        unmatched_df,
        mismatch_df
    )


if __name__ == "__main__":
    main()
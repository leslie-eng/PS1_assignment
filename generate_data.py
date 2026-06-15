"""
generate_data.py
----------------
Generates four messy CSV files that mimic a simplified e-commerce data model.
Intentional data quality issues are injected as specified in the assessment:
  - ~8%  exact duplicate rows across all tables
  - Mixed date formats: YYYY-MM-DD and DD/MM/YYYY
  - ~4%  NULL customer_id in orders
  - ~2%  NULL total_amount in orders
  - ~3%  negative total_amount in orders (flagged, not dropped)
  - ~4%  orphaned order_items (reference non-existent order IDs)
  - Inconsistent casing on customer_tier (bronze, Bronze, BRONZE, etc.)
  - ~5%  refund_amount exceeding original order total in returns

Usage:
  python generate_data.py
  python generate_data.py --rows 2000 --seed 42 --out ./data
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
from datetime import date
from faker import Faker


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce CSVs.")
    parser.add_argument("--rows", type=int, default=2000, help="Base row count for orders (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", type=str, default="./data", help="Output directory (default: ./data)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_date(fake, start_year=2021, end_year=2024):
    """Return a random date between start_year and end_year."""
    return fake.date_between(
        start_date=date(start_year, 1, 1),
        end_date=date(end_year, 12, 31)
    )


def format_date_mixed(date_obj, rng):
    """
    Return the date as either YYYY-MM-DD or DD/MM/YYYY randomly.
    ~50% chance of each format to ensure both appear across the dataset.
    """
    if rng.random() < 0.5:
        return date_obj.strftime("%Y-%m-%d")
    else:
        return date_obj.strftime("%d/%m/%Y")


def inject_duplicates(df, frac, rng_seed):
    """Append a fraction of randomly sampled rows as exact duplicates."""
    n_dupes = max(1, int(len(df) * frac))
    dupes = df.sample(n=n_dupes, random_state=rng_seed, replace=True)
    return pd.concat([df, dupes], ignore_index=True)


TIER_VARIANTS = {
    "bronze": ["bronze", "Bronze", "BRONZE", "Bronze "],
    "silver": ["silver", "Silver", "SILVER", "Silver "],
    "gold":   ["gold",   "Gold",   "GOLD",   "Gold "],
}

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Toys", "Beauty"]
STATUSES   = ["completed", "pending", "cancelled", "refunded"]
RETURN_REASONS = ["defective", "wrong item", "changed mind", "damaged in shipping", "not as described"]


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_customers(n, fake, rng):
    """Generate the customers master table."""
    tiers = list(TIER_VARIANTS.keys())
    records = []
    for i in range(1, n + 1):
        tier_key = random.choice(tiers)
        # Pick a messy casing variant
        tier_display = random.choice(TIER_VARIANTS[tier_key])
        signup = random_date(fake, 2018, 2022)
        records.append({
            "customer_id":   f"C{i:05d}",
            "customer_name": fake.name(),
            "email":         fake.email(),
            "country":       fake.country_code(representation="alpha-2"),
            "customer_tier": tier_display,
            "signup_date":   format_date_mixed(signup, rng),
        })
    return pd.DataFrame(records)


def generate_orders(n, customer_ids, fake, rng):
    """Generate the orders table with intentional quality issues."""
    records = []
    for i in range(1, n + 1):
        order_date = random_date(fake, 2021, 2024)

        # ~4% NULL customer_id
        cust_id = random.choice(customer_ids) if rng.random() > 0.04 else None

        # ~2% NULL total_amount
        if rng.random() < 0.02:
            total_amount = None
        # ~3% negative total_amount
        elif rng.random() < 0.03:
            total_amount = round(-rng.uniform(1, 500), 2)
        else:
            total_amount = round(rng.uniform(10, 2000), 2)

        discount_pct = round(random.choice([0, 5, 10, 15, 20, 25]), 2)

        records.append({
            "order_id":     f"O{i:06d}",
            "customer_id":  cust_id,
            "order_date":   format_date_mixed(order_date, rng),
            "status":       random.choice(STATUSES),
            "total_amount": total_amount,
            "discount_pct": discount_pct,
        })
    return pd.DataFrame(records)


def generate_order_items(orders_df, fake, rng):
    """
    Generate order_items with 1–5 line items per real order,
    then inject ~4% orphaned rows referencing non-existent order IDs.
    """
    real_order_ids = orders_df["order_id"].dropna().tolist()
    records = []

    item_counter = 1
    for order_id in real_order_ids:
        n_items = rng.integers(1, 6)
        for _ in range(n_items):
            records.append({
                "item_id":    f"I{item_counter:07d}",
                "order_id":  order_id,
                "product_name": fake.word().capitalize() + " " + fake.word().capitalize(),
                "category":   random.choice(CATEGORIES),
                "quantity":   int(rng.integers(1, 11)),
                "unit_price": round(rng.uniform(5, 500), 2),
            })
            item_counter += 1

    df = pd.DataFrame(records)

    # Inject ~4% orphaned items (order IDs that don't exist)
    n_orphans = max(1, int(len(df) * 0.04))
    orphan_records = []
    for _ in range(n_orphans):
        orphan_records.append({
            "item_id":      f"I{item_counter:07d}",
            "order_id":     f"O{rng.integers(900000, 999999):06d}",
            "product_name": fake.word().capitalize() + " " + fake.word().capitalize(),
            "category":     random.choice(CATEGORIES),
            "quantity":     int(rng.integers(1, 11)),
            "unit_price":   round(rng.uniform(5, 500), 2),
        })
        item_counter += 1

    return pd.concat([df, pd.DataFrame(orphan_records)], ignore_index=True)


def generate_returns(orders_df, fake, rng):
    """
    Generate returns for ~20% of completed/refunded orders.
    Inject ~5% where refund_amount exceeds the original order total.
    """
    eligible = orders_df[
        orders_df["status"].isin(["refunded", "completed"]) &
        orders_df["total_amount"].notna() &
        orders_df["order_id"].notna()
    ].copy()

    # Sample ~20% of eligible orders
    sample = eligible.sample(frac=0.20, random_state=int(rng.integers(0, 9999)))

    records = []
    return_counter = 1
    for _, row in sample.iterrows():
        original = float(row["total_amount"])
        return_date = random_date(fake, 2021, 2024)

        # ~5% refund anomaly: refund exceeds original amount
        if rng.random() < 0.05:
            refund_amount = round(original * rng.uniform(1.05, 1.5), 2)
        else:
            refund_amount = round(rng.uniform(1, max(1, original)), 2)

        records.append({
            "return_id":     f"R{return_counter:06d}",
            "order_id":      row["order_id"],
            "return_date":   format_date_mixed(return_date, rng),
            "reason":        random.choice(RETURN_REASONS),
            "refund_amount": refund_amount,
        })
        return_counter += 1

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    fake = Faker()
    Faker.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out, exist_ok=True)

    n_orders    = args.rows
    n_customers = max(100, n_orders // 5)   # ~5 orders per customer on average

    print(f"[generate_data] seed={args.seed} | orders={n_orders} | customers={n_customers}")

    # --- Customers ---
    print("[generate_data] Generating customers...")
    customers_df = generate_customers(n_customers, fake, rng)
    customers_df = inject_duplicates(customers_df, frac=0.08, rng_seed=args.seed)
    customers_df.to_csv(os.path.join(args.out, "customers.csv"), index=False)
    print(f"             -> {len(customers_df)} rows (incl. duplicates)")

    # --- Orders ---
    print("[generate_data] Generating orders...")
    customer_ids = customers_df["customer_id"].unique().tolist()
    orders_df_raw = generate_orders(n_orders, customer_ids, fake, rng)
    orders_df = inject_duplicates(orders_df_raw, frac=0.08, rng_seed=args.seed)
    orders_df.to_csv(os.path.join(args.out, "orders.csv"), index=False)
    print(f"             -> {len(orders_df)} rows (incl. duplicates)")

    # --- Order Items ---
    print("[generate_data] Generating order_items...")
    items_df = generate_order_items(orders_df_raw, fake, rng)
    items_df = inject_duplicates(items_df, frac=0.08, rng_seed=args.seed)
    items_df.to_csv(os.path.join(args.out, "order_items.csv"), index=False)
    print(f"             -> {len(items_df)} rows (incl. duplicates + orphans)")

    # --- Returns ---
    print("[generate_data] Generating returns...")
    returns_df = generate_returns(orders_df_raw, fake, rng)
    returns_df = inject_duplicates(returns_df, frac=0.08, rng_seed=args.seed)
    returns_df.to_csv(os.path.join(args.out, "returns.csv"), index=False)
    print(f"             -> {len(returns_df)} rows (incl. duplicates)")

    print(f"\n[generate_data] Done. Files written to: {os.path.abspath(args.out)}/")
    print("  customers.csv")
    print("  orders.csv")
    print("  order_items.csv")
    print("  returns.csv")


if __name__ == "__main__":
    main()

"""
Generate synthetic operations and product data for the 100 US retail stores
that match the scraped store catalog (USR-001 … USR-100).

Reads store_catalog from the scraped Excel workbook (or CSV fallback), assigns
retailer-aware archetypes, and writes correlated ops + product tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse generators and vocabularies from the main dummy-data script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_dummy_data import (  # noqa: E402
    ARCHETYPES,
    CATEGORIES,
    PRODUCT_TEMPLATES,
    _archetype_map,
    _format_sqft,
    generate_assortment,
    generate_operational_reports,
    generate_operations_weekly,
)
from scraped_100_data_dictionary import build_data_dictionary  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Retailer → hidden archetype (drives ops KPIs and assortment depth)
RETAILER_ARCHETYPE = {
    "Walmart": "value_suburban",
    "Target": "premium_urban",
    "Kroger": "value_suburban",
    "Costco": "affluent_supercenter",
    "Albertsons": "value_suburban",
    "Publix": "premium_urban",
    "Whole Foods": "premium_urban",
    "CVS": "high_traffic_express",
    "Walgreens": "high_traffic_express",
    "Home Depot": "affluent_supercenter",
    "Lowe's": "value_suburban",
    "Best Buy": "premium_urban",
    "Dollar General": "rural_heartland",
    "Sam's Club": "affluent_supercenter",
    "Aldi": "promo_driven_suburban",
    "H-E-B": "premium_urban",
    "Meijer": "value_suburban",
    "Safeway": "value_suburban",
    "Sprouts": "premium_urban",
    "Trader Joe's": "premium_urban",
}

# Extra product lines by retailer type (name, description, attrs, category)
RETAILER_PRODUCTS: dict[str, list[tuple[str, str, str, str]]] = {
    "Walmart": [
        ("Great Value Whole Milk Gallon", "Store-brand vitamin D milk — everyday low price.", "private_label,value", "Dairy"),
        ("Equate Pain Relief 200ct", "Ibuprofen tablets — pharmacy aisle staple.", "private_label,value", "Health & Beauty"),
        ("Mainstays Bath Towel Set", "Soft cotton towels — home basics collection.", "private_label,value", "Household"),
        ("Marketside Fresh Sub Kit", "Deli sub ingredients — ready for assembly.", "value", "Grocery"),
        ("Ozark Trail Camping Chair", "Foldable outdoor chair — seasonal general merchandise.", "value", "Household"),
    ],
    "Target": [
        ("Good & Gather Organic Pasta", "Target-owned organic penne — clean label.", "organic,premium,private_label", "Grocery"),
        ("Up&Up Disinfecting Wipes 75ct", "Private-label wipes — household essentials.", "private_label,value", "Household"),
        ("Cat & Jack Kids Tee", "Soft cotton graphic tee — apparel cross-sell.", "value", "Grocery"),
        ("Favorite Day Cookie Butter Cookies", "Private-label snack — impulse endcap item.", "private_label,premium", "Snacks"),
    ],
    "Costco": [
        ("Kirkland Signature Paper Towels 12-Pack", "Warehouse bulk pack — high-velocity SKU.", "private_label,premium", "Household"),
        ("Rotisserie Chicken", "Fresh prepared protein — member favorite.", "value", "Fresh Produce"),
        ("Kirkland Signature Olive Oil 2L", "Extra virgin — pantry staple in club size.", "private_label,premium", "Grocery"),
        ("Executive Membership Renewal", "Annual membership fee SKU — loyalty driver.", "premium", "Grocery"),
    ],
    "Sam's Club": [
        ("Member's Mark Purified Water 40-Pack", "Bulk hydration — high basket attach rate.", "private_label,value", "Beverages"),
        ("Sam's Club Bakery Muffin 12-Pack", "In-club bakery — morning traffic driver.", "value", "Grocery"),
        ("Member's Mark Laundry Pods 81ct", "Concentrated detergent — club pack value.", "private_label,value", "Household"),
    ],
    "Whole Foods": [
        ("365 Organic Almond Butter", "House brand nut butter — clean ingredients.", "organic,premium,private_label", "Grocery"),
        ("Wild-Caught Sockeye Salmon Fillet", "Sustainable seafood — seafood counter.", "organic,premium", "Fresh Produce"),
        ("Organic Heirloom Tomatoes", "Peak-season produce — local farm partner.", "organic,seasonal,premium", "Fresh Produce"),
    ],
    "Trader Joe's": [
        ("Unexpected Cheddar", "Flagship cheese — cult favorite private label.", "premium,private_label", "Dairy"),
        ("Mandarin Orange Chicken", "Frozen bestseller — high repeat purchase.", "value", "Frozen"),
        ("Everything But The Bagel Seasoning", "Signature seasoning blend — social buzz SKU.", "premium,private_label", "Grocery"),
    ],
    "CVS": [
        ("CVS Health Ibuprofen 200mg", "Private-label pain relief — front-store anchor.", "private_label,value", "Health & Beauty"),
        ("Beauty 360 Facial Tissues", "Soft tissues — health & wellness aisle.", "private_label,value", "Health & Beauty"),
        ("Gold Emblem Abound Protein Bar", "Private-label snack — checkout lane.", "private_label,premium", "Snacks"),
        ("Prescription Generic Lisinopril 30ct", "Pharmacy generic — high script volume.", "value", "Health & Beauty"),
    ],
    "Walgreens": [
        ("Walgreens Brand Allergy Relief 24ct", "Seasonal OTC — private label margin driver.", "private_label,value", "Health & Beauty"),
        ("Nice! Purified Water 24-Pack", "Private-label beverage multipack.", "private_label,value", "Beverages"),
        ("Photo Print 4x6 50-Pack", "Photo service SKU — services attachment.", "value", "Household"),
    ],
    "Home Depot": [
        ("HDX 5-Gallon Paint Bucket", "Contractor-grade bucket — paint aisle.", "private_label,value", "Household"),
        ("Ridgid 18V Drill/Driver Kit", "Power tools — pro and DIY segment.", "premium", "Household"),
        ("Scotts Turf Builder Lawn Food", "Seasonal lawn care — garden center.", "premium", "Household"),
        ("2x4 Stud Lumber 8ft", "Framing lumber — building materials bay.", "value", "Household"),
    ],
    "Lowe's": [
        ("Kobalt 24V Cordless Mower", "Private-label outdoor power — spring seasonal.", "private_label,premium", "Household"),
        ("Allen + Roth Vanity Light", "Private-label lighting — home décor.", "private_label,premium", "Household"),
        ("Sta-Green Lawn Fertilizer 15M", "Lawn care — seasonal category.", "value", "Household"),
    ],
    "Best Buy": [
        ("Insignia 55\" 4K Fire TV", "Private-label electronics — margin-friendly TV.", "private_label,premium", "Household"),
        ("Geek Squad Protection 2-Year", "Service plan attachment — high attach on CE.", "premium", "Household"),
        ("Apple AirPods Pro (3rd Gen)", "Flagship audio — traffic-driving CE.", "premium", "Household"),
        ("Samsung Galaxy S26 Unlocked", "Mobile handset — carrier partnership bay.", "premium", "Household"),
    ],
    "Dollar General": [
        ("DG Home Paper Plates 100ct", "Value party supplies — small box format.", "private_label,value", "Household"),
        ("Clover Valley Cereal 12oz", "Private-label breakfast — penny-profit driver.", "private_label,value", "Grocery"),
        ("DG Health Pain Relief 50ct", "Small-count OTC — rural convenience.", "private_label,value", "Health & Beauty"),
    ],
    "Aldi": [
        ("Simply Nature Organic Granola", "Aldi organic private label — ALDI finds.", "organic,premium,private_label", "Grocery"),
        ("Friendly Farms Greek Yogurt", "Dairy private label — weekly special.", "private_label,value", "Dairy"),
        ("Benton's Cookie Snack Packs", "Impulse cookies — checkout cooler.", "private_label,value", "Snacks"),
    ],
}

AISLE_ZONES = {
    "Grocery": "Aisle 4–12 — Center Store",
    "Dairy": "Aisle 1 — Cooler Wall",
    "Snacks": "Aisle 14 — Impulse / Endcap",
    "Beverages": "Aisle 2–3 — Cooler & Shelf",
    "Health & Beauty": "Aisle 18–22 — HBC",
    "Household": "Aisle 15–17 — Household",
    "Fresh Produce": "Produce Section — Front Perimeter",
    "Frozen": "Aisle 23–25 — Freezer Bank",
}

PACK_SIZES = {
    "value": ["12 oz", "16 oz", "1 lb", "24 ct", "1 gallon", "2-liter"],
    "premium": ["8 oz", "10 oz", "32 oz", "4-pack", "6-pack", "2 lb"],
    "mid": ["14 oz", "20 oz", "18 ct", "3-pack", "1.5 liter"],
}


def load_store_catalog(path: Path) -> pd.DataFrame:
    """Load store_catalog from Excel workbook or CSV."""
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name="store_catalog")
    return pd.read_csv(path)


def build_stores_for_generation(catalog: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Enrich catalog with archetype and sq_ft required by ops generators."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in catalog.iterrows():
        retailer = row["retailer"]
        store_format = row["store_format"]
        archetype_name = RETAILER_ARCHETYPE.get(retailer, "value_suburban")
        if store_format == "Supercenter" and archetype_name == "value_suburban":
            archetype_name = "promo_driven_suburban"
        sq_ft = _format_sqft(store_format, rng)
        rows.append({
            **row.to_dict(),
            "archetype": archetype_name,
            "sq_ft": sq_ft,
            "has_pharmacy": retailer in ("CVS", "Walgreens", "Kroger", "Walmart", "Target", "Costco", "Sam's Club"),
            "has_online_pickup": retailer not in ("Dollar General",),
        })
    return pd.DataFrame(rows)


def generate_detailed_products(stores: pd.DataFrame, products_per_store: int, seed: int) -> pd.DataFrame:
    """SKU-level product catalog with pricing, aisle, and velocity fields."""
    rng = np.random.default_rng(seed + 20)
    archetypes = _archetype_map()
    format_depth = {"Supercenter": 1.0, "Neighborhood": 0.8, "Express": 0.5}
    rows = []
    price_bands = {"value": (1.49, 12.99), "mid": (4.99, 24.99), "premium": (8.99, 89.99)}

    for _, store in stores.iterrows():
        arch = archetypes[store["archetype"]]
        depth = format_depth.get(store["store_format"], 0.8)
        retailer = store["retailer"]
        target_count = max(12, int(products_per_store * depth * rng.normal(1.0, 0.08)))

        candidates: list[tuple[str, str, str, str]] = []
        for category, templates in PRODUCT_TEMPLATES.items():
            for tpl in templates:
                candidates.append((tpl[0], tpl[1], tpl[2], category))

        for extra in RETAILER_PRODUCTS.get(retailer, []):
            candidates.append(extra)

        rng.shuffle(candidates)
        chosen = candidates[:target_count]

        sku_idx = 0
        for name, description, attrs, category in chosen:
            if "premium" in arch["name"] or "affluent" in arch["name"]:
                if "value" in attrs and rng.random() < 0.25:
                    continue
            if arch["name"] in ("rural_heartland", "promo_driven_suburban"):
                if "premium" in attrs and rng.random() < 0.30:
                    continue

            sku_idx += 1
            price_tier = (
                "premium" if "premium" in attrs else
                "value" if "value" in attrs else "mid"
            )
            low, high = price_bands[price_tier]
            unit_price = round(rng.uniform(low, high), 2)
            margin = round(rng.uniform(0.12, 0.42) if "private_label" in attrs else rng.uniform(0.08, 0.28), 3)
            facings = int(max(1, rng.integers(1, 6) if store["store_format"] == "Neighborhood" else rng.integers(2, 8)))

            rows.append({
                "sku_id": f"SKU-{store['store_id']}-{sku_idx:03d}",
                "store_id": store["store_id"],
                "retailer": retailer,
                "store_name": store["store_name"],
                "city": store["city"],
                "state": store["state"],
                "category": category,
                "product_name": name,
                "description": description,
                "attributes": attrs,
                "brand_type": (
                    "private_label" if "private_label" in attrs
                    else rng.choice(["national", "regional"], p=[0.78, 0.22])
                ),
                "in_stock": rng.random() > arch["oos_rate"],
                "price_tier": price_tier,
                "unit_price": unit_price,
                "pack_size": rng.choice(PACK_SIZES[price_tier]),
                "aisle_zone": AISLE_ZONES.get(category, "General Merchandise"),
                "facings": facings,
                "weekly_units_sold": int(max(5, rng.poisson(45 if price_tier == "value" else 18))),
                "margin_pct": margin,
                "upc": f"{rng.integers(100000, 999999)}{rng.integers(100000, 999999)}",
                "vendor": f"{retailer} DC" if "private_label" in attrs else rng.choice(
                    ["National Brands Co", "Regional Foods LLC", "Global Consumer Goods"]
                ),
            })

    return pd.DataFrame(rows)


def enrich_with_store_context(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Add retailer / store_name for easier Excel browsing."""
    ctx = stores[["store_id", "retailer", "store_name", "city", "state", "store_format"]]
    if "retailer" in df.columns:
        return df
    return df.merge(ctx, on="store_id", how="left")


def export_excel(
    catalog: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dictionary = build_data_dictionary()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data_dictionary.to_excel(writer, sheet_name="data_dictionary", index=False)
        catalog.to_excel(writer, sheet_name="store_catalog", index=False)
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic operations + products for scraped 100 US stores"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "excel" / "scraped_100_reviews_news1.xlsx",
        help="Excel with store_catalog sheet (or CSV with same columns)",
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "us_retail_stores_100.csv",
        help="Fallback CSV if Excel catalog missing",
    )
    parser.add_argument("--weeks", type=int, default=52, help="Weeks of ops history")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--products-per-store", type=int, default=22, help="Target SKUs per store")
    parser.add_argument("--reports-per-store", type=int, default=10, help="Operational reports per store")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "scraped_100" / "synthetic",
        help="CSV output directory",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=PROJECT_ROOT / "data" / "excel" / "scraped_100_operations_products.xlsx",
        help="Combined Excel workbook path",
    )
    args = parser.parse_args()

    catalog_path = args.catalog if args.catalog.exists() else args.catalog_csv
    print(f"Loading store catalog: {catalog_path}")
    catalog = load_store_catalog(catalog_path)
    assert len(catalog) == 100, f"Expected 100 stores, got {len(catalog)}"

    stores = build_stores_for_generation(catalog, args.seed)
    assortment = generate_assortment(stores, args.seed)
    operations = generate_operations_weekly(stores, args.weeks, args.seed)
    reports = generate_operational_reports(stores, args.weeks, args.reports_per_store, args.seed)
    products = generate_detailed_products(stores, args.products_per_store, args.seed)

    operations_out = enrich_with_store_context(operations, stores)
    reports_out = enrich_with_store_context(reports, stores)
    assortment_out = enrich_with_store_context(assortment, stores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.output_dir / "store_catalog.csv", index=False)
    assortment_out.to_csv(args.output_dir / "assortment_snapshot.csv", index=False)
    operations_out.to_csv(args.output_dir / "operations_weekly.csv", index=False)
    reports_out.to_csv(args.output_dir / "operational_reports.csv", index=False)
    products.to_csv(args.output_dir / "product_descriptions.csv", index=False)

    meta = {
        "source_catalog": str(catalog_path),
        "generated_at": date.today().isoformat(),
        "n_stores": len(catalog),
        "n_weeks": args.weeks,
        "seed": args.seed,
        "tables": {
            "store_catalog": "100 US stores — same IDs as scraped reviews/news",
            "assortment_snapshot": "Category SKU counts, facings, private-label share",
            "operations_weekly": "Weekly labor, shrink, OOS, fulfillment, complaints",
            "operational_reports": "Field audits and visit reports with issues",
            "product_descriptions": "Detailed SKU catalog per store with price and aisle",
        },
        "retailers": sorted(catalog["retailer"].unique().tolist()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    export_excel(
        catalog,
        {
            "assortment_snapshot": assortment_out,
            "operations_weekly": operations_out,
            "operational_reports": reports_out,
            "product_descriptions": products,
        },
        args.excel,
    )

    print(f"\nWrote CSVs → {args.output_dir}")
    print(f"Wrote Excel → {args.excel}\n")
    print(f"  store_catalog.csv           {len(catalog):>8,} rows")
    print(f"  assortment_snapshot.csv     {len(assortment_out):>8,} rows")
    print(f"  operations_weekly.csv       {len(operations_out):>8,} rows")
    print(f"  operational_reports.csv     {len(reports_out):>8,} rows")
    print(f"  product_descriptions.csv    {len(products):>8,} rows")
    print("\nDone.")


if __name__ == "__main__":
    main()

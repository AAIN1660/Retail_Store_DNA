"""
Generate synthetic Retail StoreDNA datasets for local development.

Creates correlated store-level data so clustering and peer matching are meaningful:
each store is assigned a hidden "archetype" that drives sales, ops, and demographics.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Archetypes — hidden store personas that shape all downstream metrics
# ---------------------------------------------------------------------------
ARCHETYPES = [
  {
    "name": "premium_urban",
    "format_weights": {"Supercenter": 0.2, "Neighborhood": 0.6, "Express": 0.2},
    "base_weekly_revenue": 520_000,
    "promo_dependency": 0.12,
    "basket_size": 42,
    "shrink_pct": 0.018,
    "oos_rate": 0.04,
    "median_income": 92_000,
    "urbanicity": "urban",
  },
  {
    "name": "value_suburban",
    "format_weights": {"Supercenter": 0.55, "Neighborhood": 0.35, "Express": 0.10},
    "base_weekly_revenue": 410_000,
    "promo_dependency": 0.28,
    "basket_size": 34,
    "shrink_pct": 0.024,
    "oos_rate": 0.06,
    "median_income": 58_000,
    "urbanicity": "suburban",
  },
  {
    "name": "rural_heartland",
    "format_weights": {"Supercenter": 0.70, "Neighborhood": 0.25, "Express": 0.05},
    "base_weekly_revenue": 290_000,
    "promo_dependency": 0.18,
    "basket_size": 31,
    "shrink_pct": 0.015,
    "oos_rate": 0.08,
    "median_income": 48_000,
    "urbanicity": "rural",
  },
  {
    "name": "high_traffic_express",
    "format_weights": {"Supercenter": 0.05, "Neighborhood": 0.25, "Express": 0.70},
    "base_weekly_revenue": 180_000,
    "promo_dependency": 0.15,
    "basket_size": 22,
    "shrink_pct": 0.030,
    "oos_rate": 0.05,
    "median_income": 72_000,
    "urbanicity": "urban",
  },
  {
    "name": "promo_driven_suburban",
    "format_weights": {"Supercenter": 0.45, "Neighborhood": 0.45, "Express": 0.10},
    "base_weekly_revenue": 380_000,
    "promo_dependency": 0.38,
    "basket_size": 36,
    "shrink_pct": 0.022,
    "oos_rate": 0.07,
    "median_income": 54_000,
    "urbanicity": "suburban",
  },
  {
    "name": "affluent_supercenter",
    "format_weights": {"Supercenter": 0.80, "Neighborhood": 0.15, "Express": 0.05},
    "base_weekly_revenue": 610_000,
    "promo_dependency": 0.10,
    "basket_size": 48,
    "shrink_pct": 0.016,
    "oos_rate": 0.03,
    "median_income": 105_000,
    "urbanicity": "suburban",
  },
]

CATEGORIES = [
  "Grocery",
  "Dairy",
  "Snacks",
  "Beverages",
  "Health & Beauty",
  "Household",
  "Fresh Produce",
  "Frozen",
]

# Category share of revenue (rough grocery mix)
CATEGORY_SHARE = {
  "Grocery": 0.22,
  "Dairy": 0.10,
  "Snacks": 0.12,
  "Beverages": 0.14,
  "Health & Beauty": 0.11,
  "Household": 0.13,
  "Fresh Produce": 0.10,
  "Frozen": 0.08,
}

STATES = [
  ("TX", "Dallas"), ("TX", "Houston"), ("TX", "Austin"),
  ("CA", "Los Angeles"), ("CA", "San Diego"), ("CA", "Sacramento"),
  ("FL", "Miami"), ("FL", "Tampa"), ("FL", "Orlando"),
  ("NY", "Buffalo"), ("NY", "Albany"), ("NY", "Rochester"),
  ("IL", "Chicago"), ("IL", "Springfield"),
  ("GA", "Atlanta"), ("GA", "Savannah"),
  ("OH", "Columbus"), ("OH", "Cleveland"),
  ("PA", "Philadelphia"), ("PA", "Pittsburgh"),
  ("WA", "Seattle"), ("WA", "Spokane"),
]

# --- Unstructured / semi-structured data vocabularies ---
COMPLAINT_TAGS = [
  "checkout_wait", "out_of_stock", "cleanliness", "staff_attitude",
  "pricing", "parking", "product_quality", "store_layout", "online_pickup",
]

IMAGE_ZONES = [
  "exterior", "main_aisle", "produce_section", "checkout",
  "endcap_display", "freezer_aisle", "pharmacy_front",
]

REPORT_TYPES = [
  "mystery_shop", "safety_audit", "inventory_audit",
  "facilities", "loss_prevention", "district_visit",
]

ISSUE_CATEGORIES = [
  "staffing", "cleaning", "equipment", "inventory", "safety",
  "signage", "planogram", "shrink", "customer_service",
]

NEWS_EVENT_TYPES = [
  "weather", "local_festival", "road_closure", "new_competitor",
  "community_event", "school_holiday", "construction", "sports_event",
]

REVIEW_SNIPPETS = {
  "positive": [
    "Great selection and the store was well organized today.",
    "Friendly staff helped me find everything quickly.",
    "Clean aisles and short checkout lines — pleasant visit.",
    "Fresh produce section looks excellent compared to nearby stores.",
  ],
  "neutral": [
    "Average shopping trip — nothing stood out good or bad.",
    "Store was busy but manageable. Some shelves looked thin.",
    "Decent prices but checkout took longer than expected.",
    "Standard experience; parking lot was crowded on weekend.",
  ],
  "negative": [
    "Several items were out of stock and aisles were messy.",
    "Long wait at checkout with only two lanes open.",
    "Floor was dirty near the entrance and carts were sticky.",
    "Staff seemed overwhelmed; couldn't get help in dairy aisle.",
    "Prices higher than competitors and promotional tags were wrong.",
  ],
}

NEWS_HEADLINES = {
  "weather": [
    "{city} braces for severe storms this weekend",
    "Heat wave expected to push {city} temperatures above 100F",
    "Winter storm warning issued for {city} metro area",
  ],
  "local_festival": [
    "{city} summer food festival draws record crowds downtown",
    "Annual {city} street fair returns — road closures planned",
  ],
  "road_closure": [
    "Major highway repair near {city} store corridor causes detours",
    "Bridge closure on Main St impacts access to retail district",
  ],
  "new_competitor": [
    "Discount grocer announces new {city} location opening Q3",
    "Competitor supercenter remodel planned two miles from {city} plaza",
  ],
  "community_event": [
    "{city} high school graduation weekend boosts local foot traffic",
    "Charity 5K in {city} park expected to draw thousands Saturday",
  ],
  "school_holiday": [
    "{city} schools closed for spring break — families stock up",
    "Back-to-school week begins across {city} district",
  ],
  "construction": [
    "Retail plaza construction in {city} limits parking through August",
  ],
  "sports_event": [
    "{city} hosts regional championship — hotels and stores see surge",
  ],
}

PRODUCT_TEMPLATES = {
  "Grocery": [
    ("Organic Whole Grain Bread", "Artisan loaf baked daily with organic whole wheat flour.", "organic,premium"),
    ("Value White Bread", "Soft sandwich bread — family pack at everyday low price.", "value"),
    ("Gluten-Free Multigrain Loaf", "Certified gluten-free bread with seeds and ancient grains.", "gluten_free,premium"),
  ],
  "Dairy": [
    ("Farm Fresh 2% Milk Gallon", "Locally sourced milk — vitamin D fortified.", "value"),
    ("Organic Greek Yogurt 4-Pack", "Thick strained yogurt, live cultures, no artificial sweeteners.", "organic,premium"),
    ("Plant-Based Oat Milk", "Barista blend oat milk — dairy-free alternative.", "premium,plant_based"),
  ],
  "Snacks": [
    ("Classic Potato Chips Family Size", "Crispy kettle chips — original sea salt flavor.", "value"),
    ("Protein Trail Mix Grab & Go", "Almonds, cashews, dried cranberries — 10g protein per serving.", "premium"),
    ("Store Brand Tortilla Chips", "Restaurant style chips — great for parties.", "private_label,value"),
  ],
  "Beverages": [
    ("Sparkling Water 12-Pack", "Zero calorie flavored seltzer — variety pack.", "premium"),
    ("Cola 2-Liter Bottle", "Classic cola — promotional multipack available.", "value"),
    ("Cold Brew Coffee Concentrate", "Ready-to-drink cold brew — premium arabica beans.", "premium"),
  ],
  "Health & Beauty": [
    ("Sensitive Skin Moisturizer", "Dermatologist tested — fragrance free daily lotion.", "premium"),
    ("Value Shampoo Family Size", "Everyday clean formula — large bottle value pack.", "value"),
    ("Organic Lip Balm 3-Pack", "Beeswax and coconut oil — USDA organic.", "organic,premium"),
  ],
  "Household": [
    ("Concentrated Laundry Detergent", "HE compatible — 64 load bottle.", "value"),
    ("Eco-Friendly Surface Cleaner", "Plant-based spray — safe for kitchen counters.", "organic,premium"),
    ("Paper Towels 8-Roll Pack", "Absorbent sheets — store brand savings.", "private_label,value"),
  ],
  "Fresh Produce": [
    ("Organic Baby Spinach 5oz", "Triple washed organic greens — salad ready.", "organic,premium"),
    ("Bananas per lb", "Everyday fruit staple — ripen at home.", "value"),
    ("Local Seasonal Berry Mix", "Farm partnership program — peak season berries.", "seasonal,premium"),
  ],
  "Frozen": [
    ("Frozen Vegetable Stir-Fry Blend", "Broccoli, peppers, snap peas — steam in bag.", "value"),
    ("Premium Ice Cream Pint", "Super-premium butterfat — limited edition flavor.", "premium"),
    ("Family Pizza Night 3-Pack", "Frozen pizzas — cheese, pepperoni, veggie variety.", "value"),
  ],
}


def _pick_format(rng: np.random.Generator, archetype: dict) -> str:
  formats = list(archetype["format_weights"].keys())
  weights = list(archetype["format_weights"].values())
  return rng.choice(formats, p=weights)


def _format_sqft(format_name: str, rng: np.random.Generator) -> int:
  ranges = {
    "Supercenter": (120_000, 200_000),
    "Neighborhood": (40_000, 90_000),
    "Express": (10_000, 25_000),
  }
  low, high = ranges[format_name]
  return int(rng.integers(low, high))


def generate_store_master(n_stores: int, seed: int) -> pd.DataFrame:
  """One row per store with master attributes and hidden archetype label."""
  rng = np.random.default_rng(seed)
  rows = []

  for i in range(1, n_stores + 1):
    archetype = ARCHETYPES[i % len(ARCHETYPES)]
    state, city = STATES[rng.integers(0, len(STATES))]
    store_format = _pick_format(rng, archetype)
    sq_ft = _format_sqft(store_format, rng)
    open_year = int(rng.integers(1998, 2023))

    rows.append({
      "store_id": f"STR-{i:04d}",
      "retailer": "RetailCo",
      "banner": "RetailCo",
      "store_name": f"RetailCo {city} #{i}",
      "store_format": store_format,
      "address": f"{rng.integers(100, 9999)} Main St",
      "city": city,
      "state": state,
      "zip_code": f"{rng.integers(10000, 99999)}",
      "sq_ft": sq_ft,
      "open_date": f"{open_year}-{rng.integers(1, 12):02d}-01",
      "has_pharmacy": store_format == "Supercenter" or rng.random() < 0.4,
      "has_online_pickup": rng.random() < 0.85,
      "archetype": archetype["name"],  # useful for validating DNA models; drop in prod
    })

  return pd.DataFrame(rows)


def generate_demographics(stores: pd.DataFrame, seed: int) -> pd.DataFrame:
  """Store-level trade-area demographics aligned to archetype."""
  rng = np.random.default_rng(seed + 1)
  archetype_map = {a["name"]: a for a in ARCHETYPES}
  rows = []

  for _, store in stores.iterrows():
    arch = archetype_map[store["archetype"]]
    income = arch["median_income"] * rng.normal(1.0, 0.08)
    rows.append({
      "store_id": store["store_id"],
      "population_5mi": int(rng.integers(25_000, 180_000)),
      "median_income": int(max(35_000, income)),
      "median_age": round(rng.normal(38, 5), 1),
      "household_size": round(rng.normal(2.6, 0.4), 2),
      "urbanicity": arch["urbanicity"],
      "competitor_count_3mi": int(rng.integers(1, 8)),
    })

  return pd.DataFrame(rows)


def generate_assortment(stores: pd.DataFrame, seed: int) -> pd.DataFrame:
  """Category-level assortment snapshot per store."""
  rng = np.random.default_rng(seed + 2)
  archetype_map = {a["name"]: a for a in ARCHETYPES}
  rows = []

  format_sku_multiplier = {"Supercenter": 1.4, "Neighborhood": 1.0, "Express": 0.55}

  for _, store in stores.iterrows():
    mult = format_sku_multiplier[store["store_format"]]
    for category in CATEGORIES:
      base_skus = {"Grocery": 1200, "Dairy": 180, "Snacks": 420, "Beverages": 350,
                   "Health & Beauty": 500, "Household": 380, "Fresh Produce": 220, "Frozen": 260}
      sku_count = int(base_skus[category] * mult * rng.normal(1.0, 0.1))
      rows.append({
        "store_id": store["store_id"],
        "category": category,
        "sku_count": max(20, sku_count),
        "avg_facings": round(rng.normal(3.5, 0.8), 1),
        "private_label_share": round(rng.normal(0.22, 0.05), 3),
      })

  return pd.DataFrame(rows)


def _week_ends(n_weeks: int, end_date: date) -> list[date]:
  """Return n_weeks Sunday week-end dates counting backward from end_date."""
  weeks = []
  d = end_date
  while d.weekday() != 6:
    d -= timedelta(days=1)
  for _ in range(n_weeks):
    weeks.append(d)
    d -= timedelta(days=7)
  return list(reversed(weeks))


def generate_sales_weekly(stores: pd.DataFrame, n_weeks: int, seed: int) -> pd.DataFrame:
  """Weekly category sales with archetype-driven revenue and seasonality."""
  rng = np.random.default_rng(seed + 3)
  archetype_map = {a["name"]: a for a in ARCHETYPES}
  week_dates = _week_ends(n_weeks, date.today())
  rows = []

  for _, store in stores.iterrows():
    arch = archetype_map[store["archetype"]]
    store_factor = rng.normal(1.0, 0.12)
    weekly_total = arch["base_weekly_revenue"] * store_factor

    for week_idx, week_end in enumerate(week_dates):
      # Mild seasonality — higher in summer and holidays
      month = week_end.month
      seasonality = 1.0 + 0.08 * np.sin(2 * np.pi * month / 12)
      if month in (11, 12):
        seasonality += 0.12
      week_noise = rng.normal(1.0, 0.05)
      week_revenue = weekly_total * seasonality * week_noise

      for category in CATEGORIES:
        share = CATEGORY_SHARE[category] * rng.normal(1.0, 0.06)
        cat_revenue = week_revenue * share
        promo_pct = arch["promo_dependency"] * rng.normal(1.0, 0.15)
        basket = arch["basket_size"] * rng.normal(1.0, 0.08)
        transactions = max(100, int(cat_revenue / basket))

        rows.append({
          "store_id": store["store_id"],
          "week_end_date": week_end.isoformat(),
          "category": category,
          "revenue": round(max(0, cat_revenue), 2),
          "units": int(max(0, cat_revenue / rng.uniform(2.5, 8.0))),
          "transactions": transactions,
          "promo_sales_pct": round(min(0.65, max(0.02, promo_pct)), 3),
        })

  return pd.DataFrame(rows)


def generate_operations_weekly(stores: pd.DataFrame, n_weeks: int, seed: int) -> pd.DataFrame:
  """Weekly operational KPIs correlated with store archetype."""
  rng = np.random.default_rng(seed + 4)
  archetype_map = {a["name"]: a for a in ARCHETYPES}
  week_dates = _week_ends(n_weeks, date.today())
  rows = []

  labor_per_1k_sqft = {"Supercenter": 18, "Neighborhood": 22, "Express": 28}

  for _, store in stores.iterrows():
    arch = archetype_map[store["archetype"]]
    labor_base = labor_per_1k_sqft[store["store_format"]] * (store["sq_ft"] / 1000)

    for week_end in week_dates:
      rows.append({
        "store_id": store["store_id"],
        "week_end_date": week_end.isoformat(),
        "labor_hours": round(labor_base * rng.normal(1.0, 0.06), 1),
        "shrink_pct": round(max(0.005, arch["shrink_pct"] * rng.normal(1.0, 0.12)), 4),
        "oos_rate": round(min(0.20, max(0.01, arch["oos_rate"] * rng.normal(1.0, 0.15))), 4),
        "fulfillment_rate": round(min(0.99, max(0.85, rng.normal(0.94, 0.03))), 3),
        "customer_complaints": int(max(0, rng.poisson(3 if arch["urbanicity"] == "urban" else 1))),
      })

  return pd.DataFrame(rows)


def _archetype_map() -> dict:
  return {a["name"]: a for a in ARCHETYPES}


def _random_date_in_range(rng: np.random.Generator, start: date, end: date) -> date:
  days = (end - start).days
  return start + timedelta(days=int(rng.integers(0, max(1, days))))


def _condition_base(archetype_name: str) -> float:
  """Higher = better store condition / lower complaints (premium stores score higher)."""
  scores = {
    "premium_urban": 8.2,
    "affluent_supercenter": 8.5,
    "value_suburban": 6.8,
    "rural_heartland": 7.0,
    "high_traffic_express": 6.2,
    "promo_driven_suburban": 6.5,
  }
  return scores.get(archetype_name, 7.0)


def generate_customer_reviews(
  stores: pd.DataFrame,
  n_weeks: int,
  reviews_per_store: int,
  seed: int,
) -> pd.DataFrame:
  """
  Customer reviews with sentiment and complaint tags.
  Urban / high-traffic stores get more reviews; condition drives negative share.
  """
  rng = np.random.default_rng(seed + 10)
  end = date.today()
  start = end - timedelta(weeks=n_weeks)
  rows = []

  for _, store in stores.iterrows():
    arch_name = store["archetype"]
    condition = _condition_base(arch_name)
    n_reviews = max(5, int(reviews_per_store * rng.normal(1.0, 0.2)))

    for i in range(n_reviews):
      neg_prob = max(0.08, 0.35 - (condition - 6) * 0.04)
      roll = rng.random()
      if roll < neg_prob:
        sentiment = "negative"
      elif roll < neg_prob + 0.25:
        sentiment = "neutral"
      else:
        sentiment = "positive"

      rating_map = {"positive": (4, 5), "neutral": (3, 4), "negative": (1, 3)}
      rating = int(rng.integers(rating_map[sentiment][0], rating_map[sentiment][1] + 1))

      complaints = []
      if sentiment in ("negative", "neutral"):
        n_tags = int(rng.integers(1, 3))
        pool = list(COMPLAINT_TAGS)
        if arch_name == "high_traffic_express":
          pool.extend(["checkout_wait", "checkout_wait"])
        if condition < 7:
          pool.extend(["cleanliness", "out_of_stock"])
        complaints = list(rng.choice(pool, size=min(n_tags, len(pool)), replace=False))

      snippet = rng.choice(REVIEW_SNIPPETS[sentiment])
      rows.append({
        "review_id": f"REV-{store['store_id']}-{i+1:03d}",
        "store_id": store["store_id"],
        "review_date": _random_date_in_range(rng, start, end).isoformat(),
        "source": rng.choice(["google", "yelp", "survey", "app"], p=[0.4, 0.3, 0.2, 0.1]),
        "rating": rating,
        "sentiment": sentiment,
        "review_text": snippet,
        "complaint_tags": "|".join(complaints) if complaints else "",
      })

  return pd.DataFrame(rows)


def generate_image_audits(
  stores: pd.DataFrame,
  n_weeks: int,
  images_per_store: int,
  seed: int,
) -> pd.DataFrame:
  """
  Simulated computer-vision audit outputs from store photos.
  Represents store condition, shelf utilization, and branding compliance.
  (Image file paths are synthetic — no binary files generated.)
  """
  rng = np.random.default_rng(seed + 11)
  end = date.today()
  start = end - timedelta(weeks=n_weeks)
  rows = []

  for _, store in stores.iterrows():
    condition = _condition_base(store["archetype"])
    n_images = max(3, int(images_per_store * rng.normal(1.0, 0.15)))

    for i in range(n_images):
      zone = rng.choice(IMAGE_ZONES)
      store_condition = round(min(10, max(1, condition + rng.normal(0, 1.2))), 1)

      # Shelf zones get utilization; exterior gets branding focus
      if zone in ("main_aisle", "produce_section", "endcap_display", "freezer_aisle"):
        shelf_util = round(min(98, max(40, rng.normal(78 if condition > 7 else 65, 10))), 1)
        branding = round(min(10, max(1, condition + rng.normal(0, 0.8))), 1)
      elif zone == "exterior":
        shelf_util = None
        branding = round(min(10, max(1, condition + rng.normal(0.5, 1.0))), 1)
      else:
        shelf_util = round(rng.normal(70, 12), 1) if zone != "checkout" else None
        branding = round(min(10, max(1, condition + rng.normal(0, 1.0))), 1)

      issues = []
      if store_condition < 6.5:
        issues.append(rng.choice(["dirty_floor", "damaged_signage", "poor_lighting"]))
      if shelf_util is not None and shelf_util < 60:
        issues.append("low_shelf_utilization")
      if branding is not None and branding < 6:
        issues.append("branding_non_compliance")

      rows.append({
        "image_id": f"IMG-{store['store_id']}-{i+1:03d}",
        "store_id": store["store_id"],
        "capture_date": _random_date_in_range(rng, start, end).isoformat(),
        "image_zone": zone,
        "file_path": f"images/{store['store_id']}/{zone}_{i+1:03d}.jpg",
        "store_condition_score": store_condition,
        "shelf_utilization_pct": shelf_util,
        "branding_compliance_score": branding,
        "detected_issues": "|".join(issues) if issues else "",
        "annotator": rng.choice(["cv_model_v2", "field_audit", "cv_model_v2"], p=[0.7, 0.2, 0.1]),
      })

  return pd.DataFrame(rows)


def generate_operational_reports(
  stores: pd.DataFrame,
  n_weeks: int,
  reports_per_store: int,
  seed: int,
) -> pd.DataFrame:
  """Field reports, audits, and visit notes flagging operational issues."""
  rng = np.random.default_rng(seed + 12)
  end = date.today()
  start = end - timedelta(weeks=n_weeks)
  archetypes = _archetype_map()
  rows = []

  issue_descriptions = {
    "staffing": "Insufficient coverage during peak hours — checkout and service desk",
    "cleaning": "Backroom and sales floor cleaning below standard — debris in aisles",
    "equipment": "Refrigeration unit temperature fluctuation logged in dairy section",
    "inventory": "Cycle count variance exceeds threshold in high-shrink categories",
    "safety": "Blocked emergency exit path noted during walkthrough",
    "signage": "Promotional signage expired or missing on endcap displays",
    "planogram": "Shelf tags misaligned with planogram — facings not compliant",
    "shrink": "Elevated shrink pattern detected near self-checkout area",
    "customer_service": "Wait times exceed SLA at service desk during weekend peak",
  }

  for _, store in stores.iterrows():
    arch = archetypes[store["archetype"]]
    n_reports = max(2, int(reports_per_store * rng.normal(1.0, 0.25)))

    for i in range(n_reports):
      issue_cat = rng.choice(ISSUE_CATEGORIES)
      severity = rng.choice(["low", "medium", "high"], p=[0.45, 0.35, 0.20])
      if arch["oos_rate"] > 0.06 and rng.random() < 0.3:
        issue_cat = "inventory"
      if arch["shrink_pct"] > 0.022 and rng.random() < 0.25:
        issue_cat = "shrink"

      rows.append({
        "report_id": f"RPT-{store['store_id']}-{i+1:03d}",
        "store_id": store["store_id"],
        "report_date": _random_date_in_range(rng, start, end).isoformat(),
        "report_type": rng.choice(REPORT_TYPES),
        "severity": severity,
        "issue_category": issue_cat,
        "description": issue_descriptions[issue_cat],
        "status": rng.choice(["open", "in_progress", "resolved"], p=[0.25, 0.35, 0.40]),
        "reported_by": rng.choice(["district_manager", "loss_prevention", "mystery_shopper", "store_manager"]),
      })

  return pd.DataFrame(rows)


def generate_local_news(stores: pd.DataFrame, n_weeks: int, seed: int) -> pd.DataFrame:
  """Local news events near each store that may affect foot traffic and demand."""
  rng = np.random.default_rng(seed + 13)
  end = date.today()
  start = end - timedelta(weeks=n_weeks)

  demand_impact_map = {
    "weather": {"heat wave": "positive", "storm": "negative", "winter storm": "mixed"},
    "local_festival": "positive",
    "road_closure": "negative",
    "new_competitor": "negative",
    "community_event": "positive",
    "school_holiday": "positive",
    "construction": "negative",
    "sports_event": "positive",
  }

  rows = []
  news_counter = 0

  for _, store in stores.iterrows():
    city = store["city"]
    # 2–6 news items per store over the period
    for _ in range(int(rng.integers(2, 7))):
      news_counter += 1
      event_type = rng.choice(NEWS_EVENT_TYPES)
      headline_tpl = rng.choice(NEWS_HEADLINES[event_type])
      headline = headline_tpl.format(city=city)

      impact = demand_impact_map.get(event_type, "neutral")
      if impact == "mixed":
        impact = rng.choice(["positive", "negative"])
      if isinstance(impact, dict):
        impact = rng.choice(list(impact.values()))

      rows.append({
        "news_id": f"NEWS-{news_counter:05d}",
        "store_id": store["store_id"],
        "city": city,
        "state": store["state"],
        "published_date": _random_date_in_range(rng, start, end).isoformat(),
        "headline": headline,
        "summary": f"Local coverage in {city}, {store['state']} — may affect store traffic patterns.",
        "event_type": event_type,
        "demand_impact": impact,
        "source": rng.choice(["local_tribune", "city_herald", "regional_news", "community_board"]),
      })

  return pd.DataFrame(rows)


def generate_product_descriptions(stores: pd.DataFrame, seed: int) -> pd.DataFrame:
  """
  Product descriptions tied to store assortment — category and attribute signals.
  Each store carries a subset of category templates based on format and archetype.
  """
  rng = np.random.default_rng(seed + 14)
  archetypes = _archetype_map()
  rows = []
  sku_counter = 0

  format_depth = {"Supercenter": 1.0, "Neighborhood": 0.75, "Express": 0.45}

  for _, store in stores.iterrows():
    arch = archetypes[store["archetype"]]
    depth = format_depth[store["store_format"]]

    for category, templates in PRODUCT_TEMPLATES.items():
      # Pick how many products from template list this store carries
      n_products = max(1, int(len(templates) * depth * rng.normal(1.0, 0.1)))
      chosen = rng.choice(len(templates), size=min(n_products, len(templates)), replace=False)

      for idx in chosen:
        sku_counter += 1
        name, description, attrs = templates[idx]

        # Archetype skews premium vs value assortment
        if "premium" in arch["name"] or "affluent" in arch["name"]:
          if "value" in attrs and rng.random() < 0.3:
            continue
        if "value" in arch["name"] or "rural" in arch["name"]:
          if "premium" in attrs and rng.random() < 0.35:
            continue

        rows.append({
          "sku_id": f"SKU-{sku_counter:06d}",
          "store_id": store["store_id"],
          "category": category,
          "product_name": name,
          "description": description,
          "attributes": attrs,
          "brand_type": "private_label" if "private_label" in attrs else rng.choice(
            ["national", "regional"], p=[0.75, 0.25]
          ),
          "in_stock": rng.random() > arch["oos_rate"],
          "price_tier": (
            "premium" if "premium" in attrs else
            "value" if "value" in attrs else "mid"
          ),
        })

  return pd.DataFrame(rows)


def write_metadata(output_dir: Path, config: dict, stores: pd.DataFrame) -> None:
  """Document dataset parameters for reproducibility."""
  meta = {
    **config,
    "generated_at": date.today().isoformat(),
    "tables": {
      "store_master": "One row per store — identity, format, location",
      "demographics": "Trade-area demographics per store",
      "assortment_snapshot": "Category SKU counts and facings (point-in-time)",
      "sales_weekly": "Store × category × week revenue and promo metrics",
      "operations_weekly": "Store × week labor, shrink, OOS, fulfillment",
      "customer_reviews": "Reviews with sentiment, rating, and complaint tags",
      "image_audits": "Simulated CV outputs — store condition, shelf use, branding",
      "operational_reports": "Audit and visit reports with operational issues",
      "local_news": "Local events and news affecting store demand",
      "product_descriptions": "SKU descriptions and assortment characteristics per store",
    },
    "archetypes": [a["name"] for a in ARCHETYPES],
    "categories": CATEGORIES,
    "store_count": len(stores),
    "notes": [
      "archetype column in store_master is for validation only — remove before modeling",
      "Metrics are correlated within archetype so peer matching benchmarks are realistic",
      "image_audits.file_path references synthetic paths — no binary image files are generated",
      "Unstructured tables (reviews, news, reports) align with archetype-driven store quality",
    ],
  }
  (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate synthetic StoreDNA datasets")
  parser.add_argument("--stores", type=int, default=40, help="Number of stores")
  parser.add_argument("--weeks", type=int, default=52, help="Weeks of history")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  parser.add_argument("--reviews-per-store", type=int, default=25, help="Avg customer reviews per store")
  parser.add_argument("--images-per-store", type=int, default=12, help="Avg image audits per store")
  parser.add_argument("--reports-per-store", type=int, default=8, help="Avg operational reports per store")
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parent.parent / "data" / "synthetic",
    help="Output directory",
  )
  args = parser.parse_args()

  args.output.mkdir(parents=True, exist_ok=True)

  print(f"Generating {args.stores} stores, {args.weeks} weeks → {args.output}")

  stores = generate_store_master(args.stores, args.seed)
  demographics = generate_demographics(stores, args.seed)
  assortment = generate_assortment(stores, args.seed)
  sales = generate_sales_weekly(stores, args.weeks, args.seed)
  operations = generate_operations_weekly(stores, args.weeks, args.seed)
  reviews = generate_customer_reviews(stores, args.weeks, args.reviews_per_store, args.seed)
  images = generate_image_audits(stores, args.weeks, args.images_per_store, args.seed)
  reports = generate_operational_reports(stores, args.weeks, args.reports_per_store, args.seed)
  news = generate_local_news(stores, args.weeks, args.seed)
  products = generate_product_descriptions(stores, args.seed)

  stores.to_csv(args.output / "store_master.csv", index=False)
  demographics.to_csv(args.output / "demographics.csv", index=False)
  assortment.to_csv(args.output / "assortment_snapshot.csv", index=False)
  sales.to_csv(args.output / "sales_weekly.csv", index=False)
  operations.to_csv(args.output / "operations_weekly.csv", index=False)
  reviews.to_csv(args.output / "customer_reviews.csv", index=False)
  images.to_csv(args.output / "image_audits.csv", index=False)
  reports.to_csv(args.output / "operational_reports.csv", index=False)
  news.to_csv(args.output / "local_news.csv", index=False)
  products.to_csv(args.output / "product_descriptions.csv", index=False)

  write_metadata(args.output, {
    "n_stores": args.stores,
    "n_weeks": args.weeks,
    "seed": args.seed,
  }, stores)

  print("Done.")
  print(f"  store_master.csv          {len(stores):>8,} rows")
  print(f"  demographics.csv          {len(demographics):>8,} rows")
  print(f"  assortment_snapshot.csv   {len(assortment):>8,} rows")
  print(f"  sales_weekly.csv          {len(sales):>8,} rows")
  print(f"  operations_weekly.csv     {len(operations):>8,} rows")
  print(f"  customer_reviews.csv      {len(reviews):>8,} rows")
  print(f"  image_audits.csv          {len(images):>8,} rows")
  print(f"  operational_reports.csv   {len(reports):>8,} rows")
  print(f"  local_news.csv            {len(news):>8,} rows")
  print(f"  product_descriptions.csv  {len(products):>8,} rows")


if __name__ == "__main__":
  main()

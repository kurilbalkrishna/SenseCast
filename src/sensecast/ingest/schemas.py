"""Data contracts for every feed (pandera).

These schemas ARE the data contract: ingestion rejects any row that breaks them
into data/quarantine/ with the failing check, instead of silently fixing it.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

CATEGORIES = ["beverages", "snacks", "personal_care", "monsoon_seasonal"]


def sales_schema(max_date) -> DataFrameSchema:
    return DataFrameSchema(
        {
            "date": Column("datetime64[ns]", Check.le(max_date, error="date_in_future")),
            "store_id": Column(str, Check.str_matches(r"^S\d{2}$")),
            "item_id": Column(str, Check.str_matches(r"^I\d{3}$")),
            "units_sold": Column("int64", Check.ge(0, error="negative_units")),
            "unit_price": Column(float, Check.gt(0), nullable=False),
            "on_hand_open": Column("int64", Check.ge(0)),
            "receipts": Column("int64", Check.ge(0)),
            "on_hand_close": Column("int64", Check.ge(0)),
        },
        checks=[
            Check(
                lambda df: df.on_hand_open + df.receipts - df.units_sold == df.on_hand_close,
                element_wise=False,
                error="stock_balance",
            )
        ],
        coerce=True,
        strict=True,
    )


ITEMS = DataFrameSchema(
    {
        "item_id": Column(str, unique=True),
        "category": Column(str, Check.isin(CATEGORIES)),
        "price": Column(float, Check.gt(0)),
        "pack_size": Column(int, Check.gt(0)),
        "launch_date": Column("datetime64[ns]"),
    },
    coerce=True,
    strict=True,
)

STORES = DataFrameSchema(
    {"store_id": Column(str, unique=True), "city": Column(str), "region": Column(str)},
    coerce=True,
    strict=True,
)

WEATHER = DataFrameSchema(
    {
        "date": Column("datetime64[ns]"),
        "city": Column(str),
        "temp_max": Column(float, Check.in_range(-5, 50), nullable=True),
        "rain_mm": Column(float, Check.in_range(0, 1000), nullable=True),
        "humidity": Column(float, Check.in_range(0, 100), nullable=True),
    },
    unique=["date", "city"],
    coerce=True,
)

SEARCH = DataFrameSchema(
    {
        "date": Column("datetime64[ns]"),
        "category": Column(str, Check.isin(CATEGORIES)),
        "search_index": Column(float, Check.ge(0), nullable=True),
    },
    unique=["date", "category"],
    coerce=True,
    strict=True,
)

PROMOTIONS = DataFrameSchema(
    {
        "item_id": Column(str),
        "region": Column(str),
        "date_start": Column("datetime64[ns]"),
        "date_end": Column("datetime64[ns]"),
        "discount": Column(float, Check.in_range(0, 0.9)),
    },
    checks=[Check(lambda df: df.date_end >= df.date_start, element_wise=False, error="promo_dates")],
    coerce=True,
    strict=True,
)

EVENTS = DataFrameSchema(
    {
        "date": Column("datetime64[ns]"),
        "event": Column(str),
        "kind": Column(str, Check.isin(["festival", "public_holiday"])),
        "source": Column(str),
        "span_days": Column(int, Check.ge(1)),
        "buildup_days": Column(int, Check.ge(0)),
    },
    coerce=True,
    strict=True,
)

__all__ = ["sales_schema", "ITEMS", "STORES", "WEATHER", "SEARCH", "PROMOTIONS", "EVENTS", "pa"]

"""Add immutable market dataset manifests and canonical candles."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261002_03"
down_revision: str | None = "20261002_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_instruments",
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("canonical_symbol", sa.String(32), nullable=False),
        sa.Column("asset_class", sa.String(24), nullable=False),
        sa.Column("venue", sa.String(64), nullable=True),
        sa.Column("market", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("instrument_id"),
        sa.UniqueConstraint(
            "canonical_symbol", "asset_class", "venue", "market",
            name="uq_market_instrument_identity",
        ),
        sa.CheckConstraint(
            "asset_class IN ('equity', 'fx', 'crypto', 'future', 'option', 'index', 'other')",
            name="ck_market_instrument_asset_class",
        ),
    )
    op.create_table(
        "market_datasets",
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("timeframe", sa.String(3), nullable=False),
        sa.Column("provider_id", sa.String(64), nullable=False),
        sa.Column("provider_version", sa.String(128), nullable=False),
        sa.Column("provider_symbol", sa.String(128), nullable=False),
        sa.Column("requested_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("requested_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timestamp_convention", sa.String(16), nullable=False),
        sa.Column("price_basis", sa.String(24), nullable=False),
        sa.Column("volume_units", sa.String(64), nullable=False),
        sa.Column("normalization_version", sa.String(32), nullable=False),
        sa.Column("validation_version", sa.String(32), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("quality_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provider_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("dataset_id", "dataset_version"),
        sa.ForeignKeyConstraint(
            ["instrument_id"], ["market_instruments.instrument_id"],
            name="fk_market_dataset_instrument",
        ),
        sa.UniqueConstraint(
            "dataset_id", "dataset_version", "instrument_id", "timeframe",
            name="uq_market_dataset_identity_instrument",
        ),
        sa.CheckConstraint("dataset_version > 0", name="ck_market_dataset_version_positive"),
        sa.CheckConstraint("requested_start < requested_end", name="ck_market_dataset_range"),
        sa.CheckConstraint("length(content_hash) = 64", name="ck_market_dataset_hash_length"),
        sa.CheckConstraint(
            "timeframe IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')",
            name="ck_market_dataset_timeframe",
        ),
    )
    op.create_index(
        "ix_market_datasets_instrument_timeframe",
        "market_datasets", ["instrument_id", "timeframe"],
    )
    op.create_table(
        "market_candles",
        sa.Column("dataset_id", sa.Uuid(), nullable=False),
        sa.Column("dataset_version", sa.Integer(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("timeframe", sa.String(3), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(), nullable=False),
        sa.Column("high", sa.Numeric(), nullable=False),
        sa.Column("low", sa.Numeric(), nullable=False),
        sa.Column("close", sa.Numeric(), nullable=False),
        sa.Column("volume", sa.Numeric(), nullable=False),
        sa.Column("bid", sa.Numeric(), nullable=True),
        sa.Column("ask", sa.Numeric(), nullable=True),
        sa.Column("spread", sa.Numeric(), nullable=True),
        sa.PrimaryKeyConstraint(
            "dataset_id", "dataset_version", "instrument_id", "timeframe", "timestamp",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id", "dataset_version", "instrument_id", "timeframe"],
            ["market_datasets.dataset_id", "market_datasets.dataset_version", "market_datasets.instrument_id", "market_datasets.timeframe"],
            name="fk_market_candle_dataset_instrument",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["instrument_id"], ["market_instruments.instrument_id"],
            name="fk_market_candle_instrument",
        ),
        sa.CheckConstraint("high >= low", name="ck_market_candle_high_low"),
        sa.CheckConstraint("high >= open AND high >= close", name="ck_market_candle_high_ohlc"),
        sa.CheckConstraint("low <= open AND low <= close", name="ck_market_candle_low_ohlc"),
        sa.CheckConstraint(
            "open > '-Infinity'::numeric AND open < 'Infinity'::numeric AND "
            "high > '-Infinity'::numeric AND high < 'Infinity'::numeric AND "
            "low > '-Infinity'::numeric AND low < 'Infinity'::numeric AND "
            "close > '-Infinity'::numeric AND close < 'Infinity'::numeric",
            name="ck_market_candle_ohlc_finite",
        ),
        sa.CheckConstraint(
            "volume >= 0 AND volume > '-Infinity'::numeric AND volume < 'Infinity'::numeric",
            name="ck_market_candle_volume_finite_nonnegative",
        ),
        sa.CheckConstraint("spread IS NULL OR spread >= 0", name="ck_market_candle_spread_nonnegative"),
        sa.CheckConstraint(
            "(bid IS NULL OR (bid > '-Infinity'::numeric AND bid < 'Infinity'::numeric)) AND "
            "(ask IS NULL OR (ask > '-Infinity'::numeric AND ask < 'Infinity'::numeric)) AND "
            "(spread IS NULL OR (spread > '-Infinity'::numeric AND spread < 'Infinity'::numeric))",
            name="ck_market_candle_quotes_finite",
        ),
        sa.CheckConstraint("bid IS NULL OR ask IS NULL OR bid <= ask", name="ck_market_candle_bid_ask"),
        sa.CheckConstraint(
            "bid IS NULL OR ask IS NULL OR spread IS NULL OR spread = ask - bid",
            name="ck_market_candle_spread_matches_quotes",
        ),
        sa.CheckConstraint(
            "timeframe IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')",
            name="ck_market_candle_timeframe",
        ),
    )
    op.create_index(
        "ix_market_candles_instrument_timeframe_timestamp",
        "market_candles", ["instrument_id", "timeframe", "timestamp"],
    )
    op.create_index(
        "ix_market_candles_dataset_time",
        "market_candles", ["dataset_id", "dataset_version", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_market_candles_dataset_time", table_name="market_candles")
    op.drop_index("ix_market_candles_instrument_timeframe_timestamp", table_name="market_candles")
    op.drop_table("market_candles")
    op.drop_index("ix_market_datasets_instrument_timeframe", table_name="market_datasets")
    op.drop_table("market_datasets")
    op.drop_table("market_instruments")

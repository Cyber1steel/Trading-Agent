"""PostgreSQL persistence for immutable, provenance-rich market snapshots."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class MarketInstrument(Base):
    __tablename__ = "market_instruments"
    __table_args__ = (
        UniqueConstraint(
            "canonical_symbol", "asset_class", "venue", "market",
            name="uq_market_instrument_identity",
        ),
        CheckConstraint(
            "asset_class IN ('equity', 'fx', 'crypto', 'future', 'option', 'index', 'other')",
            name="ck_market_instrument_asset_class",
        ),
    )

    instrument_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    canonical_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(24), nullable=False)
    venue: Mapped[str | None] = mapped_column(String(64))
    market: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MarketDataset(Base):
    __tablename__ = "market_datasets"
    __table_args__ = (
        ForeignKeyConstraint(
            ["instrument_id"], ["market_instruments.instrument_id"],
            name="fk_market_dataset_instrument",
        ),
        UniqueConstraint(
            "dataset_id", "dataset_version", "instrument_id", "timeframe",
            name="uq_market_dataset_identity_instrument",
        ),
        CheckConstraint("dataset_version > 0", name="ck_market_dataset_version_positive"),
        CheckConstraint("requested_start < requested_end", name="ck_market_dataset_range"),
        CheckConstraint("length(content_hash) = 64", name="ck_market_dataset_hash_length"),
        CheckConstraint(
            "timeframe IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')",
            name="ck_market_dataset_timeframe",
        ),
        Index("ix_market_datasets_instrument_timeframe", "instrument_id", "timeframe"),
    )

    dataset_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    dataset_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(3), nullable=False)
    provider_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_version: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    requested_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_convention: Mapped[str] = mapped_column(String(16), nullable=False)
    price_basis: Mapped[str] = mapped_column(String(24), nullable=False)
    volume_units: Mapped[str] = mapped_column(String(64), nullable=False)
    normalization_version: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_version: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_summary: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    provider_metadata: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MarketCandle(Base):
    __tablename__ = "market_candles"
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_id", "dataset_version", "instrument_id", "timeframe"],
            ["market_datasets.dataset_id", "market_datasets.dataset_version", "market_datasets.instrument_id", "market_datasets.timeframe"],
            name="fk_market_candle_dataset_instrument",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["instrument_id"], ["market_instruments.instrument_id"],
            name="fk_market_candle_instrument",
        ),
        CheckConstraint("high >= low", name="ck_market_candle_high_low"),
        CheckConstraint("high >= open AND high >= close", name="ck_market_candle_high_ohlc"),
        CheckConstraint("low <= open AND low <= close", name="ck_market_candle_low_ohlc"),
        CheckConstraint(
            "open > '-Infinity'::numeric AND open < 'Infinity'::numeric AND "
            "high > '-Infinity'::numeric AND high < 'Infinity'::numeric AND "
            "low > '-Infinity'::numeric AND low < 'Infinity'::numeric AND "
            "close > '-Infinity'::numeric AND close < 'Infinity'::numeric",
            name="ck_market_candle_ohlc_finite",
        ),
        CheckConstraint(
            "volume >= 0 AND volume > '-Infinity'::numeric AND volume < 'Infinity'::numeric",
            name="ck_market_candle_volume_finite_nonnegative",
        ),
        CheckConstraint("spread IS NULL OR spread >= 0", name="ck_market_candle_spread_nonnegative"),
        CheckConstraint(
            "(bid IS NULL OR (bid > '-Infinity'::numeric AND bid < 'Infinity'::numeric)) AND "
            "(ask IS NULL OR (ask > '-Infinity'::numeric AND ask < 'Infinity'::numeric)) AND "
            "(spread IS NULL OR (spread > '-Infinity'::numeric AND spread < 'Infinity'::numeric))",
            name="ck_market_candle_quotes_finite",
        ),
        CheckConstraint("bid IS NULL OR ask IS NULL OR bid <= ask", name="ck_market_candle_bid_ask"),
        CheckConstraint(
            "bid IS NULL OR ask IS NULL OR spread IS NULL OR spread = ask - bid",
            name="ck_market_candle_spread_matches_quotes",
        ),
        CheckConstraint(
            "timeframe IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')",
            name="ck_market_candle_timeframe",
        ),
        Index(
            "ix_market_candles_instrument_timeframe_timestamp",
            "instrument_id", "timeframe", "timestamp",
        ),
        Index("ix_market_candles_dataset_time", "dataset_id", "dataset_version", "timestamp"),
    )

    dataset_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    dataset_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(3), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    bid: Mapped[Decimal | None] = mapped_column(Numeric)
    ask: Mapped[Decimal | None] = mapped_column(Numeric)
    spread: Mapped[Decimal | None] = mapped_column(Numeric)

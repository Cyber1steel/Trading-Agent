"""Real PostgreSQL persistence checks; requires a migrated Phase 2D test database."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.market_data.contracts import (
    AssetClass,
    Candle,
    Instrument,
    MarketDatasetManifest,
    PriceBasis,
    TimestampConvention,
    normalized_content_hash,
)
from app.market_data.errors import DatasetAlreadyExists
from app.market_data.quality import DataQualityReport
from app.market_data.repository import MarketDataRepository
from app.market_data.timeframes import Timeframe
from app.models.market_data import MarketCandle, MarketDataset


DATABASE_URL = os.getenv("MARKET_DATA_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="requires a migrated PostgreSQL + pgvector database with Phase 2D schema",
)


def snapshot(instrument: Instrument, dataset_id, candles):
    now = datetime.now(timezone.utc)
    return MarketDatasetManifest(
        dataset_id=dataset_id,
        dataset_version=1,
        instrument=instrument,
        timeframe=Timeframe.M1,
        provider_id="integration-fixture",
        provider_version="1",
        provider_symbol=instrument.symbol,
        requested_start=candles[0].timestamp,
        requested_end=candles[-1].timestamp + timedelta(minutes=1),
        retrieved_at=now,
        ingested_at=now,
        timestamp_convention=TimestampConvention.BAR_OPEN,
        price_basis=PriceBasis.TRADE,
        volume_units="contracts",
        normalization_version="1",
        validation_version="1",
        content_hash=normalized_content_hash(tuple(candles)),
        quality=DataQualityReport(
            input_record_count=len(candles), accepted_record_count=len(candles)
        ),
        provider_metadata={"kind": "integration_fixture"},
    )


def test_postgres_snapshot_persistence_range_provenance_and_immutability():
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, expire_on_commit=False)
    repository = MarketDataRepository(factory)
    instrument = Instrument(
        symbol=f"T{uuid4().hex[:10]}", asset_class=AssetClass.OTHER, venue="TEST"
    )
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = tuple(
        Candle(
            instrument=instrument,
            timeframe=Timeframe.M1,
            timestamp=base + timedelta(minutes=index),
            open=Decimal("-1.00"), high=Decimal("0.50"), low=Decimal("-2.00"),
            close=Decimal("0.00"), volume=Decimal(index),
        )
        for index in (0, 1)
    )
    dataset_id = uuid4()
    manifest = snapshot(instrument, dataset_id, candles)
    try:
        repository.create_snapshot(manifest, candles)
        selected = repository.get_range(
            dataset_id=dataset_id,
            dataset_version=1,
            instrument=instrument,
            timeframe=Timeframe.M1,
            start=base,
            end=base + timedelta(minutes=2),
        )
        assert selected.manifest.provider_id == "integration-fixture"
        assert selected.manifest.content_hash == manifest.content_hash
        assert [candle.timestamp for candle in selected.candles] == [base, base + timedelta(minutes=1)]
        assert [candle.open for candle in selected.candles] == [Decimal("-1.00")] * 2
        with pytest.raises(DatasetAlreadyExists):
            repository.create_snapshot(manifest, candles)
    finally:
        with factory() as session, session.begin():
            session.execute(delete(MarketDataset).where(MarketDataset.dataset_id == dataset_id))
        engine.dispose()


def test_postgres_candle_constraints_reject_invalid_ohlc_volume_and_duplicates():
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, expire_on_commit=False)
    repository = MarketDataRepository(factory)
    instrument = Instrument(
        symbol=f"T{uuid4().hex[:10]}", asset_class=AssetClass.OTHER, venue="TEST"
    )
    timestamp = datetime(2024, 2, 1, tzinfo=timezone.utc)
    valid = Candle(
        instrument=instrument, timeframe=Timeframe.M1, timestamp=timestamp,
        open=Decimal("1"), high=Decimal("2"), low=Decimal("0"), close=Decimal("1"),
        volume=Decimal("1"),
    )
    dataset_id = uuid4()
    manifest = snapshot(instrument, dataset_id, (valid,))
    try:
        repository.create_snapshot(manifest, (valid,))
        with pytest.raises(IntegrityError):
            with factory() as session, session.begin():
                session.add(MarketCandle(
                    dataset_id=dataset_id,
                    dataset_version=1,
                    instrument_id=instrument.instrument_id,
                    timeframe="1m",
                    timestamp=timestamp + timedelta(minutes=1),
                    open=Decimal("3"), high=Decimal("2"), low=Decimal("1"), close=Decimal("2"),
                    volume=Decimal("1"),
                ))
        with pytest.raises(IntegrityError):
            with factory() as session, session.begin():
                session.add(MarketCandle(
                    dataset_id=dataset_id,
                    dataset_version=1,
                    instrument_id=instrument.instrument_id,
                    timeframe="1m",
                    timestamp=timestamp + timedelta(minutes=2),
                    open=valid.open, high=valid.high, low=valid.low, close=valid.close,
                    volume=Decimal("-1"),
                ))
        with pytest.raises(IntegrityError):
            with factory() as session, session.begin():
                session.add(MarketCandle(
                    dataset_id=dataset_id,
                    dataset_version=1,
                    instrument_id=instrument.instrument_id,
                    timeframe="1m",
                    timestamp=timestamp,
                    open=valid.open, high=valid.high, low=valid.low, close=valid.close,
                    volume=valid.volume,
                ))
    finally:
        with factory() as session, session.begin():
            session.execute(delete(MarketDataset).where(MarketDataset.dataset_id == dataset_id))
        engine.dispose()

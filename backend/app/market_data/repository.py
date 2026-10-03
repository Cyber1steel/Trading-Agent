"""Explicit dataset-scoped persistence and range queries for canonical candles."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.market_data.contracts import (
    Candle,
    Instrument,
    MarketDataSlice,
    MarketDatasetManifest,
    normalized_content_hash,
)
from app.market_data.errors import DatasetAlreadyExists, DatasetNotFound, InstrumentIdentityConflict
from app.market_data.quality import DataQualityReport
from app.market_data.timeframes import Timeframe
from app.models.market_data import MarketCandle, MarketDataset, MarketInstrument


class MarketDataRepository:
    def __init__(self, session_factory=SessionLocal):
        self._session_factory = session_factory

    def create_snapshot(
        self,
        manifest: MarketDatasetManifest,
        candles: tuple[Candle, ...],
    ) -> None:
        instrument = manifest.instrument
        instrument_id = instrument.instrument_id
        if not manifest.quality.can_persist:
            raise ValueError("Datasets with hard quality failures cannot be persisted")
        if manifest.timestamp_convention.value != "bar_open":
            raise ValueError("Only UTC BAR-OPEN datasets can be persisted")
        if manifest.quality.accepted_record_count != len(candles):
            raise ValueError("Manifest accepted-record count does not match candle count")
        if normalized_content_hash(candles) != manifest.content_hash:
            raise ValueError("Manifest content hash does not match canonical candles")
        prior_timestamp = None
        for candle in candles:
            if candle.instrument != instrument or candle.timeframe != manifest.timeframe:
                raise ValueError("All snapshot candles must match manifest instrument and timeframe")
            if not manifest.requested_start <= candle.timestamp < manifest.requested_end:
                raise ValueError("Snapshot candle falls outside manifest bounds")
            if prior_timestamp is not None and candle.timestamp <= prior_timestamp:
                raise ValueError("Snapshot candles must be strictly timestamp ordered")
            prior_timestamp = candle.timestamp
        try:
            with self._session_factory() as session, session.begin():
                existing_instrument = session.get(MarketInstrument, instrument_id)
                if existing_instrument is not None and (
                    existing_instrument.canonical_symbol != instrument.symbol
                    or existing_instrument.asset_class != instrument.asset_class.value
                    or existing_instrument.venue != instrument.venue
                    or existing_instrument.market != instrument.market
                ):
                    raise InstrumentIdentityConflict(
                        f"Stored instrument {instrument_id} conflicts with its canonical identity."
                    )
                if existing_instrument is None:
                    session.add(MarketInstrument(
                        instrument_id=instrument_id,
                        canonical_symbol=instrument.symbol,
                        asset_class=instrument.asset_class.value,
                        venue=instrument.venue,
                        market=instrument.market,
                    ))
                if session.get(MarketDataset, (manifest.dataset_id, manifest.dataset_version)) is not None:
                    raise DatasetAlreadyExists(
                        f"Dataset {manifest.dataset_id} version {manifest.dataset_version} is immutable and already exists."
                    )
                session.add(MarketDataset(
                    dataset_id=manifest.dataset_id,
                    dataset_version=manifest.dataset_version,
                    instrument_id=instrument_id,
                    timeframe=manifest.timeframe.value,
                    provider_id=manifest.provider_id,
                    provider_version=manifest.provider_version,
                    provider_symbol=manifest.provider_symbol,
                    requested_start=manifest.requested_start,
                    requested_end=manifest.requested_end,
                    retrieved_at=manifest.retrieved_at,
                    ingested_at=manifest.ingested_at,
                    timestamp_convention=manifest.timestamp_convention.value,
                    price_basis=manifest.price_basis.value,
                    volume_units=manifest.volume_units,
                    normalization_version=manifest.normalization_version,
                    validation_version=manifest.validation_version,
                    content_hash=manifest.content_hash,
                    quality_summary=manifest.quality.model_dump(mode="json"),
                    provider_metadata=manifest.provider_metadata,
                ))
                session.add_all([
                    MarketCandle(
                        dataset_id=manifest.dataset_id,
                        dataset_version=manifest.dataset_version,
                        instrument_id=instrument_id,
                        timeframe=candle.timeframe.value,
                        timestamp=candle.timestamp,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                        bid=candle.bid,
                        ask=candle.ask,
                        spread=candle.spread,
                    )
                    for candle in candles
                ])
        except IntegrityError as exc:
            raise DatasetAlreadyExists(
                f"Snapshot {manifest.dataset_id} version {manifest.dataset_version} violated an immutable uniqueness constraint."
            ) from exc

    def get_range(
        self,
        *,
        dataset_id: UUID,
        dataset_version: int,
        instrument: Instrument,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> MarketDataSlice:
        timeframe = Timeframe.parse(timeframe)
        if start.tzinfo is None or start.utcoffset() is None or end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("Range bounds must be timezone-aware")
        if start >= end:
            raise ValueError("Range must satisfy start < end")
        with self._session_factory() as session:  # type: Session
            dataset = session.get(MarketDataset, (dataset_id, dataset_version))
            if dataset is None or dataset.instrument_id != instrument.instrument_id:
                raise DatasetNotFound(f"Dataset {dataset_id} version {dataset_version} was not found for this instrument.")
            if dataset.timeframe != timeframe.value:
                raise DatasetNotFound("The selected dataset does not contain the requested timeframe.")
            start_utc = start.astimezone(timezone.utc)
            end_utc = end.astimezone(timezone.utc)
            if start_utc < dataset.requested_start or end_utc > dataset.requested_end:
                raise ValueError("Range must remain within the selected dataset's requested bounds")
            rows = session.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.dataset_id == dataset_id,
                    MarketCandle.dataset_version == dataset_version,
                    MarketCandle.instrument_id == instrument.instrument_id,
                    MarketCandle.timeframe == timeframe.value,
                    MarketCandle.timestamp >= start_utc,
                    MarketCandle.timestamp < end_utc,
                )
                .order_by(MarketCandle.timestamp.asc())
            ).scalars().all()
            manifest = MarketDatasetManifest(
                dataset_id=dataset.dataset_id,
                dataset_version=dataset.dataset_version,
                instrument=instrument,
                timeframe=dataset.timeframe,
                provider_id=dataset.provider_id,
                provider_version=dataset.provider_version,
                provider_symbol=dataset.provider_symbol,
                requested_start=dataset.requested_start,
                requested_end=dataset.requested_end,
                retrieved_at=dataset.retrieved_at,
                ingested_at=dataset.ingested_at,
                timestamp_convention=dataset.timestamp_convention,
                price_basis=dataset.price_basis,
                volume_units=dataset.volume_units,
                normalization_version=dataset.normalization_version,
                validation_version=dataset.validation_version,
                content_hash=dataset.content_hash,
                quality=DataQualityReport.model_validate(dataset.quality_summary),
                provider_metadata=dataset.provider_metadata,
            )
            candles = tuple(Candle(
                instrument=instrument,
                timeframe=row.timeframe,
                timestamp=row.timestamp,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                bid=row.bid,
                ask=row.ask,
                spread=row.spread,
            ) for row in rows)
            return MarketDataSlice(manifest=manifest, candles=candles)

"""Provider-to-canonical orchestration and immutable snapshot creation."""

from datetime import datetime, timezone
from uuid import uuid4

from app.market_data.contracts import (
    MarketDataRequest,
    MarketDatasetManifest,
    ProviderSymbolMapping,
)
from app.market_data.errors import MarketDataValidationError
from app.market_data.normalization import (
    NORMALIZATION_VERSION,
    VALIDATION_VERSION,
    normalize_batch,
)
from app.market_data.providers.base import HistoricalMarketDataProvider
from app.market_data.quality_policy import CompletenessPolicy
from app.market_data.repository import MarketDataRepository


class MarketDataService:
    def __init__(self, provider: HistoricalMarketDataProvider, repository: MarketDataRepository):
        self.provider = provider
        self.repository = repository

    def ingest_historical(
        self,
        request: MarketDataRequest,
        *,
        symbol_mapping: ProviderSymbolMapping,
        completeness_policy: CompletenessPolicy | None = None,
    ) -> MarketDatasetManifest:
        batch = self.provider.fetch_historical(request)
        normalized = normalize_batch(
            batch,
            request,
            symbol_mapping,
            completeness_policy=completeness_policy,
        )
        if not normalized.quality.can_persist or normalized.content_hash is None:
            raise MarketDataValidationError(normalized.quality)
        manifest = MarketDatasetManifest(
            dataset_id=uuid4(),
            dataset_version=1,
            instrument=request.instrument,
            timeframe=request.timeframe,
            provider_id=batch.provider_id,
            provider_version=batch.provider_version,
            provider_symbol=batch.provider_symbol,
            requested_start=request.start,
            requested_end=request.end,
            retrieved_at=batch.retrieved_at,
            ingested_at=datetime.now(timezone.utc),
            timestamp_convention=batch.timestamp_convention,
            price_basis=batch.price_basis,
            volume_units=batch.volume_units,
            normalization_version=NORMALIZATION_VERSION,
            validation_version=VALIDATION_VERSION,
            content_hash=normalized.content_hash,
            quality=normalized.quality,
            provider_metadata=batch.provider_metadata,
        )
        self.repository.create_snapshot(manifest, normalized.candles)
        return manifest

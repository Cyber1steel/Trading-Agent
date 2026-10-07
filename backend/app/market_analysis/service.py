"""Read-only deterministic analysis over explicitly selected Phase 2D snapshots."""

from app.market_analysis.contracts import (
    ANALYSIS_VERSION, AnalysisRequest, AnalysisResult, DatasetProvenance,
    MarketObservation, SelectedInput, SessionTransition,
    normalized_hash,
)
from app.market_analysis.errors import AnalysisCutoffError, AnalysisDataError
from app.market_analysis.features import candle_features
from app.market_analysis.multi_timeframe import align_parent_context
from app.market_analysis.structure import classify_swings, structure_at_times
from app.market_analysis.swings import detect_swings
from app.market_analysis.volatility import average_true_ranges, true_ranges
from app.market_context.sessions import SessionClassifier, generic_session_definitions


class MarketAnalysisService:
    def __init__(self, repository, session_classifier=None):
        self.repository = repository
        self.session_classifier = session_classifier or SessionClassifier(
            generic_session_definitions(), calendar_timezone="UTC"
        )

    def _load(self, ref, request):
        selected = self.repository.get_range(
            dataset_id=ref.dataset_id, dataset_version=ref.dataset_version,
            instrument=ref.instrument, timeframe=ref.timeframe,
            start=request.input_start, end=request.analysis_end,
        )
        manifest = selected.manifest
        if (manifest.dataset_id != ref.dataset_id or manifest.dataset_version != ref.dataset_version
            or manifest.instrument != ref.instrument or manifest.timeframe != ref.timeframe):
            raise AnalysisDataError("Returned manifest identity does not match explicit DatasetRef")
        if manifest.timestamp_convention.value != "bar_open":
            raise AnalysisDataError("Phase 2E requires BAR-OPEN dataset timestamps")
        if not manifest.quality.can_persist:
            raise AnalysisDataError("Selected dataset carries hard quality failures")
        for candle in selected.candles:
            if candle.instrument != ref.instrument or candle.timeframe != ref.timeframe:
                raise AnalysisDataError("Candle identity does not match explicit DatasetRef")
            if not request.input_start <= candle.timestamp < request.analysis_end:
                raise AnalysisDataError("Repository returned candle outside the requested half-open range")
        if any(a.timestamp >= b.timestamp for a, b in zip(selected.candles, selected.candles[1:])):
            raise AnalysisDataError("Selected candles must be strictly timestamp ordered")
        return selected

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        refs = (request.primary, *request.higher_timeframes)
        slices = tuple(self._load(ref, request) for ref in refs)
        primary_manifest = slices[0].manifest
        for part in slices[1:]:
            manifest = part.manifest
            if (manifest.provider_id != primary_manifest.provider_id
                or manifest.provider_version != primary_manifest.provider_version
                or manifest.price_basis != primary_manifest.price_basis
                or manifest.timestamp_convention != primary_manifest.timestamp_convention
                or manifest.instrument != primary_manifest.instrument):
                raise AnalysisDataError("Higher-timeframe provenance must match the primary dataset")

        base = slices[0]
        duration = request.primary.timeframe.nominal_duration
        requested_output = tuple(c for c in base.candles
                                 if request.analysis_start <= c.timestamp < request.analysis_end)
        if any(c.timestamp + duration > request.cutoff_at for c in requested_output):
            raise AnalysisCutoffError("Requested primary observation is not closed by cutoff_at")
        # Bars not closed by the declared cutoff cannot participate in any output.
        candles = tuple(c for c in base.candles if c.timestamp + duration <= request.cutoff_at)
        if not candles:
            raise AnalysisDataError("Primary dataset has no candles in the explicit input range")
        output = tuple(c for c in candles if request.analysis_start <= c.timestamp < request.analysis_end)
        if not output:
            raise AnalysisDataError("No primary candles fall in the requested analysis range")

        ranges = true_ranges(candles)
        atrs = average_true_ranges(ranges, request.parameters.atr_period)
        raw_swings = detect_swings(candles, request.parameters.swing_left, request.parameters.swing_right)
        observation_times = tuple(c.timestamp + duration for c in output)
        structural_snapshots = structure_at_times(raw_swings, observation_times)

        parent_data = []
        effective_candles = [candles]
        for part in slices[1:]:
            parent_duration = part.manifest.timeframe.nominal_duration
            pcandles = tuple(c for c in part.candles if c.timestamp + parent_duration <= request.cutoff_at)
            swings = detect_swings(pcandles, request.parameters.swing_left, request.parameters.swing_right)
            tagged, _, _ = classify_swings(swings)
            parent_times = tuple(parent.timestamp + parent.timeframe.nominal_duration for parent in pcandles)
            snapshots = structure_at_times(swings, parent_times)
            states = tuple(snapshot[0] for snapshot in snapshots)
            parent_data.append((part, pcandles, tagged, states))
            effective_candles.append(pcandles)

        parent_aligned = []
        for part, pcandles, swings, states in parent_data:
            parent_aligned.append(align_parent_context(
                observation_times, pcandles, states, swings, part.manifest.timeframe
            ))

        observations = []
        # Feature returns are based on the previous selected input candle, including warm-up.
        feature_by_time = {}
        for i, candle in enumerate(candles):
            feature_by_time[candle.timestamp] = candle_features(candle, candles[i - 1].close if i else None)
        prior_session_labels = None
        index_by_timestamp = {item.timestamp: i for i, item in enumerate(candles)}
        for out_index, candle in enumerate(output):
            known_at = candle.timestamp + duration
            state, eligible, transitions = structural_snapshots[out_index]
            transition = transitions[-1] if transitions and transitions[-1].known_at == known_at else None
            input_index = index_by_timestamp[candle.timestamp]
            session = self.session_classifier.classify(known_at)
            session_transition = None
            if prior_session_labels is not None:
                before, after = set(prior_session_labels), set(session.labels)
                session_transition = SessionTransition(
                    entered=tuple(sorted(after - before)), exited=tuple(sorted(before - after))
                )
            prior_session_labels = session.labels
            observations.append(MarketObservation(
                bar_open=candle.timestamp, known_at=known_at,
                open=candle.open, high=candle.high, low=candle.low, close=candle.close,
                volume=candle.volume, bid=candle.bid, ask=candle.ask, spread=candle.spread,
                volume_units=base.manifest.volume_units,
                features=feature_by_time[candle.timestamp], true_range=ranges[input_index],
                atr=atrs[input_index], structural_state=state,
                confirmed_swings=eligible,
                structural_transition=transition,
                higher_timeframes=tuple(items[out_index] for items in parent_aligned), session=session,
                session_transition=session_transition,
            ))

        selected_inputs = tuple(SelectedInput(
            provenance=DatasetProvenance.from_manifest(part.manifest),
            slice_hash=normalized_hash([c.model_dump(mode="json") for c in chosen]),
        ) for part, chosen in zip(slices, effective_candles))
        fields = dict(
            analysis_version=ANALYSIS_VERSION, parameters=request.parameters,
            input_start=request.input_start, analysis_start=request.analysis_start,
            analysis_end=request.analysis_end, cutoff_at=request.cutoff_at,
            inputs=selected_inputs, observations=tuple(observations),
        )
        fingerprint = normalized_hash({k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v)
                                       for k, v in fields.items()})
        return AnalysisResult(**fields, fingerprint=fingerprint)

"""Causal structural labels from already-confirmed swings."""

from app.market_analysis.contracts import (
    ConfirmedSwing, StructuralState, StructuralTransition, SwingComparison, SwingKind,
)


def classify_swings(events: tuple[ConfirmedSwing, ...]):
    tracker = _StructureTracker()
    tagged, transitions = [], []
    for event in events:
        item, transition = tracker.push(event)
        tagged.append(item)
        if transition is not None:
            transitions.append(transition)
    return tuple(tagged), tuple(transitions), tracker.state


class _StructureTracker:
    """O(1)-per-event structure state used by both batch and time-aligned analysis."""

    def __init__(self):
        self.highs = []
        self.lows = []
        self.state = StructuralState.INSUFFICIENT
        self.established = None

    def push(self, event):
        points = self.highs if event.kind == SwingKind.HIGH else self.lows
        prior = points[-1] if points else None
        if prior is None:
            comparison = None
        elif event.kind == SwingKind.HIGH:
            comparison = (SwingComparison.HIGHER_HIGH if event.price > prior.price else
                          SwingComparison.LOWER_HIGH if event.price < prior.price else SwingComparison.EQUAL_HIGH)
        else:
            comparison = (SwingComparison.HIGHER_LOW if event.price > prior.price else
                          SwingComparison.LOWER_LOW if event.price < prior.price else SwingComparison.EQUAL_LOW)
        tagged = event.model_copy(update={"comparison": comparison})
        points.append(tagged)
        if len(self.highs) < 2 or len(self.lows) < 2:
            self.state = StructuralState.INSUFFICIENT
        else:
            rising = self.highs[-1].price > self.highs[-2].price and self.lows[-1].price > self.lows[-2].price
            falling = self.highs[-1].price < self.highs[-2].price and self.lows[-1].price < self.lows[-2].price
            self.state = StructuralState.UP if rising else StructuralState.DOWN if falling else StructuralState.MIXED
        transition = None
        if self.state in (StructuralState.UP, StructuralState.DOWN) and self.established is not None and self.established != self.state:
            transition = StructuralTransition(previous_state=self.established, new_state=self.state,
                known_at=tagged.known_at, responsible_swing=tagged.kind)
        if self.state in (StructuralState.UP, StructuralState.DOWN):
            self.established = self.state
        return tagged, transition


def structure_at_times(events: tuple[ConfirmedSwing, ...], times):
    """Return state, newly confirmed events, and any transition per sorted query time."""
    tracker = _StructureTracker()
    event_index = 0
    snapshots = []
    for known_at in times:
        fresh, transitions = [], []
        while event_index < len(events) and events[event_index].known_at <= known_at:
            tagged, transition = tracker.push(events[event_index])
            fresh.append(tagged)
            if transition is not None:
                transitions.append(transition)
            event_index += 1
        snapshots.append((tracker.state, tuple(fresh), tuple(transitions)))
    return tuple(snapshots)

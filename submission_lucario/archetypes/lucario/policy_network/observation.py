"""The single definition of "what a policy observes", shared by every
consumer — offline training, live play, and whatever replaces behavioral
cloning later.

Before this module, two places assembled an observation: ``dataset.py``
(seeking into Parquet) and ``live.py`` (accumulating a match as it happens).
They agreed only because ``test_parity.py`` checked that they did, after the
fact. Any new consumer — an RL rollout buffer, a DAgger relabeller, an
evaluation harness — would have been a third copy to keep in sync, and the
agreement is not something a policy can survive getting wrong: a model
trained on one assembly and run on another is being fed a different input
than it learned from, silently.

So the assembly lives here once, and the two existing paths keep only the
part that genuinely differs — *where the rows come from*. ``dataset.py``
reads them out of Parquet; ``live.py`` appends them as the game unfolds.
Both hand this module the same ``read_row(index)`` callable and get the same
observation back. Behavioral cloning is now just one caller among several
rather than the thing the pipeline is built around.

``ObservationSpec`` carries the choices that must match between training and
serving. Passing the same spec to both paths is what makes them consistent;
it is a value object precisely so it can be saved next to a checkpoint and
replayed later, rather than living as default arguments in two files.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable

from features import decision_chain, extract_features, opponent_history

#: How ``opponent_history`` picks the frames it diffs.
#:
#: ``"opponent_frames"`` snapshots the opponent's board at the opponent's own
#: decision frames — one entry per opponent turn, the most detailed view, and
#: what the offline replays support since they record both players.
#:
#: ``"own_frames"`` snapshots the opponent's board at *this* player's frames
#: instead, so each entry is "what changed on their board since I last
#: acted". Strictly less information, but it is the only view obtainable when
#: a policy is invoked solely for its own decisions — which is exactly the
#: competition harness's shape: ``agent()`` is never called for the
#: opponent's choices, so their individual frames simply do not exist on that
#: side.
#:
#: The default is ``"own_frames"`` for that reason: a policy must be trained
#: on the view it will actually be served, and the richer view is unavailable
#: at serve time. Choosing ``"opponent_frames"`` for training buys detail the
#: deployed policy can never see, which is a train/serve mismatch rather than
#: a free improvement.
OPPONENT_FRAMES = "opponent_frames"
OWN_FRAMES = "own_frames"


@dataclass(frozen=True)
class ObservationSpec:
    """Every choice that has to be identical between training and serving.

    Frozen because a spec silently mutating between the two is the failure
    this module exists to prevent; use ``replace(spec, ...)`` for a variant.
    """

    opponent_history_size: int = 60
    decision_chain_size: int = 60
    opponent_history_source: str = OWN_FRAMES

    def __post_init__(self) -> None:
        if self.opponent_history_source not in (OPPONENT_FRAMES, OWN_FRAMES):
            raise ValueError(
                f"opponent_history_source must be {OPPONENT_FRAMES!r} or "
                f"{OWN_FRAMES!r}, got {self.opponent_history_source!r}"
            )

    def variant(self, **changes) -> "ObservationSpec":
        return replace(self, **changes)


DEFAULT_SPEC = ObservationSpec()


def build_observation(
    read_row: Callable[[int], dict[str, Any]],
    idx: int,
    row: dict[str, Any],
    spec: ObservationSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Assemble one ``{"features": ..., "meta": ...}`` observation.

    ``read_row`` resolves a row index within the current episode's row
    sequence; ``idx`` is ``row``'s own index in it. The backward history
    scans walk that callable, so they neither know nor care whether the rows
    came from Parquet or from a live match in progress.
    """
    player_index = row["player_index"]
    features = extract_features(
        row["state"], row["selection"], row["options"], player_index
    )
    if spec.opponent_history_size > 0:
        features["opponent_history"] = opponent_history(
            read_row, idx, row, spec.opponent_history_size,
            source=spec.opponent_history_source,
        )
    if spec.decision_chain_size > 0:
        features["decision_chain"] = decision_chain(
            read_row, idx, row, spec.decision_chain_size
        )
    meta = {
        "episode_id": row["episode_id"],
        "frame_index": row["frame_index"],
        "player_index": player_index,
        "player_name": row.get("player_name"),
    }
    return {"features": features, "meta": meta}

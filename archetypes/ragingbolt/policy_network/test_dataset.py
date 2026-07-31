"""Quick manual check of PolicyFeatureDataset's output shape.

Usage:
    python archetypes/ragingbolt/policy_network/test_dataset.py data/policy_decisions_ragingbolt.parquet
    python archetypes/ragingbolt/policy_network/test_dataset.py data/policy_decisions_ragingbolt.parquet 20 --verbose
"""

import csv
from pprint import pprint
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset import PolicyFeatureDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
CARD_DATA_PATH = REPO_ROOT / "EN_Card_Data.csv"


def load_card_names() -> dict[int, str]:
    if not CARD_DATA_PATH.is_file():
        return {}
    with open(CARD_DATA_PATH, newline="", encoding="utf-8") as file:
        return {int(row["Card ID"]): row["Card Name"] for row in csv.DictReader(file)}


def card_label(card: dict | None, card_names: dict[int, str]) -> str:
    if card is None:
        return "None"
    name = card_names.get(card["id"], "") if card_names else ""
    label = f"id={card['id']}"
    if name:
        label += f" ({name})"
    label += f" serial={card['serial']}"
    return label


def pokemon_label(pokemon: dict | None, card_names: dict[int, str]) -> str:
    if pokemon is None:
        return "None"
    label = card_label(pokemon, card_names)
    label += f" hp={pokemon['hp']}/{pokemon['maxHp']} energies={pokemon['energies']}"
    if pokemon["tools"]:
        label += f" tools=[{', '.join(card_label(t, card_names) for t in pokemon['tools'])}]"
    if pokemon["preEvolution"]:
        label += f" evolved_from=[{', '.join(card_label(p, card_names) for p in pokemon['preEvolution'])}]"
    return label


def print_board(board: dict, card_names: dict[int, str]) -> None:
    print(f"  active: {'; '.join(pokemon_label(p, card_names) for p in board['active']) or '(none)'}")
    print(f"  bench ({len(board['bench'])}):")
    for pokemon in board["bench"]:
        print(f"    - {pokemon_label(pokemon, card_names)}")
    if board["hand"] is None:
        print(f"  hand: <hidden> (handCount={board['handCount']})")
    else:
        print(f"  hand ({board['handCount']}): {', '.join(card_label(c, card_names) for c in board['hand'])}")
    print(f"  deckCount: {board['deckCount']}")
    print(f"  discard ({len(board['discard'])}): "
          f"{', '.join(card_label(c, card_names) for c in board['discard']) or '(none)'}")
    print(f"  prize ({len(board['prize'])}): "
          f"{', '.join(card_label(c, card_names) for c in board['prize']) or '(none)'}")
    conditions = [c for c in ("poisoned", "burned", "asleep", "paralyzed", "confused") if board[c]]
    print(f"  status: {', '.join(conditions) if conditions else '(none)'}")


def print_decision_context(decision_context: dict, card_names: dict[int, str]) -> None:
    selection = decision_context["selection"]
    print(f"  selection: type={selection['type']} context={selection['context']} "
          f"minCount={selection['minCount']} maxCount={selection['maxCount']}")
    print(f"  options ({len(decision_context['options'])}):")
    for i, option in enumerate(decision_context["options"]):
        parts = [f"[{i}]"]
        if option.get("cardId") is not None:
            parts.append(card_label({"id": option["cardId"], "serial": option.get("serial")}, card_names))
        for field in ("type", "number", "area", "index", "attackId", "targets_opponent"):
            if option.get(field) is not None:
                parts.append(f"{field}={option[field]}")
        print("    " + " ".join(parts))


def print_opponent_history(history: list[dict], card_names: dict[int, str]) -> None:
    for turn_diff in history:
        print(f"  turn {turn_diff['turn']}: hand={turn_diff['hand_count']} deck={turn_diff['deck_count']} "
              f"prize={turn_diff['prize_count']}")
        if turn_diff["discarded_cards"]:
            print(f"    discarded: {', '.join(card_label(c, card_names) for c in turn_diff['discarded_cards'])}")
        if turn_diff["new_pokemon"]:
            print("    new pokemon:")
            for pokemon in turn_diff["new_pokemon"]:
                print(f"      + {pokemon_label(pokemon, card_names)}")
        if turn_diff["removed_pokemon"]:
            print("    removed pokemon:")
            for pokemon in turn_diff["removed_pokemon"]:
                print(f"      - {pokemon_label(pokemon, card_names)}")
        if turn_diff["energy_attached"]:
            for attach in turn_diff["energy_attached"]:
                print(f"    energy attached: serial={attach['serial']} id={attach['id']} "
                      f"types={attach['new_energy_types']}")
        if turn_diff["hp_lost"]:
            print(f"    hp_lost: {turn_diff['hp_lost']}")
        if turn_diff["status_applied"]:
            print(f"    status_applied: {turn_diff['status_applied']}")


def print_history_lengths(features: dict, dataset: PolicyFeatureDataset) -> None:
    """One-line summary of how much history this sample actually carries.

    Both sizes are caps, not guarantees: the chain is bounded by how many
    decisions this player has made so far in the episode, and the history by
    how many turns the opponent has had.  Early-game samples are therefore
    short even with the caps set to 60.
    """
    chain, history = features["decision_chain"], features["opponent_history"]
    turn = features["global_state"]["turn"]
    chain_turns = f" spanning turns {chain[0]['turn']}..{chain[-1]['turn']}" if chain else ""
    this_turn = sum(1 for step in chain if step["turn"] == turn)
    print(f"  decision_chain: {len(chain)}/{dataset.decision_chain_size} entries"
          f"{chain_turns} ({this_turn} of them in the current turn {turn})")
    history_turns = f" for turns {[entry['turn'] for entry in history]}" if history else ""
    print(f"  opponent_history: {len(history)}/{dataset.opponent_history_size} turn diffs"
          f"{history_turns}")


# def main() -> None:
parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data/policy_decisions_ragingbolt.parquet"
positional = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
verbose = "--verbose" in sys.argv[2:]
idx = int(positional[0]) if positional else 3

card_names = load_card_names() if verbose else {}

dataset = PolicyFeatureDataset(parquet_path)
# dataset[0]
pprint('player turn:')
pprint(dataset[5][0]['meta'])
print("features:")
pprint(dataset[8][0]['features'])
#     print(f"dataset size: {len(dataset)}")
#     print(f"opponent_history_size={dataset.opponent_history_size} "
#           f"decision_chain_size={dataset.decision_chain_size} (caps, not guarantees)\n")

#     observation, target_action = dataset[idx]
#     features, meta = observation["features"], observation["meta"]

#     print(f"=== sample {idx} ===")
#     print(f"episode_id={meta['episode_id']} frame_index={meta['frame_index']} "
#           f"player_index={meta['player_index']} player_name={meta['player_name']}")
#     print_history_lengths(features, dataset)
#     print()

#     print("--- state (yours) ---")
#     print_board(features["state"], card_names)
#     print()

#     print("--- opponent_state ---")
#     print_board(features["opponent_state"], card_names)
#     print()

#     print("--- global_state ---")
#     for key, value in features["global_state"].items():
#         print(f"  {key}: {value}")
#     print()

#     print("--- decision_context ---")
#     print_decision_context(features["decision_context"], card_names)
#     print()

#     history = features["opponent_history"]
#     print(f"--- opponent_history ({len(history)} turn diffs, oldest first) ---")
#     if history:
#         print_opponent_history(history, card_names)
#     else:
#         print("  (none — the opponent has not had a turn yet)")
#     print()

#     chain = features["decision_chain"]
#     print(f"--- decision_chain ({len(chain)} of this actor's own earlier "
#           f"decisions, oldest first) ---")
#     if chain:
#         for position, step in enumerate(chain):
#             print(f"  [{position}] turn={step['turn']} "
#                   f"selection_type={step['selection']['type']} "
#                   f"options={len(step['options'])} chosen target={step['target_action']}")
#             if verbose:
#                 pprint(step)
#     else:
#         print("  (none — this is this player's first decision of the episode)")
#     print()

#     print(f"target_action: {target_action}")

#     own_hand = features["state"]["hand"]
#     opp_hand = features["opponent_state"]["hand"]
#     print(f"\nsanity check: own hand visible={own_hand is not None}, "
#           f"opponent hand visible={opp_hand is not None} (expected: True, False)")

#     if not verbose:
#         print("\n(pass --verbose to resolve card ids to names via EN_Card_Data.csv)")


# if __name__ == "__main__":
#     main()

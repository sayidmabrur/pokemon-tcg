
"""Play one full match driving the bundled agent() exactly as the harness
does — same loading mechanism, same per-decision call, nothing from the
source tree. Run in a subprocess with cwd set to the bundle so an accidental
dependency on the repo shows up as a failure here rather than at grading."""
import random
import sys
from pathlib import Path

from cg.game import battle_finish, battle_select, battle_start

# Load main.py the way kaggle_environments.get_last_callable does: read the
# source and exec it against a bare globals dict. Deliberately NOT ``import
# main`` — an import binds __file__, __name__ and a real module object, none
# of which exist under the grader. Anything the module does at import time
# that depends on them passes here and fails there, which is exactly how a
# NameError on __file__ reached a live submission.
_source = Path("main.py").read_text()
main = {}
exec(compile(_source, "main.py", "exec"), main)

agent = main["agent"]

# get_last_callable hands the grader the *last* callable defined in the
# module, not one named "agent". If a later edit appends another function,
# that one silently becomes the submission.
last_callable = [k for k, v in main.items() if callable(v) and not k.startswith("__")][-1]
assert last_callable == "agent", (
    f"the last callable defined in main.py is {last_callable!r}, not 'agent' — "
    f"kaggle_environments would submit that instead"
)

deck = main["read_deck_csv"]()
assert len(deck) == 60, f"deck.csv yielded {len(deck)} cards"

obs, _ = battle_start(deck, deck)
decisions = 0
while obs["current"]["result"] == -1:
    select = obs["select"]
    if obs["current"]["yourIndex"] == 0:
        action = agent(obs)
        decisions += 1
    else:
        # Opponent seat: any legal move. The point is to exercise our agent
        # inside a real game, not to make the opponent good.
        count = min(random.randint(select["minCount"], select["maxCount"]),
                    len(select["option"]))
        action = random.sample(range(len(select["option"])), count)
    obs = battle_select(action)

print(f"OK: match finished, result={obs['current']['result']}, "
      f"{decisions} agent decisions")

# The harness opens a match by calling agent() with select=None to collect
# the decklist. Exercise that branch against a real observation shape rather
# than a synthetic dict, since to_observation_class validates the whole
# payload and a hand-built stub would only prove the stub is wrong.
deck_again = agent({**obs, "select": None})
assert len(deck_again) == 60, f"agent() returned {len(deck_again)} cards"
print("OK: decklist branch returned 60 cards")

if main["_policy"].failed:
    print("FAIL: the policy raised and fell back to random", file=sys.stderr)
    sys.exit(1)
if decisions == 0:
    print("FAIL: agent() was never called for a decision", file=sys.stderr)
    sys.exit(1)
battle_finish()

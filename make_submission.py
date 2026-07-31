"""Package the trained policy into a self-contained submission bundle.

    python make_submission.py

Produces ``submission/`` plus ``submission.zip`` containing:

    main.py                                  agent() entry point (inference)
    deck.csv                                 60 card ids
    bc_policy.pt                             trained weights
    cg/                                      game engine (incl. native libs)
    archetypes/alakazam/policy_network/      feature pipeline + network

The policy_network files keep their full repo-relative path inside the
bundle rather than being flattened to the top level. That is deliberate:
``vocab.py`` reaches the engine via ``parents[3]`` relative to its own
location, so flattening the layout would silently break that import. Keeping
the depth identical means every module is copied verbatim, with no
submission-only patches to drift out of sync with the originals.

The deck defaults to ``alakazam_deck.csv`` — the decklist reconstructed from
the replays the policy was cloned from — not the repo-root ``deck.csv``,
which is a different archetype the network has never seen played.

Verify the result before submitting:

    python make_submission.py --verify

which imports the bundle in a fresh interpreter and plays a full match
through it, exercising the same ``agent()`` the harness will call.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

#: Engine package. The native libraries for all three platforms are included
#: — they total ~4MB and which one gets loaded depends on the grader's OS,
#: which is not worth guessing wrong.
_CG_FILES = (
    "__init__.py", "api.py", "game.py", "sim.py", "utils.py",
    "libcg.so", "libcg-arm64.so", "libcg.dylib", "cg.dll",
)

#: Inference-time modules only. ``bc_train.py``/``convert_replays.py``/
#: ``test_*.py`` are training and tooling; they would drag in pyarrow and
#: the parquet dataset for no runtime benefit.
_POLICY_FILES = (
    "features.py", "vocab.py", "dataset.py", "collate.py",
    "policy_experimental.py", "live.py", "observation.py",
)

_POLICY_DIR = Path("archetypes/alakazam/policy_network")


def build(deck: Path, checkpoint: Path, out_dir: Path, zip_path: Path | None) -> Path:
    missing = [p for p in (deck, checkpoint) if not p.is_file()]
    if missing:
        raise SystemExit(
            "missing required file(s): " + ", ".join(str(p) for p in missing)
            + "\n(build the deck with archetypes/alakazam/build_alakazam_deck.py, "
              "train weights with archetypes/alakazam/policy_network/bc_train.py)"
        )

    deck_ids = [line for line in deck.read_text().split("\n") if line.strip()]
    if len(deck_ids) != 60:
        raise SystemExit(f"{deck} has {len(deck_ids)} cards; the engine requires exactly 60")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / _POLICY_DIR).mkdir(parents=True)
    (out_dir / "cg").mkdir(parents=True)

    shutil.copy2(_ROOT / "submission_main.py", out_dir / "main.py")
    shutil.copy2(deck, out_dir / "deck.csv")
    shutil.copy2(checkpoint, out_dir / "bc_policy.pt")
    for name in _CG_FILES:
        source = _ROOT / "cg" / name
        if source.is_file():
            shutil.copy2(source, out_dir / "cg" / name)
        else:
            print(f"  note: cg/{name} not present, skipping")
    for name in _POLICY_FILES:
        shutil.copy2(_ROOT / _POLICY_DIR / name, out_dir / _POLICY_DIR / name)

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"built {out_dir}/ — {total / 1e6:.1f} MB")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out_dir)}  ({path.stat().st_size / 1e3:.0f} KB)")

    if zip_path is not None:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(out_dir))
        print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return out_dir


_VERIFY_SCRIPT = '''
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
'''


def verify(out_dir: Path) -> int:
    script = out_dir / "_verify.py"
    script.write_text(_VERIFY_SCRIPT)
    try:
        result = subprocess.run(
            [sys.executable, "_verify.py"], cwd=out_dir, capture_output=True, text=True
        )
    finally:
        script.unlink(missing_ok=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        print("VERIFY FAILED — do not submit this bundle", file=sys.stderr)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default=str(_ROOT / "alakazam_deck.csv"))
    parser.add_argument("--checkpoint", default=str(_ROOT / "checkpoints/bc_policy.pt"))
    parser.add_argument("--out", default=str(_ROOT / "submission"))
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--verify", action="store_true",
        help="play a full match through the packaged agent() before submitting",
    )
    args = parser.parse_args()

    out_dir = build(
        Path(args.deck), Path(args.checkpoint), Path(args.out),
        None if args.no_zip else Path(args.out + ".zip"),
    )
    if args.verify:
        raise SystemExit(verify(out_dir))


if __name__ == "__main__":
    main()

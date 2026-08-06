from pathlib import Path
import json

def extract_samples(replay_file):
    with open(replay_file, "r") as f:
        replay = json.load(f)

    samples = []

    match_id = Path(replay_file).stem

    for step_idx, step in enumerate(replay["steps"]):

        for player_idx, player_step in enumerate(step):

            obs = player_step.get("observation")
            action = player_step.get("action")

            if obs is None:
                continue

            select = obs.get("select")

            # only decision states
            if select is None:
                continue


extract_samples("replays/benarg/2023-09-30_20-51-17.json")
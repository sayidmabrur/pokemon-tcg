"""Adapted from the repo-root ``duel_inference.py`` — same rule-based-vs-RL
duel loop, but the placeholder ``RandomPolicy`` is swapped for the
imitation-learned ``PolicyNetwork`` (trained via ``policy_network/bc_train.py``),
so you can watch the BC policy actually play against ``PolicyRuleBased``.
"""

import random
import sys
from pathlib import Path
from pprint import pprint

import torch

# This file lives in archetypes/alakazam/ (not the repo root, like the
# original), so the repo root isn't on sys.path by default — needed for
# ``main``/``cg.game``/``crustle_rule_based_agent``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent / "policy_network"))

from main import read_deck_csv
from cg.game import battle_finish, battle_select, battle_start
from crustle_rule_based_agent import PolicyRuleBased

from live import LiveFeatureExtractor
from dataset import transform
from policy_experimental import PolicyNetwork
from collate import collate_features


episodes = 10

p0 = read_deck_csv()
p1 = read_deck_csv()


class BCPolicy:
    """Wraps the trained imitation-learning ``PolicyNetwork`` behind the same
    ``.act(obs) -> list[int]`` interface as ``PolicyRuleBased``.

    Unlike ``PolicyRuleBased`` (which reads straight off ``obs``), this needs
    the *featurised* observation — but the duel loop already runs one shared
    ``LiveFeatureExtractor`` per decision (both players' history depends on
    seeing each other's frames — see ``live.py``), so this reuses that
    already-computed ``observation`` rather than extracting its own (a second
    extractor instance would double-append rows and break the decision_chain/
    opponent_history scans).
    """

    def __init__(self, checkpoint: str | None = None):
        self.policy = PolicyNetwork()
        # Resolve relative to this file, not the process's cwd — otherwise
        # where the checkpoint is found depends on which directory you
        # happened to launch the script from.
        if checkpoint is None:
            checkpoint_path = Path(__file__).parent / "policy_network" / "bc_policy.pt"
        else:
            checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_file():
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        else:
            print(f"[BCPolicy] no checkpoint at {checkpoint_path} — using randomly initialized weights")
        self.policy.eval()

    def act(self, obs: dict, observation: dict) -> list[int]:
        # PolicyNetwork is batch-only (see collate.py) — wrap this single
        # live decision into a batch of 1.
        features = collate_features([transform(observation)])
        with torch.no_grad():
            logits = self.policy(features)[0]  # (max_options,)

        select = obs["select"]
        num_valid = int(torch.isfinite(logits).sum().item())
        count = min(random.randint(select["minCount"], select["maxCount"]), num_valid)
        return torch.topk(logits, count).indices.tolist()


rule_based_policy = PolicyRuleBased()
# The repo-root copy (earliest timestamp of the 3 on disk) is the one that
# predates the later debug/smoke-test runs which clobbered the
# policy_network/ copy with a throwaway 1000-sample checkpoint — this is
# the real ~5-hour trained one.
rl_policy = BCPolicy(checkpoint=str(_REPO_ROOT / "bc_policy.pt"))


policies = [rule_based_policy, rl_policy]

# One shared extractor: both players' decisions belong to the same
# observation stream, and building either side's opponent history needs the
# other side's frames.  It reads yourIndex per obs, so it stays correct.
feature_extractor = LiveFeatureExtractor()

trajectories = {0: [], 1: []}

# result = -1 means that the game is running
p0_win = 0
p1_win = 0
for i in range(episodes):
    step = 1
    obs, start_data = battle_start(p0, p1)
    feature_extractor.reset(episode_id=i)
    j = 0
    step_p1 = 0
    while obs["current"]["result"] == -1:
        player_idx = obs["current"]["yourIndex"]
        policy = policies[player_idx]
        select = obs["select"]
        option = select["option"]

        # Call the extractor exactly once per decision: each call appends a
        # row to the episode memory that decision_chain/opponent_history scan.
        observation = feature_extractor(obs)
        if player_idx == 1:
            observation_p1 = observation
            step_p1 = step_p1 + 1
            # print("observation:")
            # pprint(observation)
        if isinstance(policy, BCPolicy):
            action = policy.act(obs, observation)
        else:
            action = policy.act(obs)
        feature_extractor.record_action(action)
        trajectories[player_idx].append((observation, action))
        obs = battle_select(action)
        step += 1
        # if j ==50:
            # break
        # j=j+1

    result = obs["current"]["result"]
    battle_finish()

    if result == 0:
        p0_win += 1
    else:
        p1_win += 1
    # break
    winner = "p0" if result == 0 else "p1"
    print(f"Episode {i+1}/{episodes} finished in {step} steps. Winner: {winner}")
print("rule_based_policy win:", p0_win)
print("rl_policy win:", p1_win)

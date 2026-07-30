import random
import sys
from pathlib import Path
from pprint import pprint

from main import read_deck_csv
from cg.game import battle_finish, battle_select, battle_start
from crustle_rule_based_agent import PolicyRuleBased

sys.path.insert(0, str(Path(__file__).parent / "archetypes/alakazam/policy_network"))
from live import LiveFeatureExtractor


# crustle_deck = build_deck("crustle_deck.csv")

episodes = 100

p0 = read_deck_csv()
p1 = read_deck_csv()


class RandomPolicy:
    """Placeholder for the network under training — picks a legal selection."""

    def act(self, obs):
        select = obs["select"]
        count = random.randint(select["minCount"], select["maxCount"])
        return random.sample(range(len(select["option"])), count)


rule_based_policy = PolicyRuleBased()
rl_policy = RandomPolicy()


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
        action = policy.act(obs)
        feature_extractor.record_action(action)
        trajectories[player_idx].append((observation, action))
        obs = battle_select(action)
        step += 1
        # if j ==50:
            # break
        # j=j+1
    
    print("step_p1:", step_p1)
    print("decision_chain:", len(observation_p1['features']['decision_chain']))
    print("opponent_history:", len(observation_p1['features']['opponent_history']))
    result = obs["current"]["result"]
    battle_finish()

    if result == 0:
        p0_win += 1
    else:
        p1_win += 1
    break
print("rule_based_policy win:", p0_win)
print("rl_policy win:", p1_win)

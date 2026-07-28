import random
from pprint import pprint

from main import read_deck_csv
from deck import build_deck
from cg.game import battle_finish, battle_select, battle_start
from policy import Policy
from crustle_rule_based_agent import PolicyRuleBased, rule_based_select


# crustle_deck = build_deck("crustle_deck.csv")

episodes = 100
p0 = read_deck_csv()
p1 = read_deck_csv()


rule_based_policy = PolicyRuleBased()
rl_policy = Policy()


policies = [rule_based_policy, rl_policy]

trajectories = {0: [], 1: []}

# result = -1 means that the game is running
p0_win = 0
p1_win = 0
for i in range(episodes):
    step = 1
    obs, start_data = battle_start(p0, p1)
    while obs["current"]["result"] == -1:
        player_idx = obs["current"]["yourIndex"]
        policy = policies[player_idx]
        select = obs["select"]
        option = select["option"]
    
        action = policy.act(obs)
        trajectories[player_idx].append((obs, action))
        obs = battle_select(action)
        step += 1
    
    result = obs["current"]["result"]
    battle_finish()

    if result == 0:
        p0_win += 1
    else:
        p1_win += 1
print("rule_based_policy win:", p0_win)
print("rl_policy win:", p1_win)

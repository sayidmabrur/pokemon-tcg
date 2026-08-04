# Policy Architecture
## Backbone:
backbone uses ensembled stack of Encoders

```
                    +----------------------+
Decision Chain ---->| Decision Encoder     |
                    +----------------------+

Decision Context -->| Context Encoder      |----+
                                              |
Global State ------>| Global Encoder       |   |
                                              +--> Feature Fusion --> Policy Head --> Action Logits -> Softmax -> Action Probs
Opponent History -->| History Encoder      |   |
                                              |
Player State ------>| Player Encoder       |   |
                                              |
Opponent State ---->| Player Encoder       |---+
```

## Value Function Architecture
## Backbone:
For value function, the goal is only to predict accumulated reward, so I think vanilla NN will be enough

# Training goals

## Stage 1: Supervised learning through imitation learning.
Pretrained the policy with imitation learning as a starting point. So it's not stuck at exploiting weak model when self-play training
## Stage 2: RL Training using Imperfect Information Games Method (Nash Equilibrium Convergence)
the goal is to converge to NE, and producing unexploitable strategy across different decks. If it's only able to be unexploitable against 1 or 2 decks will be a good sign.

methods worth to try:
- https://arxiv.org/abs/1906.00190
- https://arxiv.org/abs/1811.00164

# Blockers
the Environment given is written in C# / C++ & raw python code. Probably it'll need to tweak some of the environment for training purpose only, so GPU can be utilized. For now, all the training produced using only CPUs

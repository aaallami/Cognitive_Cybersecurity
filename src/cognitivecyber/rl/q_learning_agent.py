"""
Lightweight tabular Q-learning agent for defense-action selection.

Environment (implemented inline rather than via Gymnasium, to avoid a
heavy dependency for a small discrete state/action space):

  State  = (attack_category, compromised_bucket, kg_risk_bucket)
  Action = {"isolate", "patch", "monitor", "no_op"}
  Reward = -(post_action_compromised_hosts) - action_cost + containment_bonus

This is a genuine, executable RL training loop (no external RL library
required) run against `cognitivecyber.digital_twin.network_sim`. It is a
small proof-of-concept, not a production adaptive-defense system: state
space, reward shaping, and action costs below are illustrative modeling
choices, documented as such.
"""

from __future__ import annotations

import random
from collections import defaultdict

from cognitivecyber.digital_twin.network_sim import _apply_action, simulate_propagation

ACTIONS = ["isolate", "patch", "monitor", "no_op"]
ACTION_COST = {"isolate": 1.5, "patch": 1.0, "monitor": 0.2, "no_op": 0.0}


def _bucket(x: float, edges=(2, 6, 15)) -> int:
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


class QLearningDefenseAgent:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, random_state=42):
        self.q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(random_state)

    def select_action(self, state, greedy: bool = False) -> str:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        qvals = self.q[state]
        return max(qvals, key=qvals.get)

    def update(self, state, action, reward, next_state):
        best_next = max(self.q[next_state].values())
        td_target = reward + self.gamma * best_next
        self.q[state][action] += self.alpha * (td_target - self.q[state][action])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_episode(g, agent: QLearningDefenseAgent, category: str, kg_risk: float,
                 seed_host: int, train: bool = True, rng: random.Random = None):
    """One episode = one detected incident -> agent picks a defense action
    -> reward based on resulting compromise footprint. Returns
    (reward, compromised_after, action_taken, baseline_compromised)."""
    rng = rng or random.Random()
    baseline = simulate_propagation(g, seed_host, category, rng=rng)
    state = (category, _bucket(baseline), _bucket(kg_risk * 10, edges=(3, 6, 8)))

    action = agent.select_action(state, greedy=not train)

    g_work = g.copy()
    _apply_action(g_work, seed_host, action)
    compromised_after = simulate_propagation(g_work, seed_host, category, rng=rng)

    containment_bonus = 3.0 if compromised_after < baseline * 0.3 else 0.0
    reward = -compromised_after - ACTION_COST[action] + containment_bonus

    next_state = (category, _bucket(compromised_after), _bucket(kg_risk * 10, edges=(3, 6, 8)))
    if train:
        agent.update(state, action, reward, next_state)
        agent.decay()

    return reward, compromised_after, action, baseline


def static_policy(baseline_compromised: int) -> str:
    """Simple rule-based baseline policy (proxy for a non-adaptive
    SOAR-style playbook): isolate if the seed host would spread widely,
    patch if moderate, else just monitor."""
    if baseline_compromised > 10:
        return "isolate"
    elif baseline_compromised > 3:
        return "patch"
    return "monitor"

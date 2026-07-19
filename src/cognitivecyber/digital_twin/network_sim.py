"""
Rule-based enterprise-network digital twin.

A genuine, executable (if intentionally simplified) simulation:
  - networkx topology of hosts grouped into subnets
  - stochastic attack propagation (SIR-style: susceptible -> compromised)
  - defense actions (patch / isolate / monitor / no-op) as graph mutations
  - "digital-twin validation": Monte-Carlo rollouts BEFORE committing an
    action, to estimate whether it will reduce risk, vs. committing blindly

This is a simplified proof-of-concept, not a validated model of any real
enterprise network -- propagation probabilities and impact weights below
are illustrative modeling assumptions, documented as such, not measured
from real incident data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx

# Illustrative per-category propagation probability and impact weight.
# Modeling assumption, not measured from real incident data.
CATEGORY_PARAMS = {
    "DoS":            dict(p_spread=0.35, impact=0.6),
    "DDoS":           dict(p_spread=0.55, impact=0.9),
    "PortScan":       dict(p_spread=0.10, impact=0.2),
    "BruteForce":     dict(p_spread=0.25, impact=0.5),
    "Exploits":       dict(p_spread=0.40, impact=0.75),
    "Reconnaissance": dict(p_spread=0.08, impact=0.15),
    "Botnet":         dict(p_spread=0.45, impact=0.7),
    "Backdoor":       dict(p_spread=0.30, impact=0.65),
}


def build_topology(n_hosts: int = 40, n_subnets: int = 4, random_state: int = 42) -> nx.Graph:
    rng = random.Random(random_state)
    g = nx.Graph()
    subnets = [f"subnet_{i}" for i in range(n_subnets)]
    for h in range(n_hosts):
        subnet = subnets[h % n_subnets]
        g.add_node(h, subnet=subnet, compromised=False, patched=False, isolated=False)
    # Connect hosts within the same subnet densely, across subnets sparsely
    hosts_by_subnet = {s: [h for h in g.nodes if g.nodes[h]["subnet"] == s] for s in subnets}
    for s, hosts in hosts_by_subnet.items():
        for i in range(len(hosts)):
            for j in range(i + 1, len(hosts)):
                if rng.random() < 0.35:
                    g.add_edge(hosts[i], hosts[j])
    # A few inter-subnet links (gateway hosts)
    for i in range(len(subnets) - 1):
        a = rng.choice(hosts_by_subnet[subnets[i]])
        b = rng.choice(hosts_by_subnet[subnets[i + 1]])
        g.add_edge(a, b)
    return g


def _reset(g: nx.Graph):
    for n in g.nodes:
        g.nodes[n]["compromised"] = False


def simulate_propagation(g: nx.Graph, seed_host: int, category: str, steps: int = 6, rng: random.Random = None) -> int:
    """Run one stochastic propagation rollout; returns final compromised count."""
    rng = rng or random.Random()
    params = CATEGORY_PARAMS.get(category, dict(p_spread=0.2, impact=0.3))
    _reset(g)
    if g.nodes[seed_host]["isolated"] or g.nodes[seed_host]["patched"]:
        return 0
    g.nodes[seed_host]["compromised"] = True
    frontier = {seed_host}
    for _ in range(steps):
        new_frontier = set()
        for h in frontier:
            for nb in g.neighbors(h):
                if g.nodes[nb]["compromised"] or g.nodes[nb]["isolated"]:
                    continue
                p = params["p_spread"] * (0.3 if g.nodes[nb]["patched"] else 1.0)
                if rng.random() < p:
                    g.nodes[nb]["compromised"] = True
                    new_frontier.add(nb)
        if not new_frontier:
            break
        frontier = new_frontier
    return sum(1 for n in g.nodes if g.nodes[n]["compromised"])


@dataclass
class TwinValidationResult:
    predicted_reduction: float
    actual_reduction: float
    correct_direction: bool
    confidence: float


def validate_action(g: nx.Graph, seed_host: int, category: str, action: str,
                     n_rollouts: int = 20, random_state: int = 0) -> TwinValidationResult:
    """Digital-twin validation: Monte-Carlo estimate the risk reduction of
    `action` before committing it, then compare against a fresh rollout
    ("ground truth") to measure how accurate the twin's prediction was.
    """
    rng = random.Random(random_state)

    # baseline (no action) rollouts
    baseline_counts = [simulate_propagation(g, seed_host, category, rng=rng) for _ in range(n_rollouts)]
    baseline_mean = sum(baseline_counts) / len(baseline_counts)

    # apply action on a working copy, twin-simulate rollouts
    g_twin = g.copy()
    _apply_action(g_twin, seed_host, action)
    twin_counts = [simulate_propagation(g_twin, seed_host, category, rng=rng) for _ in range(n_rollouts)]
    twin_mean = sum(twin_counts) / len(twin_counts)
    predicted_reduction = baseline_mean - twin_mean

    # "ground truth": apply the action for real and run a fresh independent rollout
    g_real = g.copy()
    _apply_action(g_real, seed_host, action)
    actual_count = simulate_propagation(g_real, seed_host, category, rng=random.Random(random_state + 999))
    # average a few fresh rollouts as the "actual" outcome estimate
    actual_counts = [simulate_propagation(g_real, seed_host, category, rng=random.Random(random_state + 1000 + i))
                      for i in range(n_rollouts)]
    actual_mean = sum(actual_counts) / len(actual_counts)
    actual_reduction = baseline_mean - actual_mean

    correct_direction = (predicted_reduction > 0) == (actual_reduction > 0) or abs(actual_reduction) < 1e-6
    confidence = max(0.0, 1.0 - abs(predicted_reduction - actual_reduction) / (baseline_mean + 1e-6))

    return TwinValidationResult(predicted_reduction, actual_reduction, correct_direction, confidence)


def _apply_action(g: nx.Graph, seed_host: int, action: str):
    if action == "isolate":
        g.nodes[seed_host]["isolated"] = True
    elif action == "patch":
        g.nodes[seed_host]["patched"] = True
        for nb in g.neighbors(seed_host):
            g.nodes[nb]["patched"] = True
    elif action == "monitor":
        pass  # no structural change
    elif action == "no_op":
        pass
    else:
        raise ValueError(f"Unknown action: {action}")

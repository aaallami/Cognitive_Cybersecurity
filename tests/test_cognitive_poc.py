import random

from cognitivecyber.collaboration.trust import run_collaboration_simulation, summarize, trust_score
from cognitivecyber.digital_twin.network_sim import build_topology, simulate_propagation, validate_action
from cognitivecyber.knowledge_graph.kg_builder import AttackKnowledgeGraph
from cognitivecyber.rl.q_learning_agent import QLearningDefenseAgent, run_episode, static_policy


def test_kg_grows_monotonically():
    kg = AttackKnowledgeGraph()
    prev_nodes, prev_edges = 0, 0
    for i, cat in enumerate(["DoS", "PortScan", "Botnet", "DoS", "Backdoor"]):
        kg.add_incident(cat, episode=i)
        assert kg.graph.number_of_nodes() >= prev_nodes
        assert kg.graph.number_of_edges() >= prev_edges
        prev_nodes, prev_edges = kg.graph.number_of_nodes(), kg.graph.number_of_edges()


def test_kg_risk_score_bounded():
    kg = AttackKnowledgeGraph()
    for cat in ["DoS", "DDoS", "PortScan", "BruteForce", "Exploits", "Reconnaissance", "Botnet", "Backdoor"]:
        kg.add_incident(cat)
        score = kg.technique_risk_score(cat)
        assert 0.0 <= score <= 1.0


def test_digital_twin_topology_shape():
    g = build_topology(n_hosts=30, n_subnets=3, random_state=1)
    assert g.number_of_nodes() == 30


def test_digital_twin_propagation_bounded():
    g = build_topology(n_hosts=20, n_subnets=2, random_state=1)
    n = simulate_propagation(g, seed_host=0, category="DDoS", rng=random.Random(1))
    assert 0 <= n <= 20


def test_digital_twin_isolate_reduces_or_matches_spread():
    g = build_topology(n_hosts=30, n_subnets=3, random_state=2)
    res = validate_action(g, seed_host=0, category="DDoS", action="isolate", n_rollouts=10, random_state=2)
    assert res.predicted_reduction >= -1e-6  # isolating the seed should never make things worse
    assert 0.0 <= res.confidence <= 1.0 + 1e-6


def test_rl_agent_learns_nontrivial_policy():
    g = build_topology(n_hosts=30, n_subnets=3, random_state=3)
    agent = QLearningDefenseAgent(random_state=3)
    rng = random.Random(3)
    rewards = []
    for ep in range(300):
        cat = rng.choice(["DoS", "DDoS", "PortScan"])
        seed = rng.randrange(g.number_of_nodes())
        reward, *_ = run_episode(g, agent, cat, kg_risk=0.4, seed_host=seed, train=True, rng=rng)
        rewards.append(reward)
    # learned Q-table should have entries and not be uniformly zero everywhere
    assert len(agent.q) > 0
    nonzero = any(any(v != 0.0 for v in qvals.values()) for qvals in agent.q.values())
    assert nonzero


def test_static_policy_thresholds():
    assert static_policy(20) == "isolate"
    assert static_policy(5) == "patch"
    assert static_policy(1) == "monitor"


def test_trust_score_bounded():
    t = trust_score(agent_confidence=0.9, kg_plausibility=0.5, historical_accuracy=0.8)
    assert 0.0 <= t <= 1.0


def test_collaboration_simulation_produces_valid_summary():
    records = [
        dict(agent_confidence=0.95, kg_plausibility=0.5, historical_accuracy=0.9, agent_action_correct=True),
        dict(agent_confidence=0.6, kg_plausibility=0.3, historical_accuracy=0.5, agent_action_correct=False),
        dict(agent_confidence=0.99, kg_plausibility=0.55, historical_accuracy=0.95, agent_action_correct=True),
    ] * 20
    results = run_collaboration_simulation(records, threshold=0.88, random_state=1)
    summary = summarize(results)
    assert summary["n_incidents"] == len(records)
    assert 0.0 <= summary["analyst_workload_reduction"] <= 1.0
    assert 0.0 <= summary["human_intervention_frequency"] <= 1.0
    assert abs(summary["analyst_workload_reduction"] + summary["human_intervention_frequency"] - 1.0) < 1e-6

"""
Trust score and simulated human-AI collaboration workflow.

Implements a concrete trust function combining:
  - agent confidence (REAL: predict_proba from a trained baseline classifier)
  - KG plausibility (REAL: AttackKnowledgeGraph.technique_risk_score)
  - a simulated historical-accuracy term (proxy for "the agent's track
    record on similar past incidents", since no real deployment history
    exists yet)

and a SIMULATED analyst-approval model: the probability an analyst agrees
with the agent's recommendation is modeled as an increasing function of
trust score. This substitutes for -- and does NOT replace the need for --
the controlled user study with real cybersecurity analysts that the
manuscript's RQ5 methodology calls for. Every metric produced here should
be read as "how the trust-gating mechanism behaves under a modeled analyst
response," not as evidence about real analyst behavior.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def trust_score(agent_confidence: float, kg_plausibility: float, historical_accuracy: float,
                 w_conf: float = 0.5, w_kg: float = 0.2, w_hist: float = 0.3) -> float:
    """Weighted trust function, T = w1*confidence + w2*KG_plausibility + w3*historical_accuracy.
    Weights are illustrative (sum to 1), not fit to any labeled trust dataset."""
    return round(w_conf * agent_confidence + w_kg * kg_plausibility + w_hist * historical_accuracy, 4)


def simulated_analyst_agreement(trust: float, rng: random.Random) -> bool:
    """Models P(analyst agrees) as a logistic-shaped increasing function of
    trust. Modeling assumption -- see module docstring."""
    p_agree = 1 / (1 + pow(2.71828, -25 * (trust - 0.86)))
    return rng.random() < p_agree


@dataclass
class CollaborationEpisodeResult:
    trust: float
    auto_executed: bool
    analyst_agreed: bool | None
    correct_action: bool


def run_collaboration_simulation(records, threshold: float = 0.75, random_state: int = 42):
    """records: list of dicts with keys agent_confidence, kg_plausibility,
    historical_accuracy, agent_action_correct (bool, whether the agent's
    recommended action matched the twin-validated best action).
    """
    rng = random.Random(random_state)
    results = []
    for r in records:
        t = trust_score(r["agent_confidence"], r["kg_plausibility"], r["historical_accuracy"])
        auto = t >= threshold
        if auto:
            results.append(CollaborationEpisodeResult(t, True, None, r["agent_action_correct"]))
        else:
            agreed = simulated_analyst_agreement(t, rng)
            # if analyst disagrees, assume analyst intervention corrects the action
            correct = r["agent_action_correct"] if agreed else True
            results.append(CollaborationEpisodeResult(t, False, agreed, correct))
    return results


def summarize(results) -> dict:
    n = len(results)
    auto = sum(1 for r in results if r.auto_executed)
    escalated = n - auto
    agreed = [r for r in results if r.analyst_agreed is not None]
    agreement_rate = (sum(1 for r in agreed if r.analyst_agreed) / len(agreed)) if agreed else float("nan")
    correct = sum(1 for r in results if r.correct_action)
    return {
        "n_incidents": n,
        "analyst_workload_reduction": round(auto / n, 4),
        "human_intervention_frequency": round(escalated / n, 4),
        "decision_agreement_rate": round(agreement_rate, 4) if agreed else None,
        "mean_trust_score": round(sum(r.trust for r in results) / n, 4),
        "overall_decision_success_rate": round(correct / n, 4),
    }

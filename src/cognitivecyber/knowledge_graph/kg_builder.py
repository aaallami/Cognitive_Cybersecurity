"""
Rule-based attack-graph / knowledge-graph construction.

Builds a networkx DiGraph linking:
  attack_category -> ATT&CK technique -> ATT&CK tactic -> (next tactic in
  kill-chain order, if also observed)

This is a genuine, executable, fully offline knowledge-graph proof-of-concept
(no live threat-intel feed required). It supports:
  - incremental growth (add_incident) to simulate "knowledge evolution"
  - simple reasoning: shortest path from an attack category to "Impact"
  - graph-size metrics (nodes/edges over time) for the knowledge-evolution figure
"""

from __future__ import annotations

import networkx as nx

from .attack_reference import ATTACK_CAT_TO_TECHNIQUES, TACTIC_ORDER, TECHNIQUE_LOOKUP


class AttackKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._growth_log = []  # (episode, n_nodes, n_edges)
        self._seen_categories = set()

    def _ensure_tactic_chain(self):
        for i in range(len(TACTIC_ORDER) - 1):
            a, b = TACTIC_ORDER[i], TACTIC_ORDER[i + 1]
            if self.graph.has_node(a) and self.graph.has_node(b):
                self.graph.add_edge(a, b, kind="kill_chain")

    def add_incident(self, attack_category: str, episode: int = 0):
        """Register an observed incident of `attack_category`, growing the
        graph with any not-yet-seen technique/tactic nodes and edges."""
        self.graph.add_node(attack_category, kind="attack_category")
        techniques = ATTACK_CAT_TO_TECHNIQUES.get(attack_category, [])
        for tid in techniques:
            info = TECHNIQUE_LOOKUP.get(tid, {"name": tid, "tactic": "Unknown"})
            self.graph.add_node(tid, kind="technique", name=info["name"], tactic=info["tactic"])
            self.graph.add_edge(attack_category, tid, kind="maps_to")
            tactic = info["tactic"]
            self.graph.add_node(tactic, kind="tactic")
            self.graph.add_edge(tid, tactic, kind="belongs_to")
        self._ensure_tactic_chain()
        self._seen_categories.add(attack_category)
        self._growth_log.append((episode, self.graph.number_of_nodes(), self.graph.number_of_edges()))

    def reasoning_path_to_impact(self, attack_category: str):
        """Shortest path from an attack category to the 'Impact' tactic, if any."""
        if attack_category not in self.graph or "Impact" not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, attack_category, "Impact")
        except nx.NetworkXNoPath:
            return None

    def technique_risk_score(self, attack_category: str) -> float:
        """A simple graph-based plausibility/risk score: how many distinct
        techniques and how close to 'Impact' this category's techniques sit.
        Purely a function of graph topology -- not a learned or calibrated score.
        """
        techniques = ATTACK_CAT_TO_TECHNIQUES.get(attack_category, [])
        if not techniques:
            return 0.0
        path = self.reasoning_path_to_impact(attack_category)
        proximity = 1.0 / len(path) if path else 0.1
        breadth = min(len(techniques) / 4.0, 1.0)
        return round(0.5 * proximity + 0.5 * breadth, 4)

    def growth_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._growth_log, columns=["episode", "n_nodes", "n_edges"])

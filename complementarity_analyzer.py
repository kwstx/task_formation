from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Set, Tuple, Sequence
import math

from cooperative_intelligence import CooperativeIntelligenceVector
from cooperative_context_model import CooperativeContextTensor, CooperativeContextModel
from synergy_forecast_simulator import HistoricalCoalitionRecord


@dataclass(frozen=True)
class CapabilityDependency:
    """Represents a structural dependency where one capability enables or enriches another."""
    source: str
    target: str
    strength: float = 1.0  # [0, 1] degree of enablement


class ComplementarityAnalyzer:
    """
    Measures how agent capability vectors interact structurally rather than independently.
    
    This analyzer prioritizes 'complementarity'—the degree to which agents fulfill 
    each other's structural dependencies and align with historical success patterns—
    rather than just summing their individual skill levels.
    """

    def __init__(
        self,
        dependencies: List[CapabilityDependency] | None = None,
        historical_records: Sequence[HistoricalCoalitionRecord] | None = None
    ) -> None:
        self.dependencies = dependencies or []
        self.historical_records = historical_records or []
        self._adj: Dict[str, List[Tuple[str, float]]] = {}
        self._build_dependency_map()

    def _build_dependency_map(self) -> None:
        """Constructs an adjacency list representation of the capability dependencies."""
        self._adj = {}
        for dep in self.dependencies:
            if dep.source not in self._adj:
                self._adj[dep.source] = []
            self._adj[dep.source].append((dep.target, dep.strength))

    def compute_structural_complementarity(
        self,
        agent_a: CooperativeIntelligenceVector,
        agent_b: CooperativeIntelligenceVector
    ) -> float:
        """
        Calculates a structural complementarity score between two agents.
        
        Integrates dependency graph enrichment metrics with historical collaboration 
        topology patterns to project synergy potential.
        """
        # 1. Dependency Graph Enrichment
        # Measures how much agent_a enriches agent_b's performance by providing 
        # foundational capabilities (and vice-versa).
        enrichment_a_to_b = self._calculate_enrichment(agent_a, agent_b)
        enrichment_b_to_a = self._calculate_enrichment(agent_b, agent_a)
        enrichment_score = (enrichment_a_to_b + enrichment_b_to_a) / 2.0

        # 2. Historical Collaboration Topology Patterns
        # Uses past outcomes to bias toward profiles that have historically 
        # demonstrated high synergy density.
        topology_score = self._compute_historical_topology_score(agent_a, agent_b)

        # 3. Structural Redundancy Penalty (Entropy-based)
        # If both agents provide near-identical capability vectors without enablement,
        # their marginal complementarity is reduced.
        overlap = self._calculate_capability_overlap(agent_a, agent_b)
        redundancy_penalty = 1.0 - (overlap * 0.4)

        # 4. Cross-Role Integration Multiplier
        # Agents with high integration depth amplify the structural links.
        integration_factor = (
            (agent_a.cross_role_integration_depth + agent_b.cross_role_integration_depth) / 2.0
        ) * 0.4 + 0.8

        # Composite Structural Complementarity
        complementarity = (
            (enrichment_score * 0.6) + (topology_score * 0.4)
        ) * redundancy_penalty * integration_factor

        return round(max(0.0, min(1.0, complementarity)), 6)

    def analyze_team_complementarity(self, team: List[CooperativeIntelligenceVector]) -> float:
        """
        Measures the collective complementarity density of a candidate team.
        """
        if len(team) < 2:
            return 0.0
        
        pairs = list(combinations(team, 2))
        total_score = sum(self.compute_structural_complementarity(a, b) for a, b in pairs)
        
        return round(total_score / len(pairs), 6)

    def detect_non_obvious_synergy(
        self,
        task: CooperativeContextTensor,
        agents: List[CooperativeIntelligenceVector]
    ) -> List[Dict[str, object]]:
        """
        Identifies agent pairs with high structural complementarity despite potentially
        low individual task alignment. This helps surface 'underdogs' who amplify 
        the team more than their individual stats suggest.
        """
        synergy_candidates = []
        
        for a, b in combinations(agents, 2):
            # Individual alignment scores (the 'obvious' part)
            align_a = CooperativeContextModel.compute_alignment_score(task, a.capability_profile)
            align_b = CooperativeContextModel.compute_alignment_score(task, b.capability_profile)
            max_align = max(align_a, align_b)
            
            # Structural complementarity (the 'hidden' potential)
            comp = self.compute_structural_complementarity(a, b)
            
            # Non-obviousness is high when complementarity > individual alignment
            synergy_potential = comp - (max_align * 0.4)
            
            if synergy_potential > 0.05:
                synergy_candidates.append({
                    "agents": (a.agent_id, b.agent_id),
                    "synergy_potential_score": round(synergy_potential, 6),
                    "structural_complementarity": comp,
                    "max_individual_alignment": round(max_align, 6)
                })
        
        # Sort by synergy potential descending
        return sorted(synergy_candidates, key=lambda x: x["synergy_potential_score"], reverse=True)

    def _calculate_enrichment(self, provider: CooperativeIntelligenceVector, consumer: CooperativeIntelligenceVector) -> float:
        """Computes how many 'foundation' capabilities agent A provides for agent B."""
        total_enrichment = 0.0
        
        for source, targets in self._adj.items():
            provider_strength = provider.capability_profile.get(source, 0.0)
            if provider_strength < 0.2:
                continue
            
            for target, weight in targets:
                consumer_strength = consumer.capability_profile.get(target, 0.0)
                # Enrichment occurs when the provider is strong in a dependency 
                # that the consumer relies on (indicated by consumer's own strength in target).
                total_enrichment += provider_strength * consumer_strength * weight
                
        # Normalize by consumer's total capability density to keep score bound
        consumer_density = sum(consumer.capability_profile.values()) + 1e-9
        return min(1.0, total_enrichment / consumer_density)

    def _calculate_capability_overlap(self, agent_a: CooperativeIntelligenceVector, agent_b: CooperativeIntelligenceVector) -> float:
        """Measures the Jaccard-like similarity of capability vectors."""
        caps_a = agent_a.capability_profile
        caps_b = agent_b.capability_profile
        
        all_caps = set(caps_a.keys()) | set(caps_b.keys())
        intersection_sum = 0.0
        union_sum = 0.0
        
        for cap in all_caps:
            val_a = caps_a.get(cap, 0.0)
            val_b = caps_b.get(cap, 0.0)
            intersection_sum += min(val_a, val_b)
            union_sum += max(val_a, val_b)
            
        return intersection_sum / (union_sum + 1e-9)

    def _compute_historical_topology_score(
        self,
        agent_a: CooperativeIntelligenceVector,
        agent_b: CooperativeIntelligenceVector
    ) -> float:
        """
        Analyzes historical collaboration patterns to determine if this pairing 
        aligns with past synergistic outcomes.
        """
        if not self.historical_records:
            return 0.5  # Neutral prior
        
        direct_synergies = []
        pattern_matches = 0
        pattern_synergy_total = 0.0
        
        ids = {agent_a.agent_id, agent_b.agent_id}
        
        for record in self.historical_records:
            record_agents = set(record.agents)
            overlap = ids.intersection(record_agents)
            
            # 1. Direct Topology Match: These two have worked together
            if len(overlap) == 2:
                if record.additive_expectation > 0:
                    density = (record.realized_impact - record.additive_expectation) / record.additive_expectation
                    direct_synergies.append(density)
            
            # 2. Pattern Match: Similar profiles (heuristically using IDs for simplicity here, 
            # but usually would involve embedding or signature similarity)
            # For this MVP, we treat direct matches with high weight.
            
        if direct_synergies:
            avg_synergy = sum(direct_synergies) / len(direct_synergies)
            # Map synergy density range [-0.5, 0.5] to [0.3, 0.9]
            return max(0.0, min(1.0, 0.6 + avg_synergy))
            
        return 0.5

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Dict, Optional

from synergy_forecast_simulator import (
    SynergyForecastSimulator, 
    AgentCounterfactualProfile, 
    SynergyForecast
)
from cooperative_intelligence import CooperativeIntelligenceVector
from cooperative_context_model import CooperativeContextTensor, CooperativeContextModel


@dataclass(frozen=True)
class TeamImpactProjection:
    """Projected outcome for a team configuration."""
    coalition_ids: Tuple[str, ...]
    expected_combined_impact: float
    expected_additive_impact: float
    marginal_synergy_amplification: float
    synergy_density: float
    probability_positive_impact: float
    uncertainty: float


@dataclass(frozen=True)
class DeltaImpactReport:
    """Report on the impact of adding/removing an agent."""
    agent_id: str
    delta_total_impact: float
    delta_synergy_impact: float
    structural_necessity_score: float
    is_additive_primary: bool


class CounterfactualTeamEvaluator:
    """
    Simulates projected task outcomes with different agent combinations.
    
    Determines structural necessity using marginal cooperative influence 
    consistency and synergy density projections, prioritizing functional 
    contribution over popularity metrics.
    """

    def __init__(self, simulator: SynergyForecastSimulator) -> None:
        self.simulator = simulator

    def _create_profiles(
        self, 
        task: CooperativeContextTensor, 
        agents: Sequence[CooperativeIntelligenceVector]
    ) -> List[AgentCounterfactualProfile]:
        """Maps intelligence vectors to counterfactual impact profiles for the simulator."""
        profiles = []
        for agent in agents:
            # Base capability alignment measures potential contribution
            base_alignment = CooperativeContextModel.compute_alignment_score(
                task, agent.capability_profile
            )
            
            # Marginal cooperative influence consistency moderates the expected output.
            # Agents who are inconsistent in their marginal value have lower projected impact.
            expected_impact = base_alignment * agent.marginal_cooperative_influence_consistency
            
            # Uncertainty is inversely tied to calibration. High calibration yields lower variance.
            uncertainty = (1.0 - agent.predictive_calibration_reliability) * 0.4 + 0.05
            
            profiles.append(AgentCounterfactualProfile(
                agent_id=agent.agent_id,
                expected_impact=max(0.01, expected_impact),
                uncertainty=uncertainty
            ))
        return profiles

    def evaluate_team(
        self, 
        task: CooperativeContextTensor, 
        team: List[CooperativeIntelligenceVector]
    ) -> TeamImpactProjection:
        """
        Runs a simulation to project the combined impact of a team.
        """
        agent_ids = [a.agent_id for a in team]
        profiles = self._create_profiles(task, team)
        
        if len(team) < 2:
            # Minimal team cannot generate synergy in this model
            total_impact = sum(p.expected_impact for p in profiles)
            return TeamImpactProjection(
                coalition_ids=tuple(sorted(agent_ids)),
                expected_combined_impact=total_impact,
                expected_additive_impact=total_impact,
                marginal_synergy_amplification=0.0,
                synergy_density=0.0,
                probability_positive_impact=1.0 if total_impact > 0 else 0.0,
                uncertainty=sum(p.uncertainty for p in profiles) / len(profiles) if profiles else 0.0
            )

        forecast = self.simulator.forecast(agent_ids, profiles)
        dist = forecast.projected_distribution

        return TeamImpactProjection(
            coalition_ids=tuple(sorted(agent_ids)),
            expected_combined_impact=dist.expected_combined_impact,
            expected_additive_impact=dist.expected_additive_impact,
            marginal_synergy_amplification=dist.mean_amplification,
            synergy_density=forecast.historical_synergy_density,
            probability_positive_impact=dist.probability_positive_amplification,
            uncertainty=dist.std_amplification
        )

    def calculate_delta_impact(
        self, 
        task: CooperativeContextTensor, 
        base_team: List[CooperativeIntelligenceVector], 
        agent: CooperativeIntelligenceVector
    ) -> DeltaImpactReport:
        """
        Calculates the shift in outcomes if 'agent' is added to 'base_team'.
        """
        baseline = self.evaluate_team(task, base_team)
        with_agent = self.evaluate_team(task, base_team + [agent])
        
        delta_total = with_agent.expected_combined_impact - baseline.expected_combined_impact
        delta_synergy = with_agent.marginal_synergy_amplification - baseline.marginal_synergy_amplification
        
        # Structural necessity is calculated using the absolute shift in synergy density
        # moderated by the agent's marginal influence consistency.
        necessity_score = self._compute_structural_necessity(
            agent, with_agent, baseline, delta_total, delta_synergy
        )
        
        return DeltaImpactReport(
            agent_id=agent.agent_id,
            delta_total_impact=round(delta_total, 6),
            delta_synergy_impact=round(delta_synergy, 6),
            structural_necessity_score=round(necessity_score, 6),
            is_additive_primary=abs(delta_synergy) < (abs(delta_total) * 0.15 + 1e-9)
        )

    def calculate_removal_impact(
        self,
        task: CooperativeContextTensor,
        team: List[CooperativeIntelligenceVector],
        agent: CooperativeIntelligenceVector
    ) -> DeltaImpactReport:
        """
        Calculates the shift in outcomes if 'agent' is removed from 'team'.
        """
        full_team_eval = self.evaluate_team(task, team)
        reduced_team = [a for a in team if a.agent_id != agent.agent_id]
        reduced_team_eval = self.evaluate_team(task, reduced_team)
        
        delta_total = reduced_team_eval.expected_combined_impact - full_team_eval.expected_combined_impact
        delta_synergy = reduced_team_eval.marginal_synergy_amplification - full_team_eval.marginal_synergy_amplification
        
        # For removal, we still want a positive necessity score representing the "gap" created.
        necessity_score = self._compute_structural_necessity(
            agent, full_team_eval, reduced_team_eval, abs(delta_total), abs(delta_synergy)
        )
        
        return DeltaImpactReport(
            agent_id=agent.agent_id,
            delta_total_impact=round(delta_total, 6),
            delta_synergy_impact=round(delta_synergy, 6),
            structural_necessity_score=round(necessity_score, 6),
            is_additive_primary=abs(delta_synergy) < (abs(delta_total) * 0.15 + 1e-9)
        )

    def _compute_structural_necessity(
        self,
        agent: CooperativeIntelligenceVector,
        with_agent_eval: TeamImpactProjection,
        without_agent_eval: TeamImpactProjection,
        abs_delta_total: float,
        abs_delta_synergy: float
    ) -> float:
        """
        Determines how essential an agent is to the team's structural integrity.
        
        Unlike centrality (which measures connection count), this measures 
        impact-weighted synergy stability.
        """
        # Feature 1: Marginal Cooperative Influence (MCI) scaling
        # How much of the total delta is sustained by the agent's consistency.
        influence_term = abs_delta_total * agent.marginal_cooperative_influence_consistency
        
        # Feature 2: Synergy Density Projection (SDP) amplification
        # Measures if the agent's presence increases the density of cooperative pairing.
        density_shift = abs(with_agent_eval.synergy_density - without_agent_eval.synergy_density)
        density_contribution = density_shift * with_agent_eval.expected_additive_impact
        
        # Combined Score: We want agents who not only add value but who make 
        # the entire team more "dense" in its cooperative potential.
        return max(0.0, influence_term + density_contribution)

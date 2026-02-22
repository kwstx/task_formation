from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DomainImpactType(str, Enum):
    """Enumeration of real-world impact domains that a task can optimize."""
    ECONOMIC = "economic"           # Directly impacts resource flow or market stability
    TECHNICAL = "technical"         # Impacts structural integrity or algorithmic efficiency
    SOCIAL = "social"               # Impacts agent trust clusters or cooperation network
    ECOLOGICAL = "ecological"       # Impacts system-wide health and resource sustainability
    SYNERGETIC = "synergetic"       # Indirectly impacts future cooperative potential


@dataclass(frozen=True)
class CooperativeContextTensor:
    """
    Structured context tensor representing a task's impact requirements.
    
    This tensor drives team formation by aligning agent profiles with 
    projected downstream causal effects rather than static skill matching.
    """
    domain_impact_type: DomainImpactType
    required_capability_vectors: Dict[str, float]  # Normalized weights for capability dimensions
    expected_downstream_causal_depth: int          # How many layers of causal effect are expected
    uncertainty_tolerance: float                   # [0, 1] Risk threshold for the task
    temporal_horizon: float                        # Estimated duration or impact decay constant
    
    # Metadata for transparency and debugging
    optimization_target: str = "real_world_impact"
    version: str = "1.0.0"

    def as_dict(self) -> Dict[str, object]:
        """Returns a dictionary representation for logging or API transfer."""
        return {
            "domain_impact_type": self.domain_impact_type.value,
            "required_capability_vectors": self.required_capability_vectors,
            "expected_downstream_causal_depth": self.expected_downstream_causal_depth,
            "uncertainty_tolerance": self.uncertainty_tolerance,
            "temporal_horizon": self.temporal_horizon,
            "optimization_target": self.optimization_target,
        }


class CooperativeContextModel:
    """
    Logic engine for encoding tasks into CooperativeContextTensors.
    
    The model ensures that tasks are represented in a way that prioritizes
    outcome alignment over mere availability.
    """

    @staticmethod
    def encode_task(
        impact_domain: DomainImpactType,
        capabilities: Dict[str, float],
        causal_depth: int,
        risk_threshold: float,
        horizon: float
    ) -> CooperativeContextTensor:
        """
        Encodes raw task parameters into a structured context tensor.
        
        Performs normalization on capability vectors to ensure consistency in 
        downstream influence projection.
        """
        # Ensure capability weights are normalized
        total_weight = sum(capabilities.values())
        normalized_capabilities = (
            {k: v / total_weight for k, v in capabilities.items()}
            if total_weight > 0
            else capabilities
        )

        return CooperativeContextTensor(
            domain_impact_type=impact_domain,
            required_capability_vectors=normalized_capabilities,
            expected_downstream_causal_depth=max(1, causal_depth),
            uncertainty_tolerance=max(0.0, min(1.0, risk_threshold)),
            temporal_horizon=max(0.1, horizon)
        )

    @staticmethod
    def compute_alignment_score(
        task_tensor: CooperativeContextTensor,
        agent_capability_profile: Dict[str, float]
    ) -> float:
        """
        Computes a similarity score between a task's requirements and an agent's profile.
        
        Unlike simple keyword matching, this uses the capability vectors to measure
        the density of synergy projection.
        """
        score = 0.0
        for capability, weight in task_tensor.required_capability_vectors.items():
            agent_strength = agent_capability_profile.get(capability, 0.0)
            score += weight * agent_strength
        
        return score

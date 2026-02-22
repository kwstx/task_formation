from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import erf, exp, log, sqrt
from random import Random
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class HistoricalCoalitionRecord:
    """Observed coalition outcome used to calibrate historical synergy density."""

    agents: Tuple[str, ...]
    additive_expectation: float
    realized_impact: float


@dataclass(frozen=True)
class AgentCounterfactualProfile:
    """Counterfactual impact estimate for a candidate agent."""

    agent_id: str
    expected_impact: float
    uncertainty: float
    trust_coefficient: float = 1.0
    predictive_calibration_stability: float = 1.0


@dataclass(frozen=True)
class ProbabilisticSynergyDistribution:
    """Distribution summary for projected coalition synergy."""

    sample_count: int
    mean_amplification: float
    std_amplification: float
    p05: float
    p50: float
    p95: float
    probability_positive_amplification: float
    expected_combined_impact: float
    expected_additive_impact: float


@dataclass(frozen=True)
class SynergyForecast:
    """Forecast output for a specific candidate coalition."""

    coalition: Tuple[str, ...]
    historical_synergy_density: float
    projected_distribution: ProbabilisticSynergyDistribution
    team_prediction_reliability: float = 0.0
    trust_weight_entropy: float = 0.0
    trust_weight_max_share: float = 1.0
    structural_formation_weight: float = 1.0
    stability_penalty_factor: float = 1.0
    stability_score: float = 1.0
    instability_risk: float = 0.0


@dataclass(frozen=True)
class StabilityAssessment:
    """Stability diagnostics for a candidate coalition."""

    negotiation_convergence_time: float
    collaboration_variance: float
    conflict_resolution_frequency: float
    instability_risk: float
    stability_score: float
    penalty_factor: float


class StabilityOptimizationLayer:
    """
    Penalizes coalitions that exhibit unstable collaboration dynamics.

    The layer estimates instability from three drivers:
    - Negotiation convergence time (higher is less stable)
    - Collaboration variance (higher is less stable)
    - Conflict resolution frequency (higher is less stable)
    """

    def __init__(
        self,
        *,
        convergence_weight: float = 0.40,
        variance_weight: float = 0.35,
        conflict_weight: float = 0.25,
        max_penalty_share: float = 0.65,
        baseline_convergence_time: float = 1.0,
        baseline_collaboration_variance: float = 1.0,
        baseline_conflict_frequency: float = 1.0,
    ) -> None:
        total_weight = max(1e-9, convergence_weight + variance_weight + conflict_weight)
        self._convergence_weight = max(0.0, convergence_weight) / total_weight
        self._variance_weight = max(0.0, variance_weight) / total_weight
        self._conflict_weight = max(0.0, conflict_weight) / total_weight
        self._max_penalty_share = max(0.0, min(0.95, max_penalty_share))
        self._baseline_convergence_time = max(1e-6, baseline_convergence_time)
        self._baseline_collaboration_variance = max(1e-6, baseline_collaboration_variance)
        self._baseline_conflict_frequency = max(1e-6, baseline_conflict_frequency)

    def evaluate(
        self,
        *,
        negotiation_convergence_time: float,
        collaboration_variance: float,
        conflict_resolution_frequency: float,
    ) -> StabilityAssessment:
        convergence_ratio = max(0.0, negotiation_convergence_time) / self._baseline_convergence_time
        variance_ratio = max(0.0, collaboration_variance) / self._baseline_collaboration_variance
        conflict_ratio = max(0.0, conflict_resolution_frequency) / self._baseline_conflict_frequency

        instability_risk = self._weighted_sigmoid_risk(
            convergence_ratio=convergence_ratio,
            variance_ratio=variance_ratio,
            conflict_ratio=conflict_ratio,
        )
        stability_score = max(0.0, 1.0 - instability_risk)
        penalty_factor = max(0.0, 1.0 - (self._max_penalty_share * instability_risk))

        return StabilityAssessment(
            negotiation_convergence_time=max(0.0, negotiation_convergence_time),
            collaboration_variance=max(0.0, collaboration_variance),
            conflict_resolution_frequency=max(0.0, conflict_resolution_frequency),
            instability_risk=instability_risk,
            stability_score=stability_score,
            penalty_factor=penalty_factor,
        )

    def _weighted_sigmoid_risk(
        self,
        *,
        convergence_ratio: float,
        variance_ratio: float,
        conflict_ratio: float,
    ) -> float:
        convergence_risk = self._soft_threshold(convergence_ratio)
        variance_risk = self._soft_threshold(variance_ratio)
        conflict_risk = self._soft_threshold(conflict_ratio)
        risk = (
            self._convergence_weight * convergence_risk
            + self._variance_weight * variance_risk
            + self._conflict_weight * conflict_risk
        )
        return self._clamp01(risk)

    @staticmethod
    def _soft_threshold(ratio: float) -> float:
        # Ratios above 1 quickly increase risk, but remain bounded in [0, 1].
        return 1.0 / (1.0 + exp(-(ratio - 1.0) * 1.8))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))


class SynergyLearningLoop:
    """
    Learns structural collaboration priors from realized-vs-projected outcomes.

    Overperforming coalitions and agent pairs are up-weighted for future formation
    sampling; underperforming structures are dampened.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.45,
        pair_credit_share: float = 0.60,
        min_weight: float = 0.35,
        max_weight: float = 3.00,
        max_relative_error: float = 0.75,
    ) -> None:
        self._learning_rate = max(0.0, learning_rate)
        self._pair_credit_share = max(0.0, min(1.0, pair_credit_share))
        self._min_weight = max(1e-6, min_weight)
        self._max_weight = max(self._min_weight, max_weight)
        self._max_relative_error = max(1e-6, max_relative_error)
        self._coalition_weights: Dict[Tuple[str, ...], float] = {}
        self._pair_weights: Dict[Tuple[str, str], float] = {}

    def update_from_outcome(
        self,
        coalition: Sequence[str],
        *,
        predicted_synergy: float,
        realized_synergy: float,
    ) -> float:
        """
        Updates coalition and pair weights from one realized outcome.

        Returns:
            The new structural weight for this coalition after the update.
        """
        key = tuple(sorted(coalition))
        if len(key) < 2:
            return 1.0

        baseline = max(abs(predicted_synergy), 1.0)
        relative_error = (realized_synergy - predicted_synergy) / baseline
        clipped_error = max(-self._max_relative_error, min(self._max_relative_error, relative_error))

        coalition_factor = exp(self._learning_rate * clipped_error * (1.0 - self._pair_credit_share))
        pair_factor = exp(self._learning_rate * clipped_error * self._pair_credit_share)

        current_coalition_weight = self._coalition_weights.get(key, 1.0)
        self._coalition_weights[key] = self._clamp_weight(current_coalition_weight * coalition_factor)

        for pair in combinations(key, 2):
            current_pair_weight = self._pair_weights.get(pair, 1.0)
            self._pair_weights[pair] = self._clamp_weight(current_pair_weight * pair_factor)

        return self.structural_weight(key)

    def structural_weight(self, coalition: Sequence[str]) -> float:
        """
        Structural prior for a coalition, combining coalition and pair-level memory.
        """
        key = tuple(sorted(coalition))
        if len(key) < 2:
            return 1.0

        coalition_weight = self._coalition_weights.get(key, 1.0)
        pair_weights = [self._pair_weights.get(pair, 1.0) for pair in combinations(key, 2)]
        if not pair_weights:
            return coalition_weight

        pair_mean = sum(pair_weights) / len(pair_weights)
        return self._clamp_weight(coalition_weight * pair_mean)

    def adjust_formation_probabilities(
        self,
        coalitions: Sequence[Sequence[str]],
        base_probabilities: Sequence[float],
    ) -> List[float]:
        """
        Reweights formation probabilities using learned structural priors.
        """
        if not coalitions:
            return []
        if len(coalitions) != len(base_probabilities):
            raise ValueError("coalitions and base_probabilities must have the same length")

        normalized = self._normalize_probabilities(base_probabilities)
        weighted = [
            normalized[i] * self.structural_weight(coalitions[i])
            for i in range(len(coalitions))
        ]
        total = sum(weighted)
        if total <= 1e-12:
            return normalized
        return [p / total for p in weighted]

    def _normalize_probabilities(self, probabilities: Sequence[float]) -> List[float]:
        if not probabilities:
            return []
        total = sum(max(0.0, p) for p in probabilities)
        if total <= 1e-12:
            uniform = 1.0 / len(probabilities)
            return [uniform for _ in probabilities]
        return [max(0.0, p) / total for p in probabilities]

    def _clamp_weight(self, value: float) -> float:
        return max(self._min_weight, min(self._max_weight, value))


class SynergyForecastSimulator:
    """
    Estimates cooperative impact for candidate coalitions with a probabilistic model.

    The simulator calibrates historical synergy density from prior coalition outcomes,
    then uses counterfactual Monte Carlo draws to estimate a probability distribution
    over amplification beyond additive expectation.
    """

    def __init__(
        self,
        historical_records: Sequence[HistoricalCoalitionRecord],
        *,
        simulation_draws: int = 5000,
        random_seed: int | None = None,
        min_entropy_ratio: float = 0.72,
        synergy_learning_loop: SynergyLearningLoop | None = None,
        stability_optimization_layer: StabilityOptimizationLayer | None = None,
    ) -> None:
        if simulation_draws <= 0:
            raise ValueError("simulation_draws must be > 0")

        self._historical_records = list(historical_records)
        self._simulation_draws = simulation_draws
        self._rng = Random(random_seed)
        self._min_entropy_ratio = max(0.0, min(1.0, min_entropy_ratio))
        self._synergy_learning_loop = synergy_learning_loop
        self._stability_optimization_layer = stability_optimization_layer

    def forecast(
        self,
        candidate_agents: Sequence[str],
        counterfactual_profiles: Sequence[AgentCounterfactualProfile],
    ) -> SynergyForecast:
        if len(candidate_agents) < 2:
            raise ValueError("candidate_agents must include at least two agents")

        coalition = tuple(sorted(candidate_agents))
        profile_map = self._index_profiles(counterfactual_profiles)
        self._ensure_all_profiles_present(coalition, profile_map)
        trust_weights, trust_entropy = self._trust_adjusted_weights(coalition, profile_map)
        additive_mean, additive_sigma, team_reliability = self._coalition_additive_parameters(
            coalition,
            profile_map,
            trust_weights,
        )

        density_mean, density_std = self._historical_synergy_density_distribution(coalition)
        pair_count = len(list(combinations(coalition, 2)))
        trust_max_share = max(trust_weights.values()) if trust_weights else 1.0
        structural_weight = (
            self._synergy_learning_loop.structural_weight(coalition)
            if self._synergy_learning_loop is not None
            else 1.0
        )
        stability_assessment = self._evaluate_stability(
            coalition,
            profile_map,
            additive_mean=additive_mean,
            additive_sigma=additive_sigma,
        )

        amplification_samples: List[float] = []
        combined_samples: List[float] = []

        # Stable calibration increases confidence in propagation dynamics.
        density_sigma = density_std * max(0.55, 1.0 - (0.35 * team_reliability))
        propagation_multiplier = 0.85 + (0.30 * team_reliability)

        for _ in range(self._simulation_draws):
            additive_draw = self._rng.gauss(additive_mean, additive_sigma)
            density_draw = self._rng.gauss(density_mean, density_sigma)
            amplification_draw = (
                additive_draw
                * density_draw
                * pair_count
                * propagation_multiplier
                * structural_weight
                * stability_assessment.penalty_factor
            )
            combined_draw = additive_draw + amplification_draw

            amplification_samples.append(amplification_draw)
            combined_samples.append(combined_draw)

        distribution = self._summarize_distribution(
            amplification_samples=amplification_samples,
            combined_samples=combined_samples,
            additive_expectation=additive_mean,
        )

        return SynergyForecast(
            coalition=coalition,
            historical_synergy_density=density_mean,
            projected_distribution=distribution,
            team_prediction_reliability=team_reliability,
            trust_weight_entropy=trust_entropy,
            trust_weight_max_share=trust_max_share,
            structural_formation_weight=structural_weight,
            stability_penalty_factor=stability_assessment.penalty_factor,
            stability_score=stability_assessment.stability_score,
            instability_risk=stability_assessment.instability_risk,
        )

    def register_realized_outcome(
        self,
        coalition: Sequence[str],
        *,
        predicted_combined_impact: float,
        realized_combined_impact: float,
        predicted_additive_impact: float,
    ) -> float:
        """
        Feeds realized outcomes back into the learning loop.

        Returns:
            Updated structural weight for this coalition.
        """
        if self._synergy_learning_loop is None:
            return 1.0

        predicted_synergy = predicted_combined_impact - predicted_additive_impact
        realized_synergy = realized_combined_impact - predicted_additive_impact
        return self._synergy_learning_loop.update_from_outcome(
            coalition,
            predicted_synergy=predicted_synergy,
            realized_synergy=realized_synergy,
        )

    def adjust_structural_formation_probabilities(
        self,
        coalitions: Sequence[Sequence[str]],
        base_probabilities: Sequence[float],
    ) -> List[float]:
        """
        Applies learned structural priors to formation probabilities.
        """
        if self._synergy_learning_loop is None:
            return self._normalize_probabilities(base_probabilities)
        return self._synergy_learning_loop.adjust_formation_probabilities(
            coalitions,
            base_probabilities,
        )

    @staticmethod
    def _normalize_probabilities(probabilities: Sequence[float]) -> List[float]:
        if not probabilities:
            return []
        total = sum(max(0.0, p) for p in probabilities)
        if total <= 1e-12:
            uniform = 1.0 / len(probabilities)
            return [uniform for _ in probabilities]
        return [max(0.0, p) / total for p in probabilities]

    def _index_profiles(
        self,
        profiles: Sequence[AgentCounterfactualProfile],
    ) -> Dict[str, AgentCounterfactualProfile]:
        profile_map = {profile.agent_id: profile for profile in profiles}
        if len(profile_map) != len(profiles):
            raise ValueError("counterfactual_profiles contains duplicate agent_id entries")
        return profile_map

    def _ensure_all_profiles_present(
        self,
        coalition: Sequence[str],
        profile_map: Mapping[str, AgentCounterfactualProfile],
    ) -> None:
        for agent in coalition:
            if agent not in profile_map:
                raise ValueError(f"missing counterfactual profile for agent '{agent}'")

    def _coalition_additive_parameters(
        self,
        coalition: Iterable[str],
        profile_map: Mapping[str, AgentCounterfactualProfile],
        trust_weights: Mapping[str, float],
    ) -> Tuple[float, float, float]:
        additive_mean = 0.0
        additive_var = 0.0
        weighted_reliability = 0.0

        for agent in coalition:
            if agent not in profile_map:
                raise ValueError(f"missing counterfactual profile for agent '{agent}'")
            profile = profile_map[agent]
            trust = self._clamp01(profile.trust_coefficient)
            stability = self._clamp01(profile.predictive_calibration_stability)
            effective_trust = trust * (0.5 + 0.5 * stability)
            influence_weight = trust_weights.get(agent, 0.0)

            # Weight influence projections by predictive calibration stability.
            influence_projection = profile.expected_impact * (0.85 + 0.30 * stability)
            influence_projection *= 0.90 + (0.20 * influence_weight)
            additive_mean += influence_projection

            uncertainty_scale = max(0.35, 1.0 - (0.35 * effective_trust))
            additive_var += (max(profile.uncertainty, 0.0) * uncertainty_scale) ** 2
            weighted_reliability += influence_weight * effective_trust

        return additive_mean, sqrt(additive_var), self._clamp01(weighted_reliability)

    def _trust_adjusted_weights(
        self,
        coalition: Sequence[str],
        profile_map: Mapping[str, AgentCounterfactualProfile],
    ) -> Tuple[Dict[str, float], float]:
        if not coalition:
            return {}, 0.0

        raw_weights: Dict[str, float] = {}
        for agent in coalition:
            profile = profile_map[agent]
            trust = self._clamp01(profile.trust_coefficient)
            stability = self._clamp01(profile.predictive_calibration_stability)
            raw_weights[agent] = trust * (0.20 + (0.80 * stability))

        total = sum(raw_weights.values())
        if total <= 1e-12:
            uniform = 1.0 / len(coalition)
            weights = {agent: uniform for agent in coalition}
            return weights, self._entropy(weights.values())

        weights = {agent: value / total for agent, value in raw_weights.items()}

        entropy = self._entropy(weights.values())
        max_entropy = log(len(weights)) if len(weights) > 1 else 1.0
        entropy_ratio = entropy / max_entropy if max_entropy > 1e-12 else 1.0
        if entropy_ratio >= self._min_entropy_ratio or len(weights) <= 1:
            return weights, entropy

        # Entropy blend prevents very high-trust agents from monopolizing formation.
        deficit = self._min_entropy_ratio - entropy_ratio
        blend = min(1.0, deficit / max(self._min_entropy_ratio, 1e-9))
        uniform = 1.0 / len(weights)
        adjusted = {
            agent: ((1.0 - blend) * weight) + (blend * uniform)
            for agent, weight in weights.items()
        }
        return adjusted, self._entropy(adjusted.values())

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _entropy(values: Iterable[float]) -> float:
        entropy = 0.0
        for value in values:
            if value > 1e-12:
                entropy -= value * log(value)
        return entropy

    def _evaluate_stability(
        self,
        coalition: Sequence[str],
        profile_map: Mapping[str, AgentCounterfactualProfile],
        *,
        additive_mean: float,
        additive_sigma: float,
    ) -> StabilityAssessment:
        if self._stability_optimization_layer is None:
            return StabilityAssessment(
                negotiation_convergence_time=0.0,
                collaboration_variance=0.0,
                conflict_resolution_frequency=0.0,
                instability_risk=0.0,
                stability_score=1.0,
                penalty_factor=1.0,
            )

        pair_count = max(1, len(list(combinations(coalition, 2))))
        reliability = self._coalition_reliability_estimate(coalition, profile_map)
        normalized_variance = additive_sigma / max(1.0, abs(additive_mean))
        negotiation_convergence_time = pair_count * (1.2 + (1.8 * (1.0 - reliability)))
        collaboration_variance = normalized_variance * (1.0 + (1.5 * (1.0 - reliability)))
        conflict_resolution_frequency = pair_count * (0.4 + (0.9 * (1.0 - reliability)) + (0.8 * normalized_variance))

        return self._stability_optimization_layer.evaluate(
            negotiation_convergence_time=negotiation_convergence_time,
            collaboration_variance=collaboration_variance,
            conflict_resolution_frequency=conflict_resolution_frequency,
        )

    def _coalition_reliability_estimate(
        self,
        coalition: Sequence[str],
        profile_map: Mapping[str, AgentCounterfactualProfile],
    ) -> float:
        if not coalition:
            return 1.0

        scores = []
        for agent in coalition:
            profile = profile_map[agent]
            trust = self._clamp01(profile.trust_coefficient)
            stability = self._clamp01(profile.predictive_calibration_stability)
            scores.append(0.5 * trust + 0.5 * stability)
        return self._clamp01(sum(scores) / len(scores))

    def _historical_synergy_density_distribution(
        self,
        coalition: Sequence[str],
    ) -> Tuple[float, float]:
        densities: List[float] = []
        coalition_set = set(coalition)

        for record in self._historical_records:
            if len(record.agents) < 2 or record.additive_expectation <= 0:
                continue

            overlap = len(coalition_set.intersection(record.agents))
            if overlap < 2:
                continue

            pair_count = len(list(combinations(record.agents, 2)))
            if pair_count == 0:
                continue

            amplification = record.realized_impact - record.additive_expectation
            densities.append(amplification / (record.additive_expectation * pair_count))

        if not densities:
            # If no similar historical coalition exists, use weakly-informative prior.
            return 0.0, 0.15

        density_mean = mean(densities)
        density_std = pstdev(densities) if len(densities) > 1 else abs(density_mean) * 0.25 + 0.05
        density_std = max(density_std, 0.02)
        return density_mean, density_std

    def _summarize_distribution(
        self,
        *,
        amplification_samples: Sequence[float],
        combined_samples: Sequence[float],
        additive_expectation: float,
    ) -> ProbabilisticSynergyDistribution:
        sorted_amp = sorted(amplification_samples)
        sample_count = len(sorted_amp)

        def quantile(q: float) -> float:
            idx = max(0, min(sample_count - 1, int(round(q * (sample_count - 1)))))
            return sorted_amp[idx]

        amp_mean = mean(amplification_samples)
        amp_std = pstdev(amplification_samples) if sample_count > 1 else 0.0

        pos_prob_empirical = sum(1 for x in amplification_samples if x > 0) / sample_count
        if amp_std > 1e-9:
            z = (0.0 - amp_mean) / amp_std
            pos_prob_model = 1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))
            pos_prob = 0.5 * pos_prob_empirical + 0.5 * pos_prob_model
        else:
            pos_prob = 1.0 if amp_mean > 0 else 0.0

        return ProbabilisticSynergyDistribution(
            sample_count=sample_count,
            mean_amplification=amp_mean,
            std_amplification=amp_std,
            p05=quantile(0.05),
            p50=quantile(0.50),
            p95=quantile(0.95),
            probability_positive_amplification=max(0.0, min(1.0, pos_prob)),
            expected_combined_impact=additive_expectation + amp_mean,
            expected_additive_impact=additive_expectation,
        )

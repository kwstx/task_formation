from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import erf, sqrt
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
    ) -> None:
        if simulation_draws <= 0:
            raise ValueError("simulation_draws must be > 0")

        self._historical_records = list(historical_records)
        self._simulation_draws = simulation_draws
        self._rng = Random(random_seed)

    def forecast(
        self,
        candidate_agents: Sequence[str],
        counterfactual_profiles: Sequence[AgentCounterfactualProfile],
    ) -> SynergyForecast:
        if len(candidate_agents) < 2:
            raise ValueError("candidate_agents must include at least two agents")

        coalition = tuple(sorted(candidate_agents))
        profile_map = self._index_profiles(counterfactual_profiles)
        additive_mean, additive_sigma = self._coalition_additive_parameters(
            coalition,
            profile_map,
        )

        density_mean, density_std = self._historical_synergy_density_distribution(coalition)
        pair_count = len(list(combinations(coalition, 2)))

        amplification_samples: List[float] = []
        combined_samples: List[float] = []

        for _ in range(self._simulation_draws):
            additive_draw = self._rng.gauss(additive_mean, additive_sigma)
            density_draw = self._rng.gauss(density_mean, density_std)
            amplification_draw = additive_draw * density_draw * pair_count
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
        )

    def _index_profiles(
        self,
        profiles: Sequence[AgentCounterfactualProfile],
    ) -> Dict[str, AgentCounterfactualProfile]:
        profile_map = {profile.agent_id: profile for profile in profiles}
        if len(profile_map) != len(profiles):
            raise ValueError("counterfactual_profiles contains duplicate agent_id entries")
        return profile_map

    def _coalition_additive_parameters(
        self,
        coalition: Iterable[str],
        profile_map: Mapping[str, AgentCounterfactualProfile],
    ) -> Tuple[float, float]:
        additive_mean = 0.0
        additive_var = 0.0

        for agent in coalition:
            if agent not in profile_map:
                raise ValueError(f"missing counterfactual profile for agent '{agent}'")
            profile = profile_map[agent]
            additive_mean += profile.expected_impact
            additive_var += max(profile.uncertainty, 0.0) ** 2

        return additive_mean, sqrt(additive_var)

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
            expected_combined_impact=mean(combined_samples),
            expected_additive_impact=additive_expectation,
        )

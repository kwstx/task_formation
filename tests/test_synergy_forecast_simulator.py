from math import log

from synergy_forecast_simulator import (
    AgentCounterfactualProfile,
    HistoricalCoalitionRecord,
    SynergyForecastSimulator,
)


def _build_profiles():
    return [
        AgentCounterfactualProfile("alpha", 10.0, 1.0),
        AgentCounterfactualProfile("beta", 8.0, 1.2),
        AgentCounterfactualProfile("gamma", 7.0, 0.9),
    ]


def _build_history():
    return [
        HistoricalCoalitionRecord(("alpha", "beta"), additive_expectation=17.0, realized_impact=20.4),
        HistoricalCoalitionRecord(("beta", "gamma"), additive_expectation=15.0, realized_impact=16.5),
        HistoricalCoalitionRecord(("alpha", "gamma"), additive_expectation=16.0, realized_impact=18.4),
        HistoricalCoalitionRecord(("alpha", "beta", "gamma"), additive_expectation=25.0, realized_impact=31.5),
    ]


def test_forecast_returns_probabilistic_distribution():
    sim = SynergyForecastSimulator(_build_history(), simulation_draws=3000, random_seed=7)
    forecast = sim.forecast(["alpha", "beta", "gamma"], _build_profiles())
    dist = forecast.projected_distribution

    assert dist.sample_count == 3000
    assert dist.p05 <= dist.p50 <= dist.p95
    assert 0.0 <= dist.probability_positive_amplification <= 1.0


def test_expected_combined_is_additive_plus_amplification():
    sim = SynergyForecastSimulator(_build_history(), simulation_draws=3500, random_seed=11)
    forecast = sim.forecast(["alpha", "beta", "gamma"], _build_profiles())
    dist = forecast.projected_distribution

    lhs = dist.expected_combined_impact
    rhs = dist.expected_additive_impact + dist.mean_amplification
    assert abs(lhs - rhs) < 1e-9


def test_missing_profile_raises_clear_error():
    sim = SynergyForecastSimulator(_build_history(), simulation_draws=500, random_seed=1)

    try:
        sim.forecast(
            ["alpha", "beta", "gamma"],
            [
                AgentCounterfactualProfile("alpha", 10.0, 1.0),
                AgentCounterfactualProfile("beta", 8.0, 1.2),
            ],
        )
        raise AssertionError("Expected ValueError for missing profile")
    except ValueError as exc:
        assert "missing counterfactual profile" in str(exc)


def test_high_trust_agents_improve_prediction_reliability():
    sim = SynergyForecastSimulator(_build_history(), simulation_draws=3500, random_seed=19)

    high_trust_profiles = [
        AgentCounterfactualProfile("alpha", 10.0, 1.0, trust_coefficient=0.95, predictive_calibration_stability=0.95),
        AgentCounterfactualProfile("beta", 8.0, 1.2, trust_coefficient=0.90, predictive_calibration_stability=0.90),
        AgentCounterfactualProfile("gamma", 7.0, 0.9, trust_coefficient=0.85, predictive_calibration_stability=0.92),
    ]
    low_trust_profiles = [
        AgentCounterfactualProfile("alpha", 10.0, 1.0, trust_coefficient=0.25, predictive_calibration_stability=0.25),
        AgentCounterfactualProfile("beta", 8.0, 1.2, trust_coefficient=0.20, predictive_calibration_stability=0.30),
        AgentCounterfactualProfile("gamma", 7.0, 0.9, trust_coefficient=0.18, predictive_calibration_stability=0.22),
    ]

    high = sim.forecast(["alpha", "beta", "gamma"], high_trust_profiles)
    low = sim.forecast(["alpha", "beta", "gamma"], low_trust_profiles)

    assert high.team_prediction_reliability > low.team_prediction_reliability
    assert high.projected_distribution.std_amplification < low.projected_distribution.std_amplification


def test_entropy_constraint_prevents_trust_monopoly():
    sim = SynergyForecastSimulator(_build_history(), simulation_draws=3000, random_seed=23, min_entropy_ratio=0.75)
    skewed_profiles = [
        AgentCounterfactualProfile("alpha", 10.0, 1.0, trust_coefficient=1.0, predictive_calibration_stability=1.0),
        AgentCounterfactualProfile("beta", 8.0, 1.2, trust_coefficient=0.0, predictive_calibration_stability=0.0),
        AgentCounterfactualProfile("gamma", 7.0, 0.9, trust_coefficient=0.0, predictive_calibration_stability=0.0),
    ]

    forecast = sim.forecast(["alpha", "beta", "gamma"], skewed_profiles)
    theoretical_max_entropy = log(3)
    entropy_ratio = forecast.trust_weight_entropy / theoretical_max_entropy

    assert forecast.trust_weight_max_share < 1.0
    assert entropy_ratio >= 0.70

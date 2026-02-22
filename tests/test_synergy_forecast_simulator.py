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

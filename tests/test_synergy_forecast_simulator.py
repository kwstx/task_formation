from math import log

from synergy_forecast_simulator import (
    AgentCounterfactualProfile,
    HistoricalCoalitionRecord,
    StabilityOptimizationLayer,
    SynergyLearningLoop,
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
    high_cv = high.projected_distribution.std_amplification / abs(high.projected_distribution.mean_amplification)
    low_cv = low.projected_distribution.std_amplification / abs(low.projected_distribution.mean_amplification)
    assert high_cv < low_cv


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


def test_synergy_learning_loop_adjusts_formation_probabilities():
    learning_loop = SynergyLearningLoop(learning_rate=0.9)
    coalitions = [("alpha", "beta"), ("alpha", "gamma")]
    base = [0.5, 0.5]

    learning_loop.update_from_outcome(
        ("alpha", "beta"),
        predicted_synergy=2.0,
        realized_synergy=4.0,
    )
    learning_loop.update_from_outcome(
        ("alpha", "gamma"),
        predicted_synergy=2.0,
        realized_synergy=1.0,
    )

    adjusted = learning_loop.adjust_formation_probabilities(coalitions, base)

    assert abs(sum(adjusted) - 1.0) < 1e-12
    assert adjusted[0] > base[0]
    assert adjusted[1] < base[1]


def test_simulator_learning_updates_future_structural_weighting():
    learning_loop = SynergyLearningLoop(learning_rate=0.8)
    candidate = ["alpha", "beta", "gamma"]
    profiles = _build_profiles()

    baseline_sim = SynergyForecastSimulator(
        _build_history(),
        simulation_draws=3000,
        random_seed=29,
        synergy_learning_loop=learning_loop,
    )
    baseline = baseline_sim.forecast(candidate, profiles)

    baseline_sim.register_realized_outcome(
        candidate,
        predicted_combined_impact=baseline.projected_distribution.expected_combined_impact,
        realized_combined_impact=baseline.projected_distribution.expected_combined_impact + 3.0,
        predicted_additive_impact=baseline.projected_distribution.expected_additive_impact,
    )

    replay_sim = SynergyForecastSimulator(
        _build_history(),
        simulation_draws=3000,
        random_seed=29,
        synergy_learning_loop=learning_loop,
    )
    learned = replay_sim.forecast(candidate, profiles)

    assert learned.structural_formation_weight > 1.0
    assert learned.projected_distribution.mean_amplification > baseline.projected_distribution.mean_amplification


def test_stability_layer_penalizes_unstable_team_configurations():
    stability_layer = StabilityOptimizationLayer(
        baseline_convergence_time=3.0,
        baseline_collaboration_variance=0.08,
        baseline_conflict_frequency=1.2,
        max_penalty_share=0.70,
    )

    stable_sim = SynergyForecastSimulator(
        _build_history(),
        simulation_draws=3500,
        random_seed=31,
        stability_optimization_layer=stability_layer,
    )
    unstable_sim = SynergyForecastSimulator(
        _build_history(),
        simulation_draws=3500,
        random_seed=31,
        stability_optimization_layer=stability_layer,
    )

    stable_profiles = [
        AgentCounterfactualProfile("alpha", 13.0, 0.6, trust_coefficient=0.96, predictive_calibration_stability=0.95),
        AgentCounterfactualProfile("beta", 12.0, 0.5, trust_coefficient=0.94, predictive_calibration_stability=0.94),
        AgentCounterfactualProfile("gamma", 11.0, 0.6, trust_coefficient=0.95, predictive_calibration_stability=0.93),
    ]
    unstable_profiles = [
        AgentCounterfactualProfile("alpha", 16.0, 3.4, trust_coefficient=0.20, predictive_calibration_stability=0.18),
        AgentCounterfactualProfile("beta", 15.0, 3.2, trust_coefficient=0.18, predictive_calibration_stability=0.20),
        AgentCounterfactualProfile("gamma", 14.0, 3.1, trust_coefficient=0.22, predictive_calibration_stability=0.19),
    ]

    stable = stable_sim.forecast(["alpha", "beta", "gamma"], stable_profiles)
    unstable = unstable_sim.forecast(["alpha", "beta", "gamma"], unstable_profiles)

    assert stable.stability_penalty_factor > unstable.stability_penalty_factor
    assert stable.instability_risk < unstable.instability_risk
    assert unstable.stability_score < stable.stability_score
    assert unstable.projected_distribution.expected_additive_impact > stable.projected_distribution.expected_additive_impact
    assert unstable.projected_distribution.mean_amplification < stable.projected_distribution.mean_amplification

import unittest
from synergy_forecast_simulator import SynergyForecastSimulator, HistoricalCoalitionRecord
from cooperative_intelligence import CooperativeIntelligenceVector
from cooperative_context_model import CooperativeContextModel, DomainImpactType
from counterfactual_team_evaluator import CounterfactualTeamEvaluator

class TestCounterfactualTeamEvaluator(unittest.TestCase):
    def setUp(self):
        # Create some historical records to provide synergy density.
        # Coalition (A, B) has high synergy.
        self.records = [
            HistoricalCoalitionRecord(
                agents=("A", "B"),
                additive_expectation=1.0,
                realized_impact=1.5  # 50% synergy amplification
            ),
            HistoricalCoalitionRecord(
                agents=("A", "B", "C"),
                additive_expectation=2.0,
                realized_impact=3.0  # 50% synergy amplification
            )
        ]
        self.simulator = SynergyForecastSimulator(self.records, simulation_draws=1000, random_seed=42)
        self.evaluator = CounterfactualTeamEvaluator(self.simulator)
        
        self.task = CooperativeContextModel.encode_task(
            impact_domain=DomainImpactType.TECHNICAL,
            capabilities={"coding": 0.5, "design": 0.5},
            causal_depth=3,
            risk_threshold=0.2,
            horizon=5.0
        )
        
        self.agent_a = CooperativeIntelligenceVector(
            agent_id="A",
            predictive_calibration_reliability=0.9,
            marginal_cooperative_influence_consistency=0.9,
            cross_role_integration_depth=0.8,
            capability_profile={"coding": 1.0, "design": 0.2}
        )
        
        self.agent_b = CooperativeIntelligenceVector(
            agent_id="B",
            predictive_calibration_reliability=0.8,
            marginal_cooperative_influence_consistency=0.8,
            cross_role_integration_depth=0.7,
            capability_profile={"coding": 0.2, "design": 1.0}
        )
        
        self.agent_c = CooperativeIntelligenceVector(
            agent_id="C",
            predictive_calibration_reliability=0.5,
            marginal_cooperative_influence_consistency=0.4,
            cross_role_integration_depth=0.3,
            capability_profile={"coding": 0.5, "design": 0.5}
        )

    def test_evaluate_team_synergy(self):
        team = [self.agent_a, self.agent_b]
        projection = self.evaluator.evaluate_team(self.task, team)
        
        self.assertEqual(len(projection.coalition_ids), 2)
        self.assertIn("A", projection.coalition_ids)
        self.assertIn("B", projection.coalition_ids)
        self.assertGreater(projection.expected_combined_impact, projection.expected_additive_impact)
        self.assertGreater(projection.synergy_density, 0)
        print(f"Team AB: Combined={projection.expected_combined_impact:.3f}, Additive={projection.expected_additive_impact:.3f}, Density={projection.synergy_density:.3f}")

    def test_delta_impact_addition(self):
        base_team = [self.agent_a]
        report = self.evaluator.calculate_delta_impact(self.task, base_team, self.agent_b)
        
        self.assertEqual(report.agent_id, "B")
        self.assertGreater(report.delta_total_impact, 0)
        # B should add synergy because (A, B) is in historical records
        self.assertGreater(report.delta_synergy_impact, 0)
        self.assertGreater(report.structural_necessity_score, 0)
        print(f"Adding B to A: Delta Total={report.delta_total_impact:.3f}, Necessity={report.structural_necessity_score:.3f}")

    def test_structural_necessity_of_reliable_agent(self):
        # Compare necessity of Agent B (consistent) vs Agent C (inconsistent)
        team = [self.agent_a]
        report_b = self.evaluator.calculate_delta_impact(self.task, team, self.agent_b)
        report_c = self.evaluator.calculate_delta_impact(self.task, team, self.agent_c)
        
        # B should have higher structural necessity than C 
        # because B is more consistent and has historical synergy with A
        self.assertGreater(report_b.structural_necessity_score, report_c.structural_necessity_score)
        print(f"Necessity B: {report_b.structural_necessity_score:.3f}, Necessity C: {report_c.structural_necessity_score:.3f}")

    def test_removal_impact(self):
        team = [self.agent_a, self.agent_b]
        report = self.evaluator.calculate_removal_impact(self.task, team, self.agent_a)
        
        self.assertEqual(report.agent_id, "A")
        # Delta total impact should be negative because we lost an agent
        self.assertLess(report.delta_total_impact, 0)
        # Necessity should still be positive
        self.assertGreater(report.structural_necessity_score, 0)
        print(f"Removing A from AB: Delta Total={report.delta_total_impact:.3f}, Necessity={report.structural_necessity_score:.3f}")

if __name__ == "__main__":
    unittest.main()

import unittest
from cooperative_context_model import CooperativeContextModel, DomainImpactType, CooperativeContextTensor

class TestCooperativeContextModel(unittest.TestCase):
    def test_task_encoding(self):
        # Define complex task requirements
        capabilities = {
            "causal_inference": 0.8,
            "stochastic_modeling": 0.4,
            "domain_expertise_finance": 0.6
        }
        
        tensor = CooperativeContextModel.encode_task(
            impact_domain=DomainImpactType.ECONOMIC,
            capabilities=capabilities,
            causal_depth=3,
            risk_threshold=0.2,
            horizon=12.5
        )
        
        # Verify structure
        self.assertEqual(tensor.domain_impact_type, DomainImpactType.ECONOMIC)
        self.assertAlmostEqual(sum(tensor.required_capability_vectors.values()), 1.0)
        self.assertEqual(tensor.expected_downstream_causal_depth, 3)
        self.assertEqual(tensor.uncertainty_tolerance, 0.2)
        self.assertEqual(tensor.temporal_horizon, 12.5)

    def test_alignment_scoring(self):
        capabilities = {"math": 0.5, "physics": 0.5}
        tensor = CooperativeContextModel.encode_task(
            impact_domain=DomainImpactType.TECHNICAL,
            capabilities=capabilities,
            causal_depth=1,
            risk_threshold=0.5,
            horizon=1.0
        )
        
        agent_profile = {"math": 1.0, "physics": 0.0, "art": 1.0}
        score = CooperativeContextModel.compute_alignment_score(tensor, agent_profile)
        
        # 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        self.assertEqual(score, 0.5)

if __name__ == "__main__":
    unittest.main()

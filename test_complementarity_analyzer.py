import unittest
from cooperative_intelligence import CooperativeIntelligenceVector
from cooperative_context_model import CooperativeContextModel, DomainImpactType
from synergy_forecast_simulator import HistoricalCoalitionRecord
from complementarity_analyzer import ComplementarityAnalyzer, CapabilityDependency

class TestComplementarityAnalyzer(unittest.TestCase):
    def setUp(self):
        # Define some agents
        # Agent A is a "Frontend" person
        self.agent_frontend = CooperativeIntelligenceVector(
            agent_id="frontend_genie",
            predictive_calibration_reliability=0.8,
            marginal_cooperative_influence_consistency=0.8,
            cross_role_integration_depth=0.9,
            capability_profile={"ui_design": 0.9, "frontend_dev": 0.8}
        )
        
        # Agent B is a "Backend" person
        self.agent_backend = CooperativeIntelligenceVector(
            agent_id="backend_beast",
            predictive_calibration_reliability=0.8,
            marginal_cooperative_influence_consistency=0.8,
            cross_role_integration_depth=0.7,
            capability_profile={"api_design": 0.9, "database_mgm": 0.8}
        )
        
        # Agent C is another "Frontend" person (redundant with A)
        self.agent_frontend_2 = CooperativeIntelligenceVector(
            agent_id="frontend_clone",
            predictive_calibration_reliability=0.7,
            marginal_cooperative_influence_consistency=0.7,
            cross_role_integration_depth=0.5,
            capability_profile={"ui_design": 0.8, "frontend_dev": 0.7}
        )

        # Define dependencies: API design enables Frontend development
        self.deps = [
            CapabilityDependency(source="api_design", target="frontend_dev", strength=0.8),
            CapabilityDependency(source="database_mgm", target="api_design", strength=0.9)
        ]

    def test_structural_complementarity_favors_diverse_enablement(self):
        analyzer = ComplementarityAnalyzer(dependencies=self.deps)
        
        # Complementarity between Frontend and Backend (should be high due to dependencies)
        score_diff = analyzer.compute_structural_complementarity(self.agent_frontend, self.agent_backend)
        
        # Complementarity between two Frontenders (should be lower due to redundancy and lack of dependencies)
        score_sim = analyzer.compute_structural_complementarity(self.agent_frontend, self.agent_frontend_2)
        
        self.assertGreater(score_diff, score_sim)
        print(f"Frontend + Backend Complementarity: {score_diff}")
        print(f"Frontend + Frontend Complementarity: {score_sim}")

    def test_historical_topology_affects_score(self):
        # Initial score without history
        analyzer_no_history = ComplementarityAnalyzer(dependencies=self.deps)
        score_base = analyzer_no_history.compute_structural_complementarity(self.agent_frontend, self.agent_backend)
        
        # Add history where these two worked great together
        history = [
            HistoricalCoalitionRecord(
                agents=("frontend_genie", "backend_beast"),
                additive_expectation=10.0,
                realized_impact=15.0 # 50% synergy density
            )
        ]
        analyzer_with_history = ComplementarityAnalyzer(dependencies=self.deps, historical_records=history)
        score_history = analyzer_with_history.compute_structural_complementarity(self.agent_frontend, self.agent_backend)
        
        self.assertGreater(score_history, score_base)
        print(f"Base score: {score_base}, With positive history: {score_history}")

    def test_non_obvious_synergy_detection(self):
        task = CooperativeContextModel.encode_task(
            impact_domain=DomainImpactType.TECHNICAL,
            capabilities={"frontend_dev": 0.5, "api_design": 0.5},
            causal_depth=3,
            risk_threshold=0.5,
            horizon=5.0
        )
        
        # Agent D: Low direct skill for the task, but provides a dependency
        # Task needs frontend_dev and api_design.
        # Agent backend has high api_design.
        # Agent D has high database_mgm (which enables api_design) but low direct task skills.
        agent_d = CooperativeIntelligenceVector(
            agent_id="foundation_guy",
            predictive_calibration_reliability=0.9,
            marginal_cooperative_influence_consistency=0.9,
            cross_role_integration_depth=0.8,
            capability_profile={"database_mgm": 1.0} # Not directly needed by task!
        )
        
        analyzer = ComplementarityAnalyzer(dependencies=self.deps)
        agents = [self.agent_frontend, self.agent_backend, agent_d]
        
        non_obvious = analyzer.detect_non_obvious_synergy(task, agents)
        print(f"Non-obvious synergy candidates count: {len(non_obvious)}")
        for cand in non_obvious:
            print(f"Candidate: {cand}")
        
        # agent_d + agent_backend should be a candidate because database_mgm enables api_design
        backend_foundation_pair = next((p for p in non_obvious if "foundation_guy" in p["agents"] and "backend_beast" in p["agents"]), None)
        
        self.assertIsNotNone(backend_foundation_pair, f"Expected backend_foundation_pair in {non_obvious}")
        print(f"Successfully detected non-obvious synergy!")

if __name__ == "__main__":
    unittest.main()

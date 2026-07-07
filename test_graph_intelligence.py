"""
test_graph_intelligence.py
Unit tests for Member 1 Week 4 modules: DynamicTrafficGraph and GraphIntelligenceModule.
"""

import unittest
from traffic_graph import DynamicTrafficGraph
from graph_intelligence import GraphIntelligenceModule

class TestGraphModules(unittest.TestCase):
    
    def setUp(self):
        self.num_nodes = 4
        self.graph = DynamicTrafficGraph(num_nodes=self.num_nodes)
        self.intelligence = GraphIntelligenceModule(num_nodes=self.num_nodes)
        
    def test_dynamic_graph_initialization(self):
        snapshot = self.graph.get_snapshot()
        self.assertEqual(len(snapshot['nodes']), self.num_nodes)
        
        # Test node features
        node0 = snapshot['nodes'][0]
        self.assertIn('density', node0)
        self.assertIn('queue_length', node0)
        
    def test_dynamic_graph_update(self):
        self.graph.update_node(0, density=0.8, queue_length=15.0, waiting_time=45.0, average_speed=10.0, signal_phase=0.0)
        snapshot = self.graph.get_snapshot()
        self.assertEqual(snapshot['nodes'][0]['density'], 0.8)
        self.assertEqual(snapshot['nodes'][0]['queue_length'], 15.0)
        
    def test_graph_intelligence_train_predict(self):
        self.graph.update_node(0, density=0.8, queue_length=15.0, waiting_time=45.0, average_speed=10.0, signal_phase=0.0)
        self.graph.update_node(1, density=0.2, queue_length=2.0, waiting_time=5.0, average_speed=30.0, signal_phase=1.0)
        
        snapshot = self.graph.get_snapshot()
        
        # Dummy target: nodes' own density
        target_congestion = [n['density'] for i, n in snapshot['nodes'].items()]
        
        # Train step
        loss = self.intelligence.train_step(snapshot, target_congestion)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0.0)
        
        # Predict
        preds = self.intelligence.predict_congestion(snapshot)
        self.assertEqual(len(preds), self.num_nodes)
        
        # Node importance
        importance = self.intelligence.get_node_importance(snapshot)
        self.assertEqual(len(importance), self.num_nodes)
        # Should sum to approximately 1.0 (allow small float error)
        self.assertAlmostEqual(sum(importance), 1.0, places=5)
        
    def test_compare_with_baseline(self):
        gcn_preds = [0.8, 0.2, 0.5, 0.1]
        base_preds = [0.75, 0.25, 0.45, 0.15]
        
        comparison = self.intelligence.compare_with_baseline(gcn_preds, base_preds)
        self.assertEqual(len(comparison), 4)
        self.assertAlmostEqual(comparison[0]["difference"], 0.05, places=5)

if __name__ == '__main__':
    unittest.main()

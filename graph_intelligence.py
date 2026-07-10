"""
graph_intelligence.py — Graph Intelligence Module
Member 1: Task 2

GCN for network-wide congestion prediction using PyTorch Geometric.
Compares performance against baseline linear regression prediction.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class TrafficGCN(torch.nn.Module):
    def __init__(self, num_node_features=5, hidden_channels=16, num_classes=1):
        super(TrafficGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        # x: Node feature matrix [num_nodes, num_node_features]
        # edge_index: Graph connectivity [2, num_edges]
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 2 (Node embeddings)
        embeddings = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(embeddings)
        
        # Output predictions (e.g., predicted congestion level per node)
        out = self.out(x)
        return out, embeddings

class GraphIntelligenceModule:
    def __init__(self, num_nodes=4):
        self.model = TrafficGCN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.num_nodes = num_nodes
        
    def _dict_to_tensor(self, graph_dict):
        """Converts the dict snapshot from DynamicTrafficGraph to PyTorch tensors."""
        nodes = graph_dict['nodes']
        edges = graph_dict['edges']
        
        # Features: [density, queue_length, waiting_time, average_speed, signal_phase]
        x_list = []
        for i in range(self.num_nodes):
            n = nodes.get(i, {})
            x_list.append([
                n.get('density', 0.0),
                n.get('queue_length', 0.0),
                n.get('waiting_time', 0.0),
                n.get('average_speed', 30.0),
                n.get('signal_phase', 0.0)
            ])
        x = torch.tensor(x_list, dtype=torch.float)
        
        # Edges
        edge_indices = []
        edge_weights = []
        for u, neighbors in edges.items():
            for v, e_data in neighbors.items():
                edge_indices.append([u, v])
                # use capacity / flow as a proxy for weight, or just 1.0 for simplicity
                edge_weights.append(e_data.get('flow', 1.0))
                
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty(0, dtype=torch.float)
            
        return x, edge_index, edge_weight

    def train_step(self, graph_dict, target_congestion_list):
        """
        Online training step.
        target_congestion_list: [target_node_0, target_node_1, ...]
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x, edge_index, edge_weight = self._dict_to_tensor(graph_dict)
        target = torch.tensor(target_congestion_list, dtype=torch.float).view(-1, 1)
        
        out, _ = self.model(x, edge_index, edge_weight)
        loss = self.criterion(out, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def predict_congestion(self, graph_dict):
        """Returns predicted congestion values for all nodes."""
        self.model.eval()
        with torch.no_grad():
            x, edge_index, edge_weight = self._dict_to_tensor(graph_dict)
            out, embeddings = self.model(x, edge_index, edge_weight)
            return out.view(-1).numpy()
            
    def get_node_importance(self, graph_dict):
        """
        Identifies critical road segments based on the graph features 
        and model embeddings (e.g. proxying attention/centrality).
        """
        self.model.eval()
        with torch.no_grad():
            x, edge_index, edge_weight = self._dict_to_tensor(graph_dict)
            _, embeddings = self.model(x, edge_index, edge_weight)
            
            # Simple heuristic: L2 norm of the node embedding represents its "activation" or importance
            importance_scores = torch.norm(embeddings, p=2, dim=1).numpy()
            
            # Normalize
            if np.sum(importance_scores) > 0:
                importance_scores = importance_scores / np.sum(importance_scores)
                
            return importance_scores

    def compare_with_baseline(self, gcn_preds, baseline_preds, ground_truth=None):
        """
        Compare GCN predictions against Member 1's baseline (linear regression).
        """
        comparison = {}
        for i in range(len(gcn_preds)):
            gcn_val = gcn_preds[i]
            base_val = baseline_preds[i] if i < len(baseline_preds) else 0.0
            
            diff = abs(gcn_val - base_val)
            comparison[i] = {
                "gcn_pred": float(gcn_val),
                "baseline_pred": float(base_val),
                "difference": float(diff)
            }
            if ground_truth is not None and i < len(ground_truth):
                comparison[i]["gcn_error"] = float(abs(gcn_val - ground_truth[i]))
                comparison[i]["baseline_error"] = float(abs(base_val - ground_truth[i]))
                
        return comparison

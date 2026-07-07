"""
traffic_graph.py — Dynamic Traffic Graph Construction
Member 1: Task 1

Constructs and updates a live NetworkX directed graph modeling the intersection grid.
Features are synchronized with the simulation loop.
"""

import networkx as nx
import matplotlib.pyplot as plt
import os

class DynamicTrafficGraph:
    def __init__(self, num_nodes=4, topology=None):
        self.G = nx.DiGraph()
        self.num_nodes = num_nodes
        
        # Build base topology
        if topology is None:
            # Default line topology: 0 <-> 1 <-> 2 <-> 3
            topology = {}
            for i in range(num_nodes):
                neighbors = []
                if i > 0: neighbors.append(i - 1)
                if i < num_nodes - 1: neighbors.append(i + 1)
                topology[i] = neighbors
                
        # Initialize nodes with default features
        for i in range(num_nodes):
            self.G.add_node(i, 
                density=0.0, 
                queue_length=0.0, 
                waiting_time=0.0, 
                average_speed=30.0, 
                signal_phase=0.0 # 0: Red, 1: Green, 2: Yellow
            )
            
        # Initialize edges with default features
        for node_id, neighbors in topology.items():
            for neighbor_id in neighbors:
                self.G.add_edge(node_id, neighbor_id, 
                    flow=0.0, 
                    travel_time=10.0, 
                    propagation_rate=0.3, 
                    capacity=100.0
                )
                
        self.pos = nx.spring_layout(self.G, seed=42) # Fixed layout for consistent visualization

    def update_node(self, node_id, density, queue_length, waiting_time, average_speed, signal_phase):
        if self.G.has_node(node_id):
            self.G.nodes[node_id]['density'] = float(density)
            self.G.nodes[node_id]['queue_length'] = float(queue_length)
            self.G.nodes[node_id]['waiting_time'] = float(waiting_time)
            self.G.nodes[node_id]['average_speed'] = float(average_speed)
            self.G.nodes[node_id]['signal_phase'] = float(signal_phase)

    def update_edge(self, u, v, flow, travel_time, propagation_rate, capacity):
        if self.G.has_edge(u, v):
            self.G.edges[u, v]['flow'] = float(flow)
            self.G.edges[u, v]['travel_time'] = float(travel_time)
            self.G.edges[u, v]['propagation_rate'] = float(propagation_rate)
            self.G.edges[u, v]['capacity'] = float(capacity)

    def get_snapshot(self):
        """Returns a dict snapshot of current features for sharing."""
        return {
            "nodes": dict(self.G.nodes(data=True)),
            "edges": nx.to_dict_of_dicts(self.G)
        }
        
    def save_visualization(self, path="static/traffic_graph.png"):
        """Renders the graph and saves to image."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        plt.figure(figsize=(6, 4))
        
        # Node colors based on density
        node_colors = [self.G.nodes[n]['density'] for n in self.G.nodes]
        
        # Edge widths based on flow
        edge_widths = [max(1.0, self.G.edges[e]['flow'] * 5) for e in self.G.edges]
        
        nx.draw(self.G, self.pos, 
                node_color=node_colors, cmap=plt.cm.Reds, vmin=0, vmax=1.0,
                width=edge_widths,
                with_labels=True, font_color='white', font_weight='bold',
                node_size=600, edge_color='gray')
                
        plt.title("Dynamic Traffic Graph Live", color='white')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', facecolor='#0f141e', edgecolor='none')
        plt.close()

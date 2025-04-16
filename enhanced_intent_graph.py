import networkx as nx
import json
import matplotlib.pyplot as plt

class EnhancedIntentGraph:
    def __init__(self, json_path):
        """
        Initialize the intent graph with JSON rules
        Args:
            json_path: Path to the JSON rules file
        """
        self.graph = nx.DiGraph()
        self.rules = self._load_rules(json_path)
        self._build_graph()
        
    def get_response_style(self, intent: str) -> str:
        """Get response style for an intent"""
        if intent in self.graph.nodes:
            return self.graph.nodes[intent].get('response_style', 'neutral')
        return 'neutral'
        
    def get_clearance_for_rule(self, rule_id: str) -> int:
        """Retrieve clearance level for a given rule ID"""
        rule_node = f"RULE_{rule_id}"
        if not self.graph.has_node(rule_node):
            return 1  # Default clearance if rule not found
        return self.graph.nodes[rule_node].get('agent_level', 1)
        
    def _load_rules(self, json_path):
        """
        Load rules from JSON file
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading rules file: {str(e)}")

    def _build_graph(self):
        """
        Construct the graph hierarchy from loaded rules
        """
 
        self.graph.add_node("ROOT", type="root", clearance=0, description="Global root node")
        
        # Process all rules
        for rule in self.rules:
         
            intent = rule['intent_label']
            self.graph.add_node(
                intent,
                type="intent",
                category=rule['category'],
                clearance=rule['agent_level'] or 0,
                response_style=rule['response_style'],
                greeting=rule['greeting'],
                confidence=rule['intent_confidence']
            )
            
         
            if not self.graph.has_edge("ROOT", intent):
                self.graph.add_edge("ROOT", intent, relation="contains")
            
 
            rule_node = f"RULE_{rule['rule_id']}"
            self.graph.add_node(
                rule_node,
                type="rule",
                response=rule['response'],
                raw_text=rule['raw_text'],
                trigger_type=rule['trigger_type'],
                confidence=rule['intent_confidence'],
                agent_level=rule['agent_level']
            )
            
           
            self.graph.add_edge(intent, rule_node, relation="has_rule")
            
     
            for keyword in rule['keywords']:
                kw_text = keyword['keyword'].strip().lower()
                self.graph.add_node(
                    kw_text,
                    type="keyword",
                    score=keyword['score'],
                    normalized=kw_text
                )
                
                
                self.graph.add_edge(rule_node, kw_text, relation="triggered_by")
                
           
                self.graph.add_edge(kw_text, intent, relation="maps_to", weight=keyword['score'])

    def find_related_intents(self, keyword, threshold=0.7):
        """
        Find intents related to a keyword with confidence threshold
        Args:
            keyword: Input keyword/phrase
            threshold: Minimum confidence score (0-1)
        Returns:
            List of (intent, metadata) tuples
        """
        normalized_kw = keyword.strip().lower()
        related = []
        
        if normalized_kw in self.graph.nodes:
            for successor in self.graph.successors(normalized_kw):
                if self.graph.nodes[successor]['type'] == 'intent':
                    edge_data = self.graph.get_edge_data(normalized_kw, successor)
                    if edge_data['weight'] >= threshold:
                        related.append((
                            successor,
                            {
                                'confidence': edge_data['weight'],
                                'category': self.graph.nodes[successor]['category'],
                                'clearance': self.graph.nodes[successor]['clearance']
                            }
                        ))
        
        return sorted(related, key=lambda x: x[1]['confidence'], reverse=True)

    def get_response_flow(self, intent):
        """
        Get full response flow for an intent
        Args:
            intent: Target intent label
        Returns:
            List of response rules with metadata
        """
        flow = []
        if intent in self.graph.nodes:
            for rule_node in self.graph.successors(intent):
                if self.graph.nodes[rule_node]['type'] == 'rule':
                    flow.append({
                        'rule_id': rule_node,
                        'response': self.graph.nodes[rule_node]['response'],
                        'confidence': self.graph.nodes[rule_node]['confidence'],
                        'agent_level': self.graph.nodes[rule_node]['agent_level'],
                        'trigger_type': self.graph.nodes[rule_node]['trigger_type']
                    })
        return sorted(flow, key=lambda x: x['confidence'], reverse=True)

#visualising graph
    def visualize_subgraph(self, node, depth=2):
        """
        Visualize subgraph around a node
        Args:
            node: Center node for visualization
            depth: Exploration depth from center node
        """
        if node not in self.graph.nodes:
            raise ValueError(f"Node {node} not found in graph")
            
        plt.figure(figsize=(12, 8))
        subgraph = nx.ego_graph(self.graph, node, radius=depth)
        pos = nx.spring_layout(subgraph, k=0.5)
        
  
        node_colors = []
        for n in subgraph:
            node_type = self.graph.nodes[n].get('type', 'unknown')
            if node_type == 'intent': node_colors.append('lightgreen')
            elif node_type == 'rule': node_colors.append('lightcoral')
            elif node_type == 'keyword': node_colors.append('lightblue')
            else: node_colors.append('gray')
            
        nx.draw(subgraph, pos, with_labels=True, node_size=1500,
               node_color=node_colors, font_size=8, alpha=0.9)
        

        edge_labels = nx.get_edge_attributes(subgraph, 'relation')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)
        
        plt.title(f"Subgraph around {node}")
        plt.show()

    def get_operational_path(self, start_intent, max_depth=3):
        """
        Get operational path for mission planning
        Args:
            start_intent: Starting intent node
            max_depth: Maximum path depth
        Returns:
            List of nodes in operational sequence
        """
        if start_intent not in self.graph.nodes:
            return []
            
        path = []
        queue = [(start_intent, 0)]
        visited = set()
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
                
            visited.add(current)
            path.append(current)
            
            # Prioritize rule nodes first
            successors = sorted(
                self.graph.successors(current),
                key=lambda x: self.graph.nodes[x].get('confidence', 0),
                reverse=True
            )
            
            for successor in successors:
                if successor not in visited:
                    queue.append((successor, depth+1))
                    
        return path

#just for my purpose , no use of main function
if __name__ == "__main__":

    graph = EnhancedIntentGraph("final_enriched_rules.json")
    

    print("=== Intents related to 'thermal surveillance' ===")
    print(graph.find_related_intents("thermal surveillance"))
    
   
    print("\n=== Response flow for Emergency Protocol ===")
    print(graph.get_response_flow("Emergency Protocol")[0])  
    
    
    print("\n=== Operational path from 'Tactical Procedures' ===")
    print(graph.get_operational_path("Tactical Procedures", max_depth=3))
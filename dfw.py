import networkx as nx

class IntentGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
  
        self.graph.add_node("communication_verification", type="intent", clearance=3, chunk_id="doc_chunk_1")
        self.graph.add_node("extraction_protocols", type="intent", clearance=4, chunk_id="doc_chunk_2")
        self.graph.add_node("safehouse_access", type="intent", clearance=3, chunk_id="doc_chunk_3")
        self.graph.add_node("counter_surveillance", type="intent", clearance=4, chunk_id="doc_chunk_4")
        self.graph.add_node("high_risk_operations", type="intent", clearance=5, chunk_id="doc_chunk_5")
        self.graph.add_node("emergency_directives", type="intent", clearance=4, chunk_id="doc_chunk_6")

  
        self._add_with_chunks("communication_verification", "LCC", "protocol", "uses", "doc_chunk_1")

        for layer in [
            "Quantum Hashing",
            "One-Time Pad (OTP) Key Exchange",
            "Neural Signatures"
        ]:
            self._add_with_chunks("LCC", layer, "subprotocol", "has_layer", "doc_chunk_1")

        self._add_with_chunks("communication_verification", "Handshake Protocol", "protocol", "requires", "doc_chunk_1")

        for step in [
            "Step 1: Blink twice in 2s",
            "Step 2: Tap left wrist",
            "Step 3: Recite rotating phrase"
        ]:
            self._add_with_chunks("Handshake Protocol", step, "step", "has_step", "doc_chunk_1")

     
        self._add_with_chunks("extraction_protocols", "S-29 Protocol", "protocol", "includes", "doc_chunk_2")
        self._add_with_chunks("extraction_protocols", "Shadow Step", "protocol", "triggered_by_compromise", "doc_chunk_2")

        for phase in [
            "Disruptor Wave",
            "Persona Collapse",
            "Phase-Shift Safehouse"
        ]:
            self._add_with_chunks("Shadow Step", phase, "phase", "has_phase", "doc_chunk_2")

        for safehouse in [
            "Safehouse K-41",
            "Safehouse H-77",
            "Facility X-17",
            "The Silent Room"
        ]:
            self._add_with_chunks("safehouse_access", safehouse, "safehouse", "includes", "doc_chunk_3")

        self._add_with_chunks("counter_surveillance", "Ghost-Step Algorithm", "protocol", "requires", "doc_chunk_4")

        for tech in [
            "Digital Trace Removal",
            "Biometric Decoy",
            "Quantum Misdirection"
        ]:
            self._add_with_chunks("Ghost-Step Algorithm", tech, "method", "includes", "doc_chunk_4")

        self._add_with_chunks("high_risk_operations", "Project Eclipse", "protocol", "includes", "doc_chunk_5")

        for action in [
            "Silent Dissolution Agents",
            "Omega Wave",
            "Blackout Plan Zeta"
        ]:
            self._add_with_chunks("Project Eclipse", action, "operation", "has_action", "doc_chunk_5")

        self._add_with_chunks("emergency_directives", "Zeta-5 Protocol", "protocol", "triggered_by_release_code", "doc_chunk_6")

    def _add_with_chunks(self, parent, child, node_type, relation, chunk_id):
        self.graph.add_node(child, type=node_type, chunk_id=chunk_id)
        self.graph.add_edge(parent, child, relation=relation)

    def traverse_up(self, term):
        path = []
        queue = [term]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            parents = list(self.graph.predecessors(current))
            path.extend(parents)
            queue.extend(parents)

        return path

    def get_successors(self, node):
        return list(self.graph.successors(node))

    def get_chunk_id(self, node):
        return self.graph.nodes[node].get("chunk_id")

    def get_all_related_chunks(self, seed_node):
        related = set()
        visited = set()
        queue = [seed_node]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            chunk = self.get_chunk_id(current)
            if chunk:
                related.add(chunk)
            queue.extend(self.graph.successors(current))

        return list(related)

    def visualize(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8)
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()

if __name__ == "__main__":
    g = IntentGraph()
    print("--- Successors of Shadow Step ---")
    print(g.get_successors("Shadow Step"))

    print("--- Traversing up from 'Quantum Hashing' ---")
    print(g.traverse_up("Quantum Hashing"))

    print("--- Chunk for 'Omega Wave' ---")
    print(g.get_chunk_id("Omega Wave"))

    print("--- All related chunks from 'Ghost-Step Algorithm' ---")
    print(g.get_all_related_chunks("Ghost-Step Algorithm"))



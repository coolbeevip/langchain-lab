from langgraph.graph import Graph


class PlantUMLGraph:
    def __init__(self, graph: Graph):
        self.graph = graph

    def draw_plantuml(self) -> str:
        plantuml = []
        graph_json = self.graph.to_json()['graph']
        graph_nodes = graph_json['nodes']
        graph_edges = graph_json['edges']
        plantuml.append("@startuml")
        for node in graph_nodes:
            pass
            # plantuml.append(f"class {node['id']} {{")
            # for attr in node['attrs']:
            #     plantuml.append(f"  {attr['key']} : {attr['value']}")
            # plantuml.append("}")

        for edge in graph_edges:
            source_node = edge['source'] if edge['source'] != '__start__' else '(*)'
            target_node = edge['target'] if edge['target'] != '__end__' else '(*)'
            plantuml.append(f"{source_node} --> {target_node}")

        plantuml.append("@enduml")
        return "\n".join(plantuml)

    def print_plantuml(self):
        print(self.draw_plantuml())

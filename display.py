import networkx as nx
import matplotlib.pyplot as plt

from network_generator import generate_signed_network, generate_lfr

def pos_subgraph(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) == 1)

def neg_subgraph(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) == 1)

def draw_communities(G: nx.Graph, communities: dict) -> None:

    print(communities)
    c = [[] for _ in set(communities.values())]
    for k,v in communities.items():
        c[v].append(k)
    communities = c

    supergraph = nx.cycle_graph(len(communities))
    superpos = nx.spring_layout(supergraph, scale=2, seed=429)

    centers = list(superpos.values())
    pos = {}
    for center, comm in zip(centers, communities):
        pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center, seed=1430))

    for clr, nodes in enumerate(communities):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color="tab:blue", node_size=100, cmap="tab20")

    pos_edges = [(u,v) for u, v, d in G.edges(data=True) if d.get("sign", 1) == 1]
    neg_edges = [(u,v) for u, v, d in G.edges(data=True) if d.get("sign", 1) == -1]
    nx.draw_networkx_edges(G, edgelist=pos_edges, pos=pos)
    nx.draw_networkx_edges(G, edgelist=neg_edges, pos=pos, style="dashed")
    plt.show()
    
    return

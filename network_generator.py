import random
import networkx as nx
import numpy as np


def generate_signed_network(
    n_nodes: int = 128,
    n_communities: int = 4,
    avg_degree: int = 8,
    p_positive: float = 0.8,
    mixing_param: float = 0.1,
    seed: int = None,
) -> tuple[nx.Graph, dict]:


    def distribute_nodes(n_nodes: int, n_communities: int) -> list[int]:
        base = n_nodes // n_communities
        remainder = n_nodes % n_communities
        sizes = [base + (1 if i < remainder else 0) for i in range(n_communities)]
        return sizes

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    community_sizes = distribute_nodes(n_nodes, n_communities)
    ground_truth = {}
    node = 0
    for comm_idx, size in enumerate(community_sizes):
        for _ in range(size):
            ground_truth[node] = comm_idx
            node += 1

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    nx.set_node_attributes(G, ground_truth, "community")

    # Build community membership lists for fast lookup
    communities = [[] for _ in range(n_communities)]
    for n, c in ground_truth.items():
        communities[c].append(n)

    # Target total edges from avg_degree
    target_edges = (n_nodes * avg_degree) // 2
    n_external = int(target_edges * mixing_param)
    n_internal = target_edges - n_external

    added = 0
    attempts = 0
    max_attempts = target_edges * 20

    while added < n_internal and attempts < max_attempts:
        attempts += 1
        comm = random.randrange(n_communities)
        members = communities[comm]
        if len(members) < 2:
            continue
        u, v = random.sample(members, 2)
        if G.has_edge(u, v):
            continue
        sign = 1 if random.random() < p_positive else -1
        G.add_edge(u, v, sign=sign)
        added += 1

    # --- External edges ---
    added_ext = 0
    attempts = 0
    while added_ext < n_external and attempts < max_attempts:
        attempts += 1
        c1, c2 = random.sample(range(n_communities), 2)
        u = random.choice(communities[c1])
        v = random.choice(communities[c2])
        if G.has_edge(u, v):
            continue
        # External edges are predominantly negative
        sign = -1 if random.random() < p_positive else 1
        G.add_edge(u, v, sign=sign)
        added_ext += 1

    return G, ground_truth

def print_network_summary(G: nx.Graph, ground_truth: dict) -> None:
    signs = [d["sign"] for _, _, d in G.edges(data=True)]
    n_pos = signs.count(1)
    n_neg = signs.count(-1)

    internal, external = 0, 0
    for u, v in G.edges():
        if ground_truth[u] == ground_truth[v]:
            internal += 1
        else:
            external += 1

    print("=== Network Summary ===")
    print(f"  Nodes            : {G.number_of_nodes()}")
    print(f"  Edges            : {G.number_of_edges()}")
    print(f"  Communities      : {len(set(ground_truth.values()))}")
    print(f"  Positive edges   : {n_pos} ({100*n_pos/len(signs):.1f}%)")
    print(f"  Negative edges   : {n_neg} ({100*n_neg/len(signs):.1f}%)")
    print(f"  Internal edges   : {internal}")
    print(f"  External edges   : {external}")
    print(f"  Avg degree       : {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    print()

import numpy as np
import networkx as nx
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    precision_score,
    recall_score,
    f1_score,
)
from itertools import combinations


def nmi_sum(ground_truth: dict, detected: dict) -> float:
    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[n] for n in nodes]
    y_pred = [detected.get(n, -1) for n in nodes]
    return normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")


def ami(ground_truth: dict, detected: dict) -> float:
    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[n] for n in nodes]
    y_pred = [detected.get(n, -1) for n in nodes]
    return adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")


def ari(ground_truth: dict, detected: dict) -> float:
    nodes = sorted(ground_truth.keys())
    y_true = [ground_truth[n] for n in nodes]
    y_pred = [detected.get(n, -1) for n in nodes]
    return adjusted_rand_score(y_true, y_pred)


def f_score(ground_truth: dict, detected: dict) -> float:
    nodes = sorted(ground_truth.keys())
    y_true_pairs, y_pred_pairs = [], []
    for u, v in combinations(nodes, 2):
        y_true_pairs.append(int(ground_truth[u] == ground_truth[v]))
        y_pred_pairs.append(int(detected.get(u, -1) == detected.get(v, -1)))
    if sum(y_pred_pairs) == 0 or sum(y_true_pairs) == 0:
        return 0.0
    return f1_score(y_true_pairs, y_pred_pairs, zero_division=0)


def precision_recall(ground_truth: dict, detected: dict) -> tuple[float, float]:
    nodes = sorted(ground_truth.keys())
    y_true_pairs, y_pred_pairs = [], []
    for u, v in combinations(nodes, 2):
        y_true_pairs.append(int(ground_truth[u] == ground_truth[v]))
        y_pred_pairs.append(int(detected.get(u, -1) == detected.get(v, -1)))
    p = precision_score(y_true_pairs, y_pred_pairs, zero_division=0)
    r = recall_score(y_true_pairs, y_pred_pairs, zero_division=0)
    return p, r


def signed_modularity(G: nx.Graph, detected: dict) -> float:
    signs = nx.get_edge_attributes(G, "sign")
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) == 1]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) == -1]
    W_pos, W_neg = len(pos_edges), len(neg_edges)
    W = W_pos + W_neg
    if W == 0:
        return 0.0

    pos_H = nx.Graph(); pos_H.add_nodes_from(G.nodes()); pos_H.add_edges_from(pos_edges)
    neg_H = nx.Graph(); neg_H.add_nodes_from(G.nodes()); neg_H.add_edges_from(neg_edges)
    k_pos = dict(pos_H.degree())
    k_neg = dict(neg_H.degree())

    Qs = 0.0
    for u, v in G.edges():
        if detected.get(u) != detected.get(v):
            continue
        sign = signs.get((u, v), signs.get((v, u), 1))
        if sign == 1 and W_pos > 0:
            Qs += 1 - (k_pos[u] * k_pos[v]) / (2 * W_pos)
        elif sign == -1 and W_neg > 0:
            Qs += -1 + (k_neg[u] * k_neg[v]) / (2 * W_neg)
    return Qs / W


def frustration(G: nx.Graph, detected: dict) -> float:
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0.0

    communities = {}
    for node, comm in detected.items():
        communities.setdefault(comm, set()).add(node)

    frustrated = 0
    for comm_nodes in communities.values():
        sub = G.subgraph(comm_nodes)
        frustrated += sum(1 for _, _, d in sub.edges(data=True) if d.get("sign", 1) == -1)

    comm_list = list(communities.values())
    for i, ni in enumerate(comm_list):
        for j, nj in enumerate(comm_list):
            if j <= i:
                continue
            for _, _, d in nx.edge_boundary(G, ni, nj, data=True):
                if d.get("sign", 1) == 1:
                    frustrated += 1

    return frustrated / total_edges


def conductance(G: nx.Graph, detected: dict) -> float:
    pos_G = nx.Graph()
    pos_G.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if d.get("sign", 1) == 1:
            pos_G.add_edge(u, v)

    communities = {}
    for node, comm in detected.items():
        communities.setdefault(comm, set()).add(node)

    all_nodes = set(G.nodes())
    conductances = []
    for comm_nodes in communities.values():
        outside = all_nodes - comm_nodes
        if not outside:
            continue
        cut = sum(1 for _ in nx.edge_boundary(pos_G, comm_nodes, outside))
        vol_s = sum(d for _, d in pos_G.degree(comm_nodes))
        vol_sc = sum(d for _, d in pos_G.degree(outside))
        denom = min(vol_s, vol_sc)
        if denom > 0:
            conductances.append(cut / denom)

    return sum(conductances) / len(conductances) if conductances else 0.0


def triangle_balance_ratio(G: nx.Graph, detected: dict) -> float:
    communities = {}
    for node, comm in detected.items():
        communities.setdefault(comm, set()).add(node)

    balanced = 0
    total = 0

    for comm_nodes in communities.values():
        sub = G.subgraph(comm_nodes)
        for clique in nx.enumerate_all_cliques(sub):
            if len(clique) > 3:
                break
            if len(clique) < 3:
                continue
            u, v, w = clique
            pos_count = sum(1 for a, b in [(u, v), (v, w), (u, w)]
                            if sub[a][b].get("sign", 1) == 1)
            total += 1
            if pos_count % 2 == 1:
                balanced += 1

    return balanced / total if total > 0 else float("nan")


def modularity_density(G: nx.Graph, detected: dict) -> float:
    communities = {}
    for node, comm in detected.items():
        communities.setdefault(comm, set()).add(node)

    comm_list = list(communities.items())
    Qds = 0.0

    for idx, (comm_id, comm_nodes) in enumerate(comm_list):
        ni = len(comm_nodes)
        if ni < 2:
            continue

        sub = G.subgraph(comm_nodes)
        pos_in = sum(1 for _, _, d in sub.edges(data=True) if d.get("sign", 1) == 1)
        neg_in = sum(1 for _, _, d in sub.edges(data=True) if d.get("sign", 1) == -1)
        total_in = pos_in + neg_in

        # Internal density: fraction of possible internal edges that exist
        max_edges = ni * (ni - 1) / 2
        di = total_in / max_edges if max_edges > 0 else 0.0

        # Intra-community signed contribution
        Qds += di * (pos_in - neg_in) / ni

        # Inter-community penalty
        for jdx, (other_id, other_nodes) in enumerate(comm_list):
            if jdx <= idx:
                continue
            nj = len(other_nodes)
            boundary = list(nx.edge_boundary(G, comm_nodes, other_nodes, data=True))
            e_ij = len(boundary)
            if e_ij == 0:
                continue
            dij = e_ij / (ni * nj)
            Qds -= dij * e_ij / (ni * nj)

    return Qds


def signed_proportion(G: nx.Graph, community: set) -> float:
    if len(community) < 2:
        return 0.0
    sub = G.subgraph(community)
    signs = [d.get("sign", 1) for _, _, d in sub.edges(data=True)]
    if not signs:
        return 0.0
    return (signs.count(1) - signs.count(-1)) / len(signs)

def agreement_score(G: nx.Graph, community: set) -> float:
    boundary = list(nx.edge_boundary(G, community, data=True))
    if not boundary:
        return 0.0
    return sum(1 for _, _, d in boundary if d.get("sign", 1) == -1) / len(boundary)

def evaluate(
    G: nx.Graph,
    detected: dict,
    ground_truth: dict = None,
) -> dict:

    results = {}

    results["n_communities_detected"] = len(set(detected.values()))
    results["signed_modularity"]      = signed_modularity(G, detected)
    results["frustration"]            = frustration(G, detected)
    results["conductance"]            = conductance(G, detected)
    results["triangle_balance_ratio"] = triangle_balance_ratio(G, detected)
    results["modularity_density"]     = modularity_density(G, detected)

    if ground_truth is not None:
        results["nmi"]       = nmi_sum(ground_truth, detected)
        results["ami"]       = ami(ground_truth, detected)
        results["ari"]       = ari(ground_truth, detected)
        results["f_score"]   = f_score(ground_truth, detected)
        p, r = precision_recall(ground_truth, detected)
        results["precision"] = p
        results["recall"]    = r

    return results

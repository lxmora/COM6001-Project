"""
All algorithms follow the pattern:
    fn(G: nx.Graph, **kwargs) -> dict[node, community_label]
"""

import random as _random_module
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import warnings
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning


from metrics import signed_modularity, frustration, agreement_score, signed_proportion


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def _make_rng(seed: int | None) -> tuple[_random_module.Random, np.random.Generator]:
    rng = _random_module.Random(seed)
    nprng = np.random.default_rng(seed)
    return rng, nprng


def _random_partition(nodes: list, n_communities: int,
                      rng: _random_module.Random) -> dict:
    return {n: rng.randrange(n_communities) for n in nodes}



# Algsp (Bakhtar & Harutyunyan, 2022)

def algsp(G: nx.Graph, seed: int = None) -> dict:
    nodes = list(G.nodes())
    assigned = {}
    community_id = 0
    unassigned = set(nodes)

    while unassigned:
        seed_node = min(unassigned)
        community = {seed_node}

        # Step 1: Grow by adding unassigned neighbours that improve SP
        rejected: set = set()
        changed = True
        while changed:
            changed = False
            for candidate in [n for n in nx.node_boundary(G, community) if n in unassigned]:
                if candidate in rejected:
                    continue
                cur = signed_proportion(G, community)
                community.add(candidate)
                if signed_proportion(G, community) > cur:
                    changed = True
                    rejected.clear()
                else:
                    community.remove(candidate)
                    rejected.add(candidate)

        # Step 2: Remove nodes that improve SP when removed
        rejected.clear()
        changed = True
        while changed:
            changed = False
            if len(community) <= 1:
                break
            for node in list(community):
                if node in rejected:
                    continue
                cur = signed_proportion(G, community)
                community.remove(node)
                if signed_proportion(G, community) > cur:
                    changed = True
                    rejected.clear()
                else:
                    community.add(node)
                    rejected.add(node)

        # Step 3: Remove nodes that improve As when removed
        if len(community) > 1:
            rejected.clear()
            changed = True
            while changed:
                changed = False
                if len(community) <= 1:
                    break
                for node in list(community):
                    if node in rejected:
                        continue
                    cur = agreement_score(G, community)
                    community.remove(node)
                    if agreement_score(G, community) > cur:
                        changed = True
                        rejected.clear()
                    else:
                        community.add(node)
                        rejected.add(node)

        for n in community:
            assigned[n] = community_id
            unassigned.discard(n)
        community_id += 1

    return assigned


# SPONGE (Cucuringu et al., 2019)

def sponge(
    G: nx.Graph,
    n_communities: int = 2,
    tau_p: float = 1.0,
    tau_n: float = 1.0,
    n_init: int = 10,
    seed: int = None,
) -> dict:

    def nx_to_signed_sparse(G: nx.Graph) -> tuple[sp.csr_matrix, sp.csr_matrix, list]:
        nodes = list(G.nodes())
        n = len(nodes)
        idx = {v: i for i, v in enumerate(nodes)}
        pr, pc, nr, nc = [], [], [], []

        for u, v, d in G.edges(data=True):
            i, j = idx[u], idx[v]
            if d.get("sign", 1) == 1:
                pr += [i, j]; pc += [j, i]
            else:
                nr += [i, j]; nc += [j, i]

        ones_p = np.ones(len(pr), dtype=float)
        ones_n = np.ones(len(nr), dtype=float)

        Ap = sp.csr_matrix((ones_p, (pr, pc)), shape=(n, n)) if pr else sp.csr_matrix((n, n))
        An = sp.csr_matrix((ones_n, (nr, nc)), shape=(n, n)) if nr else sp.csr_matrix((n, n))

        return Ap, An, nodes

    #Ap, An, nodes = nx_to_signed_sparse(G)
    Ap, An, nodes = nx_to_signed_sparse(G)
    n = len(nodes)

    Dp = sp.diags(np.asarray(Ap.sum(axis=1)).flatten())
    Dn = sp.diags(np.asarray(An.sum(axis=1)).flatten())

    Lp = (Dp - Ap + tau_n * sp.eye(n)).toarray().astype(np.float64)
    Ln = (Dn - An + tau_p * sp.eye(n)).toarray().astype(np.float64)

    # Solve generalised symmetric eigenproblem: Lp v = lambda Ln v
    # We want the k smallest eigenvalues — their eigenvectors form the embedding
    k = min(n_communities, n - 1)
    eigvals, eigvecs = eigh(Lp, Ln, subset_by_index=[0, k - 1])

    # k-means on the spectral embedding
    _, nprng = _make_rng(seed)
    km_seed = int(nprng.integers(0, 2**31)) if seed is not None else None
    km = KMeans(n_clusters=n_communities, random_state=km_seed, n_init=n_init)
    labels = km.fit_predict(eigvecs)

    return {node: int(labels[i]) for i, node in enumerate(nodes)}


# CAbABC (Baofang, 2015)

def cababc(
    G: nx.Graph,
    n_communities: int = 4,
    n_employed: int = 20,
    n_onlooker: int = 20,
    n_scout_limit: int = 10,
    max_cycles: int = 100,
    seed: int = None,
) -> dict:

    rng, nprng = _make_rng(seed)
    nodes = list(G.nodes())

    def qs_score(partition):
        return signed_modularity(G, partition)

    def neighbour_solution(source):
        new_source = source.copy()
        n = rng.choice(nodes)
        candidates = [c for c in range(n_communities) if c != new_source[n]]
        new_source[n] = rng.choice(candidates)
        return new_source

    def selection_probs(qs_scores):
        arr = np.array(qs_scores, dtype=float)
        arr -= arr.min()
        arr += 1e-6
        return arr / arr.sum()

    sources = [_random_partition(nodes, n_communities, rng) for _ in range(n_employed)]
    scores = [qs_score(s) for s in sources]
    trial_counts = [0] * n_employed

    best_idx = int(np.argmax(scores))
    best_solution = sources[best_idx].copy()
    best_score = scores[best_idx]

    for _ in range(max_cycles):
        # Employed bee phase
        for i in range(n_employed):
            candidate = neighbour_solution(sources[i])
            cand_score = qs_score(candidate)
            if cand_score > scores[i]:
                sources[i] = candidate
                scores[i] = cand_score
                trial_counts[i] = 0
            else:
                trial_counts[i] += 1

        # Onlooker bee phase
        probs = selection_probs(scores)
        for _ in range(n_onlooker):
            idx = int(nprng.choice(n_employed, p=probs))
            candidate = neighbour_solution(sources[idx])
            cand_score = qs_score(candidate)
            if cand_score > scores[idx]:
                sources[idx] = candidate
                scores[idx] = cand_score
                trial_counts[idx] = 0
                probs = selection_probs(scores)

        # Scout bee phase with global-best reinjection
        reset_occurred = False
        for i in range(n_employed):
            if trial_counts[i] >= n_scout_limit:
                sources[i] = _random_partition(nodes, n_communities, rng)
                scores[i] = qs_score(sources[i])
                trial_counts[i] = 0
                reset_occurred = True

        if reset_occurred:
            worst_idx = int(np.argmin(scores))
            sources[worst_idx] = best_solution.copy()
            scores[worst_idx] = best_score
            trial_counts[worst_idx] = 0

        cycle_best_idx = int(np.argmax(scores))
        if scores[cycle_best_idx] > best_score:
            best_solution = sources[cycle_best_idx].copy()
            best_score = scores[cycle_best_idx]

    return best_solution


# SN-MOGA (Amelio & Pizzuti, 2013)

def sn_moga(
    G: nx.Graph,
    n_communities: int = 4,
    pop_size: int = 50,
    n_generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    seed: int = None,
) -> dict:

    rng, _ = _make_rng(seed)
    nodes = list(G.nodes())
    _obj_cache: dict = {}

    def objectives(partition):
        key = id(partition)
        if key not in _obj_cache:
            _obj_cache[key] = (frustration(G, partition), -signed_modularity(G, partition))
        return _obj_cache[key]

    def dominates(a, b):
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def pareto_front(population):
        objs = [objectives(p) for p in population]
        mask = [False] * len(population)
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j and not mask[j] and dominates(objs[j], objs[i]):
                    mask[i] = True
                    break
        return [p for p, m in zip(population, mask) if not m]

    def crossover(p1, p2):
        if rng.random() > crossover_rate:
            return p1.copy()
        pt = rng.randint(1, len(nodes) - 1)
        return {n: (p1[n] if i < pt else p2[n]) for i, n in enumerate(nodes)}

    def mutate(partition):
        p = partition.copy()
        for n in nodes:
            if rng.random() < mutation_rate:
                p[n] = rng.randrange(n_communities)
        return p

    population = [_random_partition(nodes, n_communities, rng) for _ in range(pop_size)]

    for _ in range(n_generations):
        front = pareto_front(population)
        offspring = []
        while len(offspring) < pop_size:
            child = crossover(rng.choice(front), rng.choice(front))
            offspring.append(mutate(child))

        combined = population + offspring
        new_front = pareto_front(combined)

        if len(new_front) >= pop_size:
            population = rng.sample(new_front, pop_size)
        else:
            population = new_front
            remainder = [p for p in combined if p not in new_front]
            rng.shuffle(remainder)
            population += remainder[:pop_size - len(population)]

    final_front = pareto_front(population)
    return min(final_front, key=lambda p: frustration(G, p))


# KST (Sun et al., 2022)

def kst(G: nx.Graph, k: int = 2) -> dict:

    def pos_subgraph(G: nx.Graph) -> nx.Graph:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) == 1)
        return H

    def is_balanced_triangle(signs: tuple) -> bool:
        return sum(1 for s in signs if s == 1) % 2 == 1


    def filter_balanced(subgraph: nx.Graph, nodes: set) -> set:
        stable = set(nodes)
        changed = True
        while changed:
            changed = False
            sub = subgraph.subgraph(stable)
            bad = set()
            for clique in nx.enumerate_all_cliques(sub):
                if len(clique) > 3:
                    break
                if len(clique) < 3:
                    continue
                u, v, w = clique
                signs = (sub[u][v].get("sign", 1), sub[v][w].get("sign", 1), sub[u][w].get("sign", 1))
                if not is_balanced_triangle(signs):
                    bad.update(clique)
            if bad:
                stable -= bad
                changed = True
        return stable


    def assign_to_nearest(G: nx.Graph, node: int, assigned: dict) -> int | None:
        scores = {}
        for nb, d in G[node].items():
            if nb in assigned and d.get("sign", 1) == 1:
                c = assigned[nb]
                scores[c] = scores.get(c, 0) + 1
        return max(scores, key=scores.get) if scores else None

    pos_G = pos_subgraph(G)
    core_graph = nx.k_core(pos_G, k)
    assigned = {}
    community_id = 0

    for component in nx.connected_components(core_graph):
        stable = filter_balanced(G.subgraph(component), component)
        for n in stable:
            if n not in assigned:
                assigned[n] = community_id
        community_id += 1

    for node in G.nodes():
        if node not in assigned:
            best_comm = assign_to_nearest(G, node, assigned)
            assigned[node] = best_comm if best_comm is not None else community_id
            if best_comm is None:
                community_id += 1

    return assigned

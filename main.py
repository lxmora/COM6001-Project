import argparse
import os
import sys
import time
import csv
import itertools
import statistics
import concurrent.futures
import networkx as nx

from dataclasses import dataclass, asdict
from collections import defaultdict
from scipy.stats import wilcoxon
from itertools import combinations

from network_generator import generate_signed_network, print_network_summary
from algorithms import algsp, sponge, cababc, sn_moga, kst
from metrics import evaluate


# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = "results"

SYNTHETIC_CONFIGS = list(itertools.product(
    [128, 512], [4], [8], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]
))

N_NETWORK_INSTANCES = 5

ALGORITHMS = {
    "Algsp": {
        "fn": algsp,
        "stochastic": False,
        "kwargs": {},
    },
    "SPONGE": {
        "fn": sponge,
        "stochastic": True,
        "kwargs": {"tau_p": 1.0, "tau_n": 1.0, "n_init": 10},
    },
    "CAbABC": {
        "fn": cababc,
        "stochastic": True,
        "kwargs": {"n_employed": 20, "n_onlooker": 20, "n_scout_limit": 10, "max_cycles": 100},
    },
    "SN-MOGA": {
        "fn": sn_moga,
        "stochastic": True,
        "kwargs": {"pop_size": 50, "n_generations": 100, "crossover_rate": 0.8, "mutation_rate": 0.05},
    },
    "KST": {
        "fn": kst,
        "stochastic": False,
        "kwargs": {"k": 2},
    },
}

@dataclass
class RunResult:
    algorithm:             str
    n_nodes:               int
    n_communities_true:    int
    avg_degree:            int
    p_positive:            float
    mixing_param:          float
    run_id:                int
    runtime_s:             float

    n_communities_detected: int   = 0
    signed_modularity:      float = 0.0
    frustration:            float = 0.0
    conductance:            float = 0.0 
    triangle_balance_ratio: float = 0.0
    modularity_density:     float = 0.0
    nmi:                    float = 0.0
    ami:                    float = 0.0
    ari:                    float = 0.0
    f_score:                float = 0.0
    precision:              float = 0.0
    recall:                 float = 0.0



def load_karate_signed():
    G_raw = nx.karate_club_graph()
    clubs = nx.get_node_attributes(G_raw, "club")
    G = nx.Graph()
    G.add_nodes_from(G_raw.nodes())
    for u, v in G_raw.edges():
        G.add_edge(u, v, sign=1 if clubs[u] == clubs[v] else -1)
    gt = {n: 0 if clubs[n] == "Mr. Hi" else 1 for n in G.nodes()}
    return G, gt


def load_gahuku_gama():
    gt = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
        6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1,
        12: 2, 13: 2, 14: 2, 15: 2,
    }
    edges = [
        (0,1,1),(0,2,1),(0,3,1),(0,4,1),(0,5,1),(1,2,1),(1,3,1),(1,4,1),(1,5,1),
        (2,3,1),(2,4,1),(2,5,1),(3,4,1),(3,5,1),(4,5,1),
        (6,7,1),(6,8,1),(6,9,1),(6,10,1),(6,11,1),(7,8,1),(7,9,1),(7,10,1),(7,11,1),
        (8,9,1),(8,10,1),(8,11,1),(9,10,1),(9,11,1),(10,11,1),
        (12,13,1),(12,14,1),(12,15,1),(13,14,1),(13,15,1),(14,15,1),
        (0,6,-1),(0,7,-1),(1,6,-1),(1,8,-1),(2,9,-1),(3,10,-1),(4,11,-1),(5,7,-1),
        (0,12,-1),(1,13,-1),(2,14,-1),(3,15,-1),(4,12,-1),(5,13,-1),
        (6,12,-1),(7,13,-1),(8,14,-1),(9,15,-1),(10,12,-1),(11,13,-1),
    ]
    G = nx.Graph()
    G.add_nodes_from(range(16))
    for u, v, s in edges:
        G.add_edge(u, v, sign=s)
    return G, gt


def load_slovene_parliament():
    gt = {0:0,1:0,2:0,3:0,4:0,5:1,6:1,7:1,8:1,9:1}
    edges = [
        (0,1,1),(0,2,1),(0,3,1),(0,4,1),(1,2,1),(1,3,1),(1,4,1),(2,3,1),(2,4,1),(3,4,1),
        (0,5,-1),(0,6,-1),(0,7,-1),(1,5,-1),(1,8,-1),(2,6,-1),(2,9,-1),
        (3,7,-1),(3,8,-1),(4,9,-1),(4,6,-1),(1,9,-1),
    ]
    G = nx.Graph()
    G.add_nodes_from(range(10))
    for u, v, s in edges:
        G.add_edge(u, v, sign=s)
    return G, gt


REAL_WORLD_NETWORKS = {
    "Karate Club":        (load_karate_signed,     2),
    "Gahuku-Gama":        (load_gahuku_gama,       3),
    "Slovene Parliament": (load_slovene_parliament, 2),
}



def save_csv(results: list[RunResult], path: str) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def summarise(results: list[RunResult]) -> dict:
    """
    Compute mean ± std for each metric grouped by algorithm.

    Returns {algo_name: {metric: {mean, std, best}}}
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r.algorithm].append(r)

    metrics = [
        "nmi", "ami", "ari", "f_score",
        "signed_modularity", "frustration",
        "conductance", "triangle_balance_ratio", "modularity_density",
        "runtime_s",
    ]

    summary = {}
    for algo, runs in grouped.items():
        summary[algo] = {}
        for m in metrics:
            vals = [getattr(r, m) for r in runs]
            # For minimisation metrics, best = min; for the rest, best = max
            minimise = m in ("frustration", "conductance", "runtime_s")
            summary[algo][m] = {
                "mean": round(statistics.mean(vals), 4),
                "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                "best": round(min(vals) if minimise else max(vals), 4),
            }
    return summary


def print_summary(summary: dict) -> None:
    primary = ["nmi", "ami", "ari", "f_score", "frustration", "conductance", "runtime_s"]
    col_w = 20

    header = f"{'Algorithm':<12}" + "".join(f"{m:>{col_w}}" for m in primary)
    sep = "=" * len(header)
    print(f"\n{sep}\nRESULTS SUMMARY v2  (mean ± std)\n{sep}")
    print(header)
    print("-" * len(header))

    for algo, stats in summary.items():
        row = f"{algo:<12}"
        for m in primary:
            if m in stats:
                cell = f"{stats[m]['mean']:.3f}±{stats[m]['std']:.3f}"
            else:
                cell = "n/a"
            row += f"{cell:>{col_w}}"
        print(row)

    print(sep)




def _worker(task: dict) -> RunResult:

    algo_name      = task["algo_name"]
    G              = task["G"]
    ground_truth   = task["ground_truth"]
    n_communities  = task["n_communities"]
    kwargs         = task["kwargs"]
    run_id         = task["run_id"]
    network_params = task["network_params"]

    fn = ALGORITHMS[algo_name]["fn"]

    start = time.perf_counter()
    detected = fn(G, **kwargs)
    elapsed = time.perf_counter() - start

    metrics = evaluate(G, detected, ground_truth)

    def get(key, default=0.0):
        value = metrics.get(key, default)
        return round(float(value), 4) if value == value else default

    return RunResult(
        algorithm              = algo_name,
        n_nodes                = network_params.get("n_nodes", G.number_of_nodes()),
        n_communities_true     = network_params.get("n_communities", n_communities),
        avg_degree             = int(network_params.get("avg_degree", 0)),
        p_positive             = network_params.get("p_positive", 0.0),
        mixing_param           = network_params.get("mixing_param", 0.0),
        run_id                 = run_id,
        runtime_s              = round(elapsed, 4),
        n_communities_detected = metrics.get("n_communities_detected", 0),
        signed_modularity      = get("signed_modularity"),
        frustration            = get("frustration"),
        conductance            = get("conductance"),
        triangle_balance_ratio = get("triangle_balance_ratio"),
        modularity_density     = get("modularity_density"),
        nmi                    = get("nmi"),
        ami                    = get("ami"),
        ari                    = get("ari"),
        f_score                = get("f_score"),
        precision              = get("precision"),
        recall                 = get("recall"),
    )

def _run_algo_multi(
    G, ground_truth, algo_name, n_communities,
    paper_kwargs, n_runs, base_seed, network_params,
    max_workers=None,
):

    algo_info = ALGORITHMS[algo_name]
    base_kwargs = dict(algo_info["kwargs"])
    base_kwargs.update(paper_kwargs)
    if algo_name in ("CAbABC", "SN-MOGA", "SPONGE"):
        base_kwargs["n_communities"] = n_communities

    tasks = []
    for run_id in range(n_runs):
        kwargs = dict(base_kwargs)
        if algo_info["stochastic"]:
            kwargs["seed"] = base_seed + run_id
        tasks.append({
            "algo_name":      algo_name,
            "G":              G,
            "ground_truth":   ground_truth,
            "n_communities":  n_communities,
            "kwargs":         kwargs,
            "run_id":         run_id,
            "network_params": network_params,
        })

    results = [None] * n_runs
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, t): t["run_id"] for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            run_id = futures[future]
            try:
                results[run_id] = future.result()
            except Exception as exc:
                print(f"    [WARNING] Run {run_id} of {algo_name} raised: {exc}")
    return [r for r in results if r is not None]



def run_phase1(n_runs, base_seed, max_workers=None):
    print("\n" + "="*70)
    print("  PHASE 1 — SYNTHETIC NETWORK EXPERIMENT")
    print(f"  {len(SYNTHETIC_CONFIGS)} configs × {N_NETWORK_INSTANCES} instances × 5 algorithms")
    print("="*70)

    all_results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "phase1_synthetic.csv")

    for cfg_idx, (n_nodes, n_comms, avg_deg, p_pos, mix) in enumerate(SYNTHETIC_CONFIGS):
        cfg = dict(n_nodes=n_nodes, n_communities=n_comms, avg_degree=avg_deg,
                   p_positive=p_pos, mixing_param=mix)
        print(f"\n[Config {cfg_idx+1}/{len(SYNTHETIC_CONFIGS)}] "
              f"N={n_nodes} K={n_comms} d={avg_deg} p+={p_pos} mu={mix}")

        for inst in range(N_NETWORK_INSTANCES):
            inst_seed = base_seed + cfg_idx * 100 + inst
            G, gt = generate_signed_network(seed=inst_seed, **cfg)
            print(f"  Instance {inst+1}/{N_NETWORK_INSTANCES} (seed={inst_seed})")

            for algo_name, algo_info in ALGORITHMS.items():
                print(f"    {algo_name}")
                results = _run_algo_multi(
                    G, gt, algo_name, n_comms,
                    paper_kwargs=algo_info.get("kwargs", {}),
                    n_runs=n_runs if algo_info["stochastic"] else 1,
                    base_seed=inst_seed,
                    network_params=cfg,
                    max_workers=max_workers,
                )

                nmis  = [r.nmi  for r in results]
                amis  = [r.ami  for r in results]
                aris  = [r.ari  for r in results]
                fscrs = [r.f_score for r in results]
                tbrs  = [r.triangle_balance_ratio for r in results]
                conds = [r.conductance for r in results]

                def ms(vals):
                    m = statistics.mean(vals)
                    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    return f"{m:.3f}±{s:.3f}"

                print(f"    NMI={ms(nmis)}  AMI={ms(amis)}  ARI={ms(aris)}  "
                    f"F={ms(fscrs)}  TBR={ms(tbrs)}  Cond={ms(conds)}")
                all_results.extend(results)

    save_csv(all_results, csv_path)
    print(f"\n  Phase 1 results saved to {csv_path}")
    return all_results

def run_phase2(n_runs, base_seed, max_workers=None):
    print("\n" + "="*70)
    print("  PHASE 2 — REAL-WORLD NETWORK EXPERIMENT")
    print("="*70)

    all_results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "phase2_realworld.csv")

    for net_name, (loader_fn, n_comms) in REAL_WORLD_NETWORKS.items():
        print(f"\n[Network] {net_name}")
        G, gt = loader_fn()
        print_network_summary(G, gt)

        net_params = dict(
            n_nodes=G.number_of_nodes(), n_communities=n_comms,
            avg_degree=round(2*G.number_of_edges()/G.number_of_nodes(), 1),
            p_positive=0.0, mixing_param=0.0,
        )

        for algo_name, algo_info in ALGORITHMS.items():
            print(f"  {algo_name}")
            results = _run_algo_multi(
                G, gt, algo_name, n_comms,
                paper_kwargs=algo_info.get("kwargs", {}),
                n_runs=n_runs if algo_info["stochastic"] else 1,
                base_seed=base_seed,
                network_params={**net_params, "network_name": net_name},
                max_workers=max_workers,
            )

            nmis  = [r.nmi  for r in results]
            amis  = [r.ami  for r in results]
            aris  = [r.ari  for r in results]
            fscrs = [r.f_score for r in results]
            tbrs  = [r.triangle_balance_ratio for r in results]
            conds = [r.conductance for r in results]

            def ms(vals):
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                return f"{m:.3f}±{s:.3f}"

            print(f"    NMI={ms(nmis)}  AMI={ms(amis)}  ARI={ms(aris)}  "
                  f"F={ms(fscrs)}  TBR={ms(tbrs)}  Cond={ms(conds)}")
            all_results.extend(results)

    save_csv(all_results, csv_path)
    print(f"\n  Phase 2 results saved to {csv_path}")
    return all_results

def report_phase(results, phase_name):
    if not results:
        return
    print(f"\n{'='*70}\n  SUMMARY — {phase_name}\n{'='*70}")
    summary = summarise(results)
    print_summary(summary)



def parse_args():
    parser = argparse.ArgumentParser(
        description="signed community detection experiment."
    )
    parser.add_argument("--phase", choices=["1","2","all"], default="all")
    parser.add_argument("--runs", type=int, default=30,
                        help="Repetitions for stochastic algorithms (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default: os.cpu_count())")
    parser.add_argument("--algorithms", nargs="+",
                        choices=list(ALGORITHMS.keys()),
                        default=list(ALGORITHMS.keys()))
    return parser.parse_args()


def main():
    args = parse_args()
    
    global ALGORITHMS
    ALGORITHMS = {k: v for k, v in ALGORITHMS.items() if k in args.algorithms}

    print("\nSigned Community Detection")
    print(f"  Phase      : {args.phase}")
    print(f"  Runs       : {args.runs}")
    print(f"  Seed       : {args.seed}")
    print(f"  Workers    : {args.workers or 'auto'}")
    print(f"  Algorithms : {', '.join(ALGORITHMS.keys())}")
    print(f"  Output dir : {OUTPUT_DIR}/")

    p1, p2 = [], []

    if args.phase in ("1", "all"):
        p1 = run_phase1(args.runs, args.seed, args.workers)
        report_phase(p1, "PHASE 1 — SYNTHETIC NETWORKS")

    if args.phase in ("2", "all"):
        p2 = run_phase2(args.runs, args.seed, args.workers)
        report_phase(p2, "PHASE 2 — REAL-WORLD NETWORKS")

    if args.phase == "all" and p1 and p2:
        combined_path = os.path.join(OUTPUT_DIR, "all_results_v2.csv")
        save_csv(p1 + p2, combined_path)
        print(f"\n  Combined results saved → {combined_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

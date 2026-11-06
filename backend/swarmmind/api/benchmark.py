from __future__ import annotations
import csv
import json
import os
import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from swarmmind.config import Config
from swarmmind.simulator import Simulator
from swarmmind.strategies.centralized import CentralizedStrategy
from swarmmind.strategies.cooperative import CooperativeStrategy
from swarmmind.strategies.uncoordinated import UncoordinatedStrategy

benchmark_cache: Dict[str, Dict[str, Any]] = {}

def create_strategy(name: str):
    if name == "uncoordinated":
        return UncoordinatedStrategy()
    elif name == "centralized":
        return CentralizedStrategy()
    elif name == "cooperative":
        return CooperativeStrategy()
    else:
        raise ValueError(f"Unknown strategy for benchmark: {name}")

def run_benchmark_headless(
    seeds: List[int],
    strategies: Optional[List[str]] = None,
    num_robots: int = 20,
    episode_ticks: int = 600,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if strategies is None:
        strategies = ["uncoordinated", "centralized", "cooperative"]

    results_by_strategy: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    raw_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        for strat_name in strategies:
            cfg = Config(
                seed=seed,
                num_robots=num_robots,
                episode_ticks=episode_ticks,
                strategies=[strat_name],
            )
            strat = create_strategy(strat_name)
            rng = np.random.default_rng(seed)
            sim = Simulator(cfg, strat, rng=rng)

            # Run until finish
            while not sim.is_finished and sim.tick < episode_ticks:
                sim.step()

            m = sim.metrics.to_dict()
            run_record = {
                "seed": seed,
                "strategy": strat_name,
                **m,
            }
            results_by_strategy[strat_name].append(run_record)
            raw_runs.append(run_record)

    # Compute summary stats: mean and std per metric
    summary: Dict[str, Dict[str, Any]] = {}
    metrics_keys = ["collisions", "near_misses", "completed", "travel_time", "distance", "wait_ticks", "avg_completion_time"]

    for strat_name, records in results_by_strategy.items():
        summary[strat_name] = {}
        for k in metrics_keys:
            vals = [r[k] for r in records if r[k] is not None]
            if vals:
                summary[strat_name][f"{k}_mean"] = round(float(np.mean(vals)), 2)
                summary[strat_name][f"{k}_std"] = round(float(np.std(vals)), 2)
            else:
                summary[strat_name][f"{k}_mean"] = 0.0
                summary[strat_name][f"{k}_std"] = 0.0

        # Recovery time
        recov_vals = [r["recovery_time"] for r in records if r["recovery_time"] is not None]
        if recov_vals:
            summary[strat_name]["recovery_time_mean"] = round(float(np.mean(recov_vals)), 2)
            summary[strat_name]["recovery_time_std"] = round(float(np.std(recov_vals)), 2)
        else:
            summary[strat_name]["recovery_time_mean"] = None
            summary[strat_name]["recovery_time_std"] = None

    benchmark_id = str(uuid.uuid4())[:8]
    output_data = {
        "benchmark_id": benchmark_id,
        "seeds": seeds,
        "robots": num_robots,
        "episode_ticks": episode_ticks,
        "strategies": strategies,
        "summary": summary,
        "runs": raw_runs,
    }

    # Write out files
    if out_dir is None:
        # Default results folder in root
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"benchmark_{benchmark_id}.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Also write a standard latest JSON
    with open(os.path.join(out_dir, "benchmark_latest.json"), "w") as f:
        json.dump(output_data, f, indent=2)

    csv_path = os.path.join(out_dir, f"benchmark_{benchmark_id}.csv")
    if raw_runs:
        fieldnames = list(raw_runs[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_runs)

        with open(os.path.join(out_dir, "benchmark_latest.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_runs)

    benchmark_cache[benchmark_id] = output_data
    return output_data


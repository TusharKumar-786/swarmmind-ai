from __future__ import annotations
import uuid
import numpy as np
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from swarmmind.api.benchmark import benchmark_cache, run_benchmark_headless
from swarmmind.api.runs import run_manager
from swarmmind.api.ws import ws_router
from swarmmind.blackbox.logger import BlackBoxLogger
from swarmmind.chaos.scenarios import ChaosScenarioGenerator, SCENARIO_REGISTRY, ResilienceScore
from swarmmind.config import Config
from swarmmind.rl.policy_strategy import RLPolicyStrategy
from swarmmind.whatif.snapshotter import SimulationSnapshot, capture_snapshot, restore_and_run

app = FastAPI(title="SwarmMind AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)

class CreateRunRequest(BaseModel):
    seed: int = 42
    strategies: Optional[List[str]] = None
    config_overrides: Optional[Dict[str, Any]] = None

class ControlRunRequest(BaseModel):
    action: str  # "start" | "pause" | "resume" | "reset" | "step"
    speed: Optional[int] = None

class EventRunRequest(BaseModel):
    type: str  # "block" | "unblock"
    cells: Optional[Any] = None  # List of [x, y] or "auto_busiest"

class BenchmarkRequest(BaseModel):
    seeds: List[int] = [1, 2, 3, 4, 5]
    strategies: Optional[List[str]] = None
    robots: int = 20
    episode_ticks: int = 600

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.get("/api/config/default")
def get_default_config():
    return Config().model_dump()

@app.get("/api/strategies")
def get_strategies():
    rl_strat = RLPolicyStrategy()
    return [
        {
            "id": "uncoordinated",
            "name": "Uncoordinated (Baseline)",
            "description": "Each robot independently plans shortest paths with A*, blind to other agents.",
            "available": True,
        },
        {
            "id": "centralized",
            "name": "Centralized Scheduler",
            "description": "Global Hungarian task allocation with prioritized space-time reservation planning.",
            "available": True,
        },
        {
            "id": "cooperative",
            "name": "Cooperative Negotiation",
            "description": "Decentralized contract-net auctions, local reservation tables, right-of-way score arbitration & obstacle broadcasting.",
            "available": True,
        },
        {
            "id": "rl_policy",
            "name": "Reinforcement Learning Policy (RLlib)",
            "description": "Learned multi-agent policy via PettingZoo environment (Optional extension).",
            "available": rl_strat.is_available,
        },
    ]

@app.post("/api/runs")
def create_run(req: CreateRunRequest):
    base_cfg = Config(seed=req.seed)
    if req.config_overrides:
        cfg_dict = base_cfg.model_dump()
        cfg_dict.update(req.config_overrides)
        base_cfg = Config(**cfg_dict)

    strats = req.strategies or ["uncoordinated", "centralized", "cooperative"]
    run = run_manager.create_run(base_cfg, strats)
    return {"run_id": run.run_id, "status": "created"}

@app.post("/api/runs/{run_id}/control")
def control_run(run_id: str, req: ControlRunRequest):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if req.speed is not None:
        run.speed = max(1, min(50, req.speed))

    action = req.action.lower()
    if action == "start":
        run.is_running = True
        run.is_paused = False
    elif action == "pause":
        run.is_paused = True
    elif action == "resume":
        run.is_paused = False
        run.is_running = True
    elif action == "reset":
        run.is_running = False
        run.is_paused = False
        run.reset()
    elif action == "step":
        run.is_running = False
        return run.step()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

    return {"status": "ok", "action": action, "is_running": run.is_running, "is_paused": run.is_paused}

@app.post("/api/runs/{run_id}/events")
def trigger_event(run_id: str, req: EventRunRequest):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    ev_type = req.type.lower()
    if ev_type == "block":
        cells = None
        if req.cells != "auto_busiest" and isinstance(req.cells, list):
            cells = [tuple(c) for c in req.cells]
        chosen = run.trigger_block(cells)
        return {"status": "ok", "event": "blocked", "cells": [list(c) for c in chosen]}
    elif ev_type == "unblock":
        cells = None
        if isinstance(req.cells, list):
            cells = [tuple(c) for c in req.cells]
        run.trigger_unblock(cells)
        return {"status": "ok", "event": "unblocked"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown event type: {req.type}")

@app.post("/api/benchmark")
def run_benchmark(req: BenchmarkRequest):
    results = run_benchmark_headless(
        seeds=req.seeds,
        strategies=req.strategies,
        num_robots=req.robots,
        episode_ticks=req.episode_ticks,
    )
    return results

@app.get("/api/benchmark/{benchmark_id}")
def get_benchmark(benchmark_id: str):
    res = benchmark_cache.get(benchmark_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Benchmark {benchmark_id} not found")
    return res


# ─── Feature 1: Memory endpoint ──────────────────────────────────────────────

@app.get("/api/runs/{run_id}/memory")
def get_memory_summary(run_id: str):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    summaries: Dict[str, Any] = {}
    for strat_name, sim in run.simulators.items():
        strategy = sim.strategy
        if hasattr(strategy, "memory_engine"):
            summaries[strat_name] = strategy.memory_engine.get_summary()
    return {"memory_summaries": summaries}


# ─── Feature 2: Chaos Engineer ───────────────────────────────────────────────

@app.get("/api/chaos/scenarios")
def list_chaos_scenarios():
    return {"scenarios": [{"id": k, "description": v} for k, v in SCENARIO_REGISTRY.items()]}


@app.post("/api/chaos/run")
def run_chaos_test(body: Dict[str, Any]):
    from swarmmind.simulator import Simulator
    from swarmmind.strategies.uncoordinated import UncoordinatedStrategy
    from swarmmind.strategies.centralized import CentralizedStrategy
    from swarmmind.strategies.cooperative import CooperativeStrategy
    from swarmmind.world import Task as WTask

    scenario_name = body.get("scenario", "random_block")
    seed = int(body.get("seed", 42))
    difficulty = int(body.get("difficulty", 1))
    episode_ticks = int(body.get("episode_ticks", 300))

    if scenario_name not in SCENARIO_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario_name}")

    gen = ChaosScenarioGenerator(seed=seed, difficulty=difficulty)
    plan = gen.generate(scenario_name)

    strategy_classes = {
        "uncoordinated": UncoordinatedStrategy,
        "centralized": CentralizedStrategy,
        "cooperative": CooperativeStrategy,
    }

    results: Dict[str, Any] = {}
    for strat_name, StratClass in strategy_classes.items():
        cfg = Config(seed=seed, num_robots=20, num_tasks=60, episode_ticks=episode_ticks)
        sim = Simulator(cfg, StratClass())

        for tick_i in range(episode_ticks):
            for event in plan.events:
                if event.tick == tick_i:
                    if event.event_type == "block_cells":
                        cells = [tuple(c) for c in event.params.get("cells", [])]
                        valid = [c for c in cells if sim.world.is_passable(c[0], c[1])]
                        if valid:
                            sim.world.block_cells(valid)
                    elif event.event_type == "robot_breakdown":
                        for rid in event.params.get("robot_ids", []):
                            if rid in sim.world.robots:
                                r = sim.world.robots[rid]
                                r.status = "IDLE"
                                r.task_id = None
                                r.path = []
                    elif event.event_type == "urgent_surge":
                        count = event.params.get("count", 3)
                        rng2 = np.random.default_rng(seed + tick_i)
                        for _ in range(count):
                            if sim.world.pick_stations and sim.world.drop_stations:
                                pick = sim.world.pick_stations[int(rng2.integers(0, len(sim.world.pick_stations)))]
                                drop = sim.world.drop_stations[int(rng2.integers(0, len(sim.world.drop_stations)))]
                                tid = max(sim.world.active_tasks.keys(), default=-1) + 1
                                new_task = WTask(id=tid, pick=pick, drop=drop, priority=2)
                                sim.world.active_tasks[tid] = new_task
            sim.step()

        m = sim.metrics
        total_tasks = max(len(sim.world.all_tasks), m.completed_deliveries, 1)
        first_tick = plan.events[0].tick if plan.events else 0
        rs = ResilienceScore()
        rs.compute(
            completed=m.completed_deliveries,
            total_tasks=total_tasks,
            collisions=m.collisions,
            episode_ticks=episode_ticks,
            first_event_tick=first_tick,
        )
        results[strat_name] = {
            "completed_deliveries": m.completed_deliveries,
            "collisions": m.collisions,
            "total_travel_time": m.total_travel_time,
            "resilience_score": rs.composite_score,
        }

    return {
        "scenario": scenario_name,
        "seed": seed,
        "difficulty": difficulty,
        "events": [e.to_dict() for e in plan.events],
        "strategy_results": results,
    }


# ─── Feature 3: Black Box ─────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/blackbox")
def get_blackbox(
    run_id: str,
    robot_id: Optional[int] = None,
    decision_type: Optional[str] = None,
    tick_from: Optional[int] = None,
    tick_to: Optional[int] = None,
):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    records: List[Dict[str, Any]] = []
    for sim in run.simulators.values():
        if hasattr(sim, "blackbox"):
            filtered = sim.blackbox.get_filtered(
                robot_id=robot_id,
                decision_type=decision_type,
                tick_from=tick_from,
                tick_to=tick_to,
            )
            records.extend([r.to_dict() for r in filtered])
    records.sort(key=lambda r: (r["tick"], r["robot_id"]))
    return {"records": records, "total": len(records)}


# ─── Feature 4: Comm Blackout ────────────────────────────────────────────────

@app.post("/api/runs/{run_id}/comms/disconnect")
def comm_disconnect(run_id: str, body: Dict[str, Any]):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    robot_id = int(body.get("robot_id", -1))
    results: Dict[str, str] = {}
    for strat_name, sim in run.simulators.items():
        strategy = sim.strategy
        if hasattr(strategy, "comm_network") and strategy.comm_network:
            strategy.comm_network.disconnect(robot_id, tick=sim.tick)
            results[strat_name] = f"Robot {robot_id} disconnected"
    return {"disconnected": robot_id, "results": results}


@app.post("/api/runs/{run_id}/comms/reconnect")
def comm_reconnect(run_id: str, body: Dict[str, Any]):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    robot_id = body.get("robot_id")
    results: Dict[str, str] = {}
    for strat_name, sim in run.simulators.items():
        strategy = sim.strategy
        if hasattr(strategy, "comm_network") and strategy.comm_network:
            if robot_id is not None:
                strategy.comm_network.reconnect(int(robot_id), tick=sim.tick)
                results[strat_name] = f"Robot {robot_id} reconnected"
            else:
                strategy.comm_network.reconnect_all(tick=sim.tick)
                results[strat_name] = "All reconnected"
    return {"results": results}


@app.post("/api/runs/{run_id}/comms/partition")
def comm_partition(run_id: str, body: Dict[str, Any]):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    groups = body.get("groups", [])
    results: Dict[str, str] = {}
    for strat_name, sim in run.simulators.items():
        strategy = sim.strategy
        if hasattr(strategy, "comm_network") and strategy.comm_network:
            strategy.comm_network.create_partition(groups, tick=sim.tick)
            results[strat_name] = f"Partitioned into {len(groups)} groups"
    return {"results": results}


@app.get("/api/runs/{run_id}/comms/status")
def comm_status(run_id: str):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    comm_statuses: Dict[str, Any] = {}
    for strat_name, sim in run.simulators.items():
        strategy = sim.strategy
        if hasattr(strategy, "comm_network") and strategy.comm_network:
            comm_statuses[strat_name] = strategy.comm_network.get_all_statuses()
            comm_statuses[f"{strat_name}_events"] = strategy.comm_network.get_event_log()
    return {"comm_statuses": comm_statuses}


# ─── Feature 5: What-If ───────────────────────────────────────────────────────

_snapshots: Dict[str, SimulationSnapshot] = {}


@app.post("/api/runs/{run_id}/snapshot")
def create_snapshot(run_id: str, body: Dict[str, Any] = {}):
    run = run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    strat_names = list(run.simulators.keys())
    strategy_name = body.get("strategy", strat_names[0] if strat_names else "uncoordinated")
    sim = run.simulators.get(strategy_name)
    if not sim:
        sim = list(run.simulators.values())[0]
        strategy_name = strat_names[0]
    snap = capture_snapshot(sim, run_id, strategy_name)
    _snapshots[snap.snapshot_id] = snap
    return snap.to_dict()


@app.post("/api/whatif")
def run_whatif(body: Dict[str, Any]):
    snapshot_id = body.get("snapshot_id")
    if not snapshot_id or snapshot_id not in _snapshots:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    snap = _snapshots[snapshot_id]
    modifications = body.get("modifications", {})
    strategy_name = body.get("strategy", snap.strategy_name)
    episode_ticks = int(body.get("episode_ticks", 200))

    result = restore_and_run(snap, strategy_name, modifications, episode_ticks)
    whatif_id = uuid.uuid4().hex[:8]

    return {
        "whatif_id": whatif_id,
        "snapshot_id": snapshot_id,
        "baseline_metrics": snap.metrics_at_snapshot,
        "alternative_metrics": result,
        "modifications": modifications,
        "delta": {
            k: result.get(k, 0) - snap.metrics_at_snapshot.get(k, 0)
            for k in ["completed_deliveries", "collisions", "total_travel_time"]
        },
    }


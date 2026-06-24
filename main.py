#!/usr/bin/env python
"""
Agentic System Identification — pipeline entry point.

Climbs the fidelity ladder (white-box → grey-box → surrogate) on the pendulum
with Coulomb + viscous friction.  Agents call the real Anthropic LLM for
Intake and Modeler; all other agents are deterministic.

Usage
-----
    python main.py                           # default pendulum config
    python main.py --desc "my plant ..."     # custom plant description
    python main.py --config custom.yaml      # custom config file
    python main.py --seed 7 --budget 150     # override seed and budget
    python main.py --data-dir /tmp/sysid     # custom data directory

Environment
-----------
    ANTHROPIC_API_KEY   required for LLM agents (Intake, Modeler)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path

import yaml

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Allow running from the repo root without installing the package.
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Default plant description ─────────────────────────────────────────────────
DEFAULT_DESCRIPTION = textwrap.dedent("""\
    A simple pendulum driven by an external torque at its pivot.
    The bob has unknown mass, mounted on a rod of unknown length.
    Only viscous (speed-proportional) friction acts on the pivot; there is no Coulomb friction.
    Input:  torque [N·m], clamped to [-2, 2].
    Output: angle θ (measured from the downward equilibrium, in radians).
    Sample time: 0.02 s.
    Physics is known (rigid-body pendulum ODE) but all parameters are uncertain.
    I want a control-ready identified model.
""")


# ── Routing helpers (mirrors LangGraph edges without compilation) ──────────────

def _run_pipeline(
    initial_dossier,
    agents:      dict,
    routers:     dict,
    ship_node:   str,
    max_steps:   int = 20,
    budget_mgr=None,
):
    """
    Execute the pipeline by following the orchestrator's routing logic.

    Mirrors the LangGraph graph transitions without requiring compilation,
    which avoids Pydantic serialisation edge-cases in LangGraph's invoke().
    """
    import logging
    log = logging.getLogger("pipeline")

    from core.orchestrator import SHIP
    from core.schemas import Budget

    dossier = initial_dossier
    node    = next(iter(agents))   # first node is always INTAKE
    _sep80  = "=" * 80

    for step in range(max_steps):
        log.info(_sep80)
        log.info("NODE [%02d]: %s", step + 1, node.upper())
        log.info(_sep80)
        print(f"  [{step + 1:02d}] {node} ...", flush=True)

        dossier = agents[node](dossier)

        # Sync dossier budget from the authoritative BudgetManager so that
        # dossier.budget.exhausted and dossier.budget.spent are always current.
        if budget_mgr is not None:
            dossier = dossier.update(budget=Budget(
                total=budget_mgr.budget.total,
                spent=budget_mgr.spent,
            ))

        if node == ship_node:
            break

        if dossier.budget.exhausted:
            log.info("Budget exhausted — shipping best model.")
            print("  ⚠  Budget exhausted — shipping best model.")
            dossier = agents[ship_node](dossier)
            break

        next_node = routers[node](dossier)
        log.info("── routing: %s → %s", node, next_node)
        node = next_node

    return dossier


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic System Identification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="configs/pendulum_config.yaml",
        help="Path to YAML config file (default: configs/pendulum_config.yaml)",
    )
    parser.add_argument(
        "--desc", default=None,
        help="Plant description string (overrides default).  "
             "Wrap in quotes if it contains spaces.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the plant simulator (overrides config)",
    )
    parser.add_argument(
        "--budget", type=float, default=None,
        help="Total experiment budget in units (overrides config)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Root directory for experiment DB and model registry (default: data/)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Anthropic model ID for LLM agents (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--no-demo", action="store_true",
        help="Skip demo visualization after the pipeline completes",
    )
    parser.add_argument(
        "--demo-output", default=None,
        help="Path for the demo animation (default: demo/demo.gif). "
             "Use .mp4 if conda-forge ffmpeg is installed.",
    )
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    from datetime import datetime as _dt
    _log_dir = Path("logs")
    _log_dir.mkdir(exist_ok=True)
    _run_ts   = _dt.now().strftime("%Y%m%d_%H%M%S")
    _log_file = _log_dir / f"run_{_run_ts}.log"

    # Console handler — user-controlled level, terse format
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(getattr(logging, args.log_level))
    _console_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)-20s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    ))

    # File handler — always DEBUG, full detail
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    _root_logger = logging.getLogger()
    _root_logger.setLevel(logging.DEBUG)
    _root_logger.addHandler(_console_handler)
    _root_logger.addHandler(_file_handler)

    # Suppress noisy HTTP-layer debug output from Anthropic SDK internals
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # JSON prompt/response log — one file per run, same timestamp as the text log
    from core.llm_logger import LLMLogger
    _prompts_file = _log_dir / f"prompts_{_run_ts}.json"
    _llm_logger   = LLMLogger(_prompts_file)

    # Write run banner directly to the log file
    _banner_lines = [
        "=" * 80,
        "AGENTIC SYSID — PIPELINE RUN LOG",
        f"Started : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model   : {args.model}",
        f"Seed    : {args.seed if args.seed is not None else 42}",
        f"Budget  : {args.budget if args.budget is not None else 'default'}",
        "=" * 80,
        "",
        "PLANT DESCRIPTION:",
        (args.desc or DEFAULT_DESCRIPTION).strip(),
        "",
        "=" * 80,
        "",
    ]
    _log_file.write_text("\n".join(_banner_lines) + "\n", encoding="utf-8")

    # ── API key check ─────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Export it before running:\n"
            "    export ANTHROPIC_API_KEY='sk-ant-...'\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    plant_cfg   = cfg.get("plant", {})
    exp_cfg     = cfg.get("experiment", {})
    contract_d  = cfg.get("contract", {})

    seed        = args.seed   if args.seed   is not None else 42
    budget      = args.budget if args.budget is not None else exp_cfg.get("total_budget", 200.0)
    plant_desc  = args.desc   if args.desc   is not None else DEFAULT_DESCRIPTION
    data_dir    = Path(args.data_dir)

    # ── Infrastructure ────────────────────────────────────────────────────────
    from tools.budget_manager import BudgetManager
    from tools.experiment_db import ExperimentDatabase
    from tools.model_registry import ModelRegistry
    from tools.plant_api import PlantAPI
    from core.schemas import (
        Assets, Budget, Dossier, EntryPath, PlantContract,
    )

    # ── Plant loading: dynamic (from config) or default (built-in pendulum) ──
    plant_class_str = plant_cfg.get("class", None)
    if plant_class_str:
        import importlib
        module_path, class_name = plant_class_str.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            PlantClass = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            print(f"ERROR: Cannot load plant class '{plant_class_str}': {exc}", file=sys.stderr)
            sys.exit(1)
        plant_kwargs = {k: v for k, v in plant_cfg.items() if k != "class"}
        plant = PlantClass(seed=seed, **plant_kwargs)
        print(f"  Plant:   {plant_class_str}")
    else:
        from plants.inverted_pendulum import PendulumPlant, PendulumParams
        params = PendulumParams(
            J=plant_cfg.get("J", 0.05),
            m=plant_cfg.get("m", 0.5),
            L=plant_cfg.get("L", 0.30),
            b_v=plant_cfg.get("b_v", 0.02),
            f_c=plant_cfg.get("f_c", 0.08),
            noise_std=plant_cfg.get("noise_std", 0.001),
            coulomb_smooth_eps=plant_cfg.get("coulomb_smooth_eps", 0.01),
        )
        plant = PendulumPlant(params=params, seed=seed)
        print(f"  Plant:   plants.inverted_pendulum.PendulumPlant (built-in demo)")

    input_limits_raw = contract_d.get("input_limits", {"torque": [-2.0, 2.0]})
    input_limits     = {k: tuple(v) for k, v in input_limits_raw.items()}

    contract = PlantContract(
        name=contract_d.get("name", "pendulum_with_friction"),
        input_names=contract_d.get("input_names", ["torque"]),
        output_names=contract_d.get("output_names", ["angle"]),
        state_names=contract_d.get("state_names", ["theta", "theta_dot"]),
        input_limits=input_limits,
        sample_time=contract_d.get("sample_time", 0.02),
        x0=contract_d.get("x0", None),
        is_unstable=contract_d.get("is_unstable", False),
        description=contract_d.get("description", ""),
    )

    budget_mgr = BudgetManager(total=budget)
    db         = ExperimentDatabase(
        db_path=str(data_dir / "experiment.db"),
        data_dir=str(data_dir / "runs"),
    )
    registry   = ModelRegistry(str(data_dir / "models"))
    api        = PlantAPI(plant, contract, budget_mgr, db, experiment_cost=2.0)

    # ── Agent instantiation ───────────────────────────────────────────────────
    from agents.intake import IntakeAgent
    from agents.modeler import ModelerAgent
    from agents.experiment_planner import ExperimentPlannerAgent
    from agents.estimator import EstimatorAgent
    from agents.validation import ValidationAgent
    from agents.greybox.agent import GreyBoxAgent
    from agents.surrogate.agent import SurrogateAgent
    from agents.ship import ShipAgent
    from core.router_agent import RouterAgent
    from memory import RetrievalService
    from core.orchestrator import (
        INTAKE, MODELER, EXPERIMENT_PLANNER, ESTIMATOR, VALIDATION,
        GREYBOX_SO, SURROGATE_SO, SHIP, ROUTER,
        _route_after_intake, _route_after_modeler, _route_after_experiment_planner,
        _route_after_estimator, _route_after_validation, _route_after_suborch,
        _route_after_router,
    )

    retrieval = RetrievalService(data_dir=str(data_dir))
    print(f"  Memory:  {retrieval.stats()['episodic_runs']} prior run(s), "
          f"{retrieval.stats()['document_chunks']} doc chunk(s)")

    agents = {
        INTAKE:             IntakeAgent(registry, budget_total=budget,
                                        model=args.model, api_key=api_key,
                                        llm_logger=_llm_logger),
        MODELER:            ModelerAgent(registry, model=args.model, api_key=api_key,
                                         retrieval_service=retrieval,
                                         llm_logger=_llm_logger),
        EXPERIMENT_PLANNER: ExperimentPlannerAgent(model=args.model, api_key=api_key),
        ESTIMATOR:          EstimatorAgent(api, registry, db, n_samples=600,
                                           retrieval_service=retrieval),
        VALIDATION:         ValidationAgent(api, registry, db),
        ROUTER:             RouterAgent(model=args.model, api_key=api_key),
        GREYBOX_SO:         GreyBoxAgent(api, registry, db,
                                         model=args.model, api_key=api_key),
        SURROGATE_SO:       SurrogateAgent(api, registry, db,
                                           model=args.model, api_key=api_key),
        SHIP:               ShipAgent(registry, retrieval_service=retrieval),
    }

    routers = {
        INTAKE:             _route_after_intake,
        MODELER:            _route_after_modeler,
        EXPERIMENT_PLANNER: _route_after_experiment_planner,
        ESTIMATOR:          _route_after_estimator,
        VALIDATION:         _route_after_validation,
        ROUTER:             _route_after_router,
        GREYBOX_SO:         _route_after_suborch,
        SURROGATE_SO:       _route_after_suborch,
    }

    # ── Initial dossier ───────────────────────────────────────────────────────
    initial_dossier = Dossier(
        entry_path=EntryPath.WHITE_BOX,
        budget=Budget(total=budget),
        status=plant_desc,   # IntakeAgent reads description from dossier.status
        assets=Assets(),
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Agentic SysID — System Identification")
    print("=" * 60)
    print(f"  Budget:  {budget:.0f} units")
    print(f"  Seed:    {seed}")
    print(f"  Model:   {args.model}")
    print(f"  Data:    {data_dir.resolve()}")
    print(f"  Log:     {_log_file.resolve()}")
    print(f"  Prompts: {_prompts_file.resolve()}")
    print("=" * 60)
    print()

    try:
        final = _run_pipeline(initial_dossier, agents, routers, ship_node=SHIP,
                              budget_mgr=budget_mgr)
    finally:
        _llm_logger.close()

    # ── Results ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Pipeline complete")
    print("=" * 60)

    if final.last_report:
        print(f"  {final.last_report.summary}")

    # Use the actually-shipped model (best by val RMSE), not the last estimated one.
    shipped_meta = final.last_report.metadata if final.last_report else {}
    model_id = shipped_meta.get("model_id") or final.artifacts.best_model_id or final.artifacts.current_model_id
    shipped_rung = shipped_meta.get("rung", final.current_rung.value)
    if model_id:
        try:
            model = registry.load_model(model_id)
            print(f"  Model type : {model.model_type.value}")
            print(f"  Rung       : {shipped_rung}")
            print(f"  Model ID   : {model_id}")
            if model.parameters:
                print("  Parameters :")
                for k, v in model.parameters.items():
                    print(f"    {k:12s} = {v:.4f}")
        except Exception:
            pass

    # Find the verdict that corresponds to the shipped (best) model.
    best_rmse = final.artifacts.best_val_rmse
    best_attempt = None
    if best_rmse is not None and final.attempt_log:
        for a in final.attempt_log:
            if a.val_rmse is not None and abs(a.val_rmse - best_rmse) < 1e-6:
                best_attempt = a
                break
    if best_attempt is not None:
        print(f"  Verdict    : {'pass' if best_rmse < 0.05 else 'fail'} / {best_attempt.gap_type}")
        print(f"  RMSE       : {best_rmse:.4f} rad")
    elif final.last_verdict:
        verdict = final.last_verdict
        print(f"  Verdict    : {verdict.verdict.value} / {verdict.gap_type.value}")
        rmse = verdict.metrics.get("rmse", None)
        if rmse is not None:
            print(f"  RMSE       : {rmse:.4f} rad")

    print(f"  Budget used: {final.budget.spent:.1f} / {final.budget.total:.1f}")
    print("=" * 60)

    # ── Demo visualization ────────────────────────────────────────────────────
    if not args.no_demo and model_id:
        demo_out = args.demo_output or str(Path("demo") / "demo.gif")
        demo_path = Path(demo_out)
        demo_path.parent.mkdir(parents=True, exist_ok=True)

        print()
        print(f"  Generating demo animation → {demo_path} ...")
        try:
            import matplotlib
            matplotlib.use("Agg")   # must be set before pyplot is imported
            from visualization.demo import run_demo_simulation, animate_demo

            demo_data = run_demo_simulation(model_id=model_id, registry=registry)
            animate_demo(demo_data, save_path=str(demo_path))
            print(f"  Demo saved  → {demo_path.resolve()}")
        except Exception as _demo_exc:
            print(f"  [warn] Demo generation failed: {_demo_exc}")

    print()
    print(f"Done. Latest log: {_log_file}")


if __name__ == "__main__":
    main()

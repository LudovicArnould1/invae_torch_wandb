"""
Script to initialize and run W&B sweeps for hyperparameter optimization.

Usage:
    # Initialize and run sweep with default config
    python run_sweep.py
    
    # Run with custom sweep config
    python run_sweep.py --sweep-config path/to/sweep_config.yaml
    
    # Run specific sweep ID
    python run_sweep.py --sweep-id your-sweep-id
    
    # Run with multiple agents (parallel)
    python run_sweep.py --count 3
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import wandb


def initialize_sweep(
    sweep_config_path: str = "src/homework_scientalab/config/sweep_config.yaml",
    project: str = "invae-pancreas",
    entity: str | None = None,
) -> str:
    """
    Initialize a W&B sweep.
    
    Args:
        sweep_config_path: Path to sweep configuration YAML
        project: W&B project name
        entity: W&B entity (username/team)
        
    Returns:
        Sweep ID string
    """
    # Load sweep configuration
    config_path = Path(sweep_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_config_path}")
    
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
    
    print("=" * 80)
    print("INITIALIZING W&B SWEEP")
    print("=" * 80)
    print(f"Project: {project}")
    print(f"Entity: {entity or 'default'}")
    print(f"Method: {sweep_config.get('method', 'random')}")
    print(f"Metric: {sweep_config.get('metric', {}).get('name', 'N/A')}")
    print(f"Goal: {sweep_config.get('metric', {}).get('goal', 'N/A')}")
    print("\nParameters to sweep:")
    for param, config in sweep_config.get("parameters", {}).items():
        if "values" in config:
            print(f"  {param}: {config['values']}")
        elif "min" in config and "max" in config:
            print(f"  {param}: [{config['min']}, {config['max']}] ({config.get('distribution', 'uniform')})")
    print("=" * 80)
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity,
    )
    
    print(f"\n✓ Sweep initialized with ID: {sweep_id}")
    print(f"  View at: https://wandb.ai/{entity or 'your-entity'}/{project}/sweeps/{sweep_id}")
    
    return sweep_id


def run_sweep_agent(
    sweep_id: str,
    project: str = "invae-pancreas",
    entity: str | None = None,
    count: int = 3,
) -> None:
    """
    Run a W&B sweep agent.
    
    Args:
        sweep_id: Sweep ID returned by initialize_sweep()
        project: W&B project name
        entity: W&B entity (username/team)
        count: Number of runs to execute (None for infinite)
    """
    print("\n" + "=" * 80)
    print("STARTING SWEEP AGENT")
    print("=" * 80)
    print(f"Sweep ID: {sweep_id}")
    print(f"Number of runs: {count}")
    print("=" * 80 + "\n")
    
    # Import the sweep training function
    from sweep_train import sweep_train
    
    # Construct full sweep ID if needed
    if "/" not in sweep_id:
        sweep_id = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
    
    # Run sweep agent
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=count,
        project=project,
        entity=entity,
    )
    
    print("\n" + "=" * 80)
    print("SWEEP AGENT COMPLETED")
    print("=" * 80)


def main():
    """Main entry point for sweep runner."""
    parser = argparse.ArgumentParser(
        description="Initialize and run W&B sweeps for inVAE hyperparameter optimization"
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default="src/homework_scientalab/config/sweep_config.yaml",
        help="Path to sweep configuration YAML",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to run (skip initialization)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="invae-pancreas",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username/team)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of sweep runs to execute",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize sweep, don't run agent",
    )
    
    args = parser.parse_args()
    
    # Initialize or use existing sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep: {sweep_id}")
    else:
        sweep_id = initialize_sweep(
            sweep_config_path=args.sweep_config,
            project=args.project,
            entity=args.entity,
        )
    
    # Run sweep agent unless --init-only
    if not args.init_only:
        run_sweep_agent(
            sweep_id=sweep_id,
            project=args.project,
            entity=args.entity,
            count=args.count,
        )
    else:
        print("\nSweep initialized. To run the agent, use:")
        print(f"  wandb agent {sweep_id}")
        print(f"  OR")
        print(f"  python run_sweep.py --sweep-id {sweep_id} --count {args.count}")


if __name__ == "__main__":
    main()


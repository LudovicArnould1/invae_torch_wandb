"""
homework_scientalab: inVAE implementation for batch correction in single-cell RNA-seq.

Main components:
- config: Configuration management with YAML support
- data: Data loading and preprocessing
- model: inVAE neural network architecture
- train: Training pipeline with W&B integration
- pipeline: Modular pipeline with separate stages
- reproducibility: Seed setting and environment tracking
"""
from homework_scientalab.monitor_and_setup.reproducibility import set_seed, get_environment_info
from homework_scientalab.pipeline import (
    Pipeline,
    PipelineStage,
    DataPreparationStage,
    TrainingStage,
    EvaluationStage,
    StageOutput,
    run_pipeline,
)

__version__ = "0.1.0"

__all__ = [
    "set_seed",
    "get_environment_info",
    "Pipeline",
    "PipelineStage",
    "DataPreparationStage",
    "TrainingStage",
    "EvaluationStage",
    "StageOutput",
    "run_pipeline",
]


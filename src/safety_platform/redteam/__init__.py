"""Red-Team Engine foundation: experiment lifecycle and attack state.

State-only milestone; later milestones add dataset, execution, and
orchestration on top of these models.
"""

from .analyzer import AnalysisSummary, summarize
from .dataset import (
    AttackCase,
    AttackDataset,
    DatasetConflictError,
    DuplicateAttackError,
    case_fingerprint,
    known_categories,
    register_category,
)
from .evaluation import EvaluationReport, compare_models, run_evaluation
from .executor import AttackExecutor, AttackResult, classify, make_defense_fn
from .groq import GroqClient, GroqError, MissingApiKeyError
from .mutation import (
    MutationResult,
    generate_variants,
    known_strategies,
    register_strategy,
)
from .orchestrator import RedTeamOrchestrator
from .selection import (
    SelectionResult,
    known_selection_strategies,
    register_selection_strategy,
    select_attacks,
)
from .state import (
    AttackState,
    ExperimentConfig,
    ExperimentState,
    InvalidTransitionError,
    LifecycleStatus,
)

__all__ = [
    "AnalysisSummary",
    "AttackCase",
    "AttackDataset",
    "AttackExecutor",
    "AttackResult",
    "AttackState",
    "DatasetConflictError",
    "DuplicateAttackError",
    "EvaluationReport",
    "ExperimentConfig",
    "ExperimentState",
    "GroqClient",
    "GroqError",
    "InvalidTransitionError",
    "LifecycleStatus",
    "MissingApiKeyError",
    "MutationResult",
    "RedTeamOrchestrator",
    "SelectionResult",
    "case_fingerprint",
    "classify",
    "compare_models",
    "generate_variants",
    "known_categories",
    "known_selection_strategies",
    "known_strategies",
    "make_defense_fn",
    "register_category",
    "register_selection_strategy",
    "register_strategy",
    "run_evaluation",
    "select_attacks",
    "summarize",
]

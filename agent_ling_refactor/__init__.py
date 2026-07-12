"""Faster, natural-message orchestration for Agent Ling.

The refactored package lives beside the original implementation so it can be
evaluated and rolled back independently.  Stable task, action, persistence and
Star Protocol adapters are reused; orchestration, prompts and conversation flow
are new.
"""

from .app import RefactoredApplication, create_refactored_runtime
from .runtime import RefactoredRuntime
from .settings import RefactorSettings, load_refactor_settings

__all__ = [
    "RefactoredApplication",
    "RefactoredRuntime",
    "RefactorSettings",
    "create_refactored_runtime",
    "load_refactor_settings",
]

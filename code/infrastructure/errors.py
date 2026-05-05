"""
Stage 9 exception hierarchy.

Every custom exception in Stage 9 inherits from Stage9Error, which inherits
from plain Exception (NOT from KeyError / ValueError / other built-ins).

Why not built-ins: a caller writing `except KeyError: val = default` would
silently swallow a missing-param bug — exactly the failure mode Principle 1
(zero silent defaults, loud failures) exists to prevent. Subclassing plain
Exception forces callers to catch the specific class by name.
"""


class Stage9Error(Exception):
    """Base class for all Stage 9 custom exceptions."""


class UnknownParamError(Stage9Error):
    """
    Raised when a param_name is not present in tenant_learning_params
    (either not seeded, or missing from an in-memory TenantParams snapshot).
    """


class LockAcquisitionError(Stage9Error):
    """
    Raised when the Redis run lock cannot be acquired or released because of
    an infrastructure problem (Redis unreachable, auth failure). NOT raised
    when a lock is simply held by another run — that returns None from
    acquire_lock() per contract.
    """


class RunAlreadyInProgressError(Stage9Error):
    """
    Raised by run() when another run for the same tenant already holds the
    Redis NX lock. No DB row is written — the run never started.
    """


class InvalidStateTransitionError(Stage9Error):
    """
    Raised by transition() when the (current → next) pair is absent from
    VALID_TRANSITIONS. Signals a programming error in the calling layer.
    No DB row is written when this is raised.
    """


class TenantParamNotFoundError(Stage9Error):
    """
    Raised by the PERCEIVING handler when tenant_learning_params has no rows
    for the tenant. Propagates to run()'s except block, which transitions the
    run to FAILED and then re-raises.
    """

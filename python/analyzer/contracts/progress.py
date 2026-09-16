"""Contract for progress reported during a long engine call.

Pose extraction runs for seconds to minutes. Without progress the desktop app
can only show an indeterminate spinner, which cannot distinguish slow work from
a hung worker -- and telling those apart is exactly what a user needs when a
job is taking longer than they expected.

Progress arrives as JSON-RPC notifications, which carry no `id` of their own, so
`request_id` is carried in the payload instead. A client with two calls in
flight can then attribute each event to the call that produced it rather than to
whichever happens to be showing a spinner.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Bump on a breaking change. The desktop app ignores events it cannot parse
# rather than failing the call they belong to: losing progress is a cosmetic
# problem, and failing a completed extraction over it would not be.
PROGRESS_SCHEMA_VERSION = 1

# The RPC method name used for these notifications.
PROGRESS_NOTIFICATION = "progress"


class ProgressUpdate(BaseModel):
    """One progress update from a running engine method.

    Named `ProgressUpdate` rather than the more obvious `ProgressEvent` because
    the generated TypeScript would otherwise collide with the DOM's built-in
    `ProgressEvent`, and a type that shadows a global in some files but not
    others is a trap for later.
    """

    schema_version: int = PROGRESS_SCHEMA_VERSION
    request_id: int | str | None = Field(
        default=None,
        description="The request this belongs to. None when the work was not RPC-driven.",
    )
    task: str = Field(description="Engine method doing the work, e.g. 'extract_poses'.")
    stage: str = Field(description="Phase within the task, e.g. 'estimating'.")
    current: int = Field(description="Units completed.")
    total: int | None = Field(
        default=None,
        description="Units expected, when known. None means the total cannot be determined.",
    )
    elapsed_s: float
    detail: str | None = Field(
        default=None, description="Short human-readable note, e.g. the model in use."
    )

    @property
    def fraction(self) -> float | None:
        """Completed proportion, or None when there is no total to divide by.

        Deliberately not a stored field: a progress bar that invents a
        denominator is worse than one that admits it does not have one.
        """
        if self.total is None or self.total <= 0:
            return None
        return min(self.current / self.total, 1.0)

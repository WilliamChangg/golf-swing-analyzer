"""Where a camera was put: the one piece of vocabulary two layers both need.

`CameraRole` began in `contracts/projects.py`, which is where it is used most and
where a reader would look for it first. Phase 8 gave it a second owner: a
calibration describes one camera, and the only name this system has for "which
camera" is its role.

That made the two modules mutually dependent -- a project holds a `CameraRig`,
and a rig's calibrations are keyed by role -- so the shared term moved down
here, below both. `projects` re-exports it, so nothing that already imported it
from there has to change and no caller has to learn a new location for a name
that has not changed meaning.

A module for one enum is worth it when the alternative is a cycle. The
alternative dodges -- a forward reference resolved at import time, or the rig
kept off `Project` and carried beside it -- both work by making the relationship
harder to see, and the relationship is real: a project has cameras, and a
calibration is about one of them.
"""

from __future__ import annotations

from enum import StrEnum


class CameraRole(StrEnum):
    """Where a camera was put, as the user declared it.

    Deliberately the *declared* role, not the measured one. Phase 6's
    `CameraView` is measured from the shoulder line at address and is the
    authority on what the footage contains; this is the authority on what the
    user intended, and the interesting case is the two disagreeing.

    OTHER exists so a third camera, or a phone propped at an angle nobody would
    call either name, can still be part of a project instead of forcing a
    dishonest label.
    """

    FACE_ON = "face_on"
    DOWN_THE_LINE = "down_the_line"
    OTHER = "other"

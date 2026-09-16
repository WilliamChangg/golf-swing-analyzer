"""Environment probing: hardware, external tooling, and model availability.

Nothing in this package may report a capability it has not measured.
"""

from analyzer.environment.doctor import run_doctor

__all__ = ["run_doctor"]

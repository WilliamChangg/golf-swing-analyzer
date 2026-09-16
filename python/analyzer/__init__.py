"""Golf swing biomechanics analysis engine.

Layering (each layer depends only on those above it):

    contracts/    typed models shared with the desktop app
    paths/        filesystem layout resolution
    environment/  hardware, tooling and model probing
    dispatch/     method registry
    worker, cli   entry points

Later phases add ingestion, pose, filtering, phases, biomechanics, and coaching
as sibling packages. Golf-specific reasoning is confined to the biomechanics,
phases and coaching packages; everything below them is general computer vision.
"""

__version__ = "0.1.0"

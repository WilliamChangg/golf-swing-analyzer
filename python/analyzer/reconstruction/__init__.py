"""Multi-view 3D reconstruction: where two calibrated rays meet.

The layer every earlier phase qualified its output against. Phase 5 labelled its
angles as projections, Phase 6 tagged each with the view it was taken from, and
Phase 8 returned unit bearings and offered no function that turned one into a
point -- because one ray does not meet anything. Two do.

    pairing      resample the target clip onto the reference clock
    triangulate  DLT, refined on reprojection error; conditioning; uncertainty
    skeleton     bone length and symmetry -- the checks the residual cannot make
    reconstruct  the entry point, and the gates

The phase's whole argument is in `analyzer/contracts/reconstruction.py`: the
reprojection residual is blind along the epipolar line, so it is not the quality
of a reconstruction, and the two things that are -- the ray convergence angle and
the consistency of a bone that cannot change length -- are reported beside it.
"""

from analyzer.reconstruction.pairing import SampledTrack, ViewTrack, track_from
from analyzer.reconstruction.reconstruct import (
    ReconstructedSequence,
    ReconstructionError,
    measured_pixel_sigma,
    reconstruct_pair,
    require_stereo_rig,
)
from analyzer.reconstruction.triangulate import (
    StereoGeometry,
    TriangulationError,
    convergence_angles,
    positional_uncertainty,
    reprojection_errors,
    triangulate,
    velocity_from_image,
)

__all__ = [
    "ReconstructedSequence",
    "ReconstructionError",
    "SampledTrack",
    "StereoGeometry",
    "TriangulationError",
    "ViewTrack",
    "convergence_angles",
    "measured_pixel_sigma",
    "positional_uncertainty",
    "reconstruct_pair",
    "reprojection_errors",
    "require_stereo_rig",
    "track_from",
    "triangulate",
    "velocity_from_image",
]

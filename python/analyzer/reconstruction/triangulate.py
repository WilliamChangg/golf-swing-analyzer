"""Two rays, one point: the numerical core of reconstruction.

Three quantities come out of this module and they answer three different
questions. Keeping them apart is the whole of Phase 9's argument, so it is worth
naming them before any code:

    reprojection_px     how far the fitted point lands from the detections
    convergence_deg     the angle the two rays met at
    uncertainty_m       what the first two imply about where the point could be

The first is the residual, and it is **blind along the epipolar line** -- a
detection displaced towards or away from the other camera's epipole moves the
answer in depth and leaves the residual untouched. The second is the geometry,
and it is what decides how much damage that invisible displacement does. The
third combines them, and it is the one to quote.

## Linear, then refined

The linear method (Hartley's DLT) stacks two rows per view from `u * P[2] - P[0]`
and takes the null space. It is fast, closed-form and minimises an *algebraic*
quantity with no geometric meaning: each row is implicitly weighted by the
point's depth in that camera, so a point seen close by one camera and far by the
other -- which is every point in a face-on plus down-the-line rig, since the two
cameras stand at different distances -- is pulled towards the nearer camera's
answer.

So it is used as a seed and then refined by Gauss-Newton on the actual
reprojection error in both views. From a linear seed this converges in two or
three steps, because the residual is small and the problem is three unknowns
against four equations. `ReconstructionConfig.refine` turns it off, which exists
so the benchmark can measure what it bought rather than assert it.

## Where the uncertainty comes from

Given the Jacobian `J` of the four reprojected coordinates with respect to the
3D point, and a per-view pixel standard deviation `sigma`, the covariance of the
fitted point is `sigma^2 (J'J)^-1`. Its largest eigenvalue is the variance along
the worst direction, which for a stereo pair is essentially always the depth
direction, and that is what is reported.

Two things make this the right number rather than a decorative one. It carries
the convergence angle automatically -- `J'J` is nearly singular when the two rays
are nearly parallel, which is exactly the case the residual cannot see -- and its
`sigma` is **measured from the clip**, as the filter's own residual RMS, rather
than assumed. It is an upper bound on the fitted trajectory's error, because
smoothing averages several samples, and it is used as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analyzer.contracts.calibration import CameraIntrinsics, StereoCalibration

# Points nearer than this to a camera's optical centre are treated as behind it.
# Not zero: a point exactly at the centre divides by zero, and one a millimetre
# in front of a lens is not a thing a camera photographed.
_MIN_DEPTH_M = 1e-3


class TriangulationError(ValueError):
    """A reconstruction was asked for something it cannot compute."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class StereoGeometry:
    """Both cameras, expressed in the reference camera's frame.

    The reference camera sits at the origin looking down +z, which makes its
    projection matrix `K [I | 0]` and makes every reconstructed point a
    `LandmarkSpace.CAMERA` coordinate by construction rather than by a later
    transformation that could be applied twice or not at all.

    `rotation` and `translation` carry a point from the reference camera's frame
    into the target's, which is the convention `StereoCalibration` states and the
    one `tests/synthetic_board.StereoRig` builds. Getting it transposed produces
    a rig pointing the wrong way, which triangulates to points behind the camera
    rather than to an error -- so `from_rig` is the only way to build one from
    stored data, and it does the composition once.
    """

    reference: CameraIntrinsics
    target: CameraIntrinsics
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]

    @classmethod
    def from_rig(
        cls,
        reference: CameraIntrinsics,
        target: CameraIntrinsics,
        stereo: StereoCalibration,
    ) -> StereoGeometry:
        """Build from a stored stereo calibration, in its own stated direction."""
        return cls(
            reference=reference,
            target=target,
            rotation=stereo.rotation_matrix(),
            translation=stereo.translation_vector(),
        )

    @property
    def target_centre(self) -> NDArray[np.float64]:
        """Where the target camera's optical centre sits, in the reference frame.

        From `X_target = R X_reference + t`, the target's centre is the reference
        point mapping to the target origin: `-R' t`. This is the second endpoint
        of every baseline and the thing every convergence angle is measured from.
        """
        return -self.rotation.T @ self.translation

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.translation))

    def projection_reference(self) -> NDArray[np.float64]:
        return self.reference.matrix() @ np.hstack(
            (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
        )

    def projection_target(self) -> NDArray[np.float64]:
        return self.target.matrix() @ np.hstack((self.rotation, self.translation.reshape(3, 1)))

    def project(self, points: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        """Where `(N, 3)` reference-frame points land in each image, in pixels.

        A pinhole projection with **no distortion term**, and that is correct
        rather than a simplification: the landmarks reaching this layer were
        undistorted in `pose/series.py`, below the filter, so the model that
        describes them is the pinhole one. Re-applying distortion here would
        distort points that have already had it removed.
        """
        return (
            _project(points, np.eye(3), np.zeros(3), self.reference),
            _project(points, self.rotation, self.translation, self.target),
        )


def _project(
    points: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> NDArray[np.float64]:
    """`(N, 3)` points to `(N, 2)` pixels, NaN where the point is behind the camera."""
    camera = points @ rotation.T + translation
    depth = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics.fx * camera[:, 0] / depth + intrinsics.cx
        v = intrinsics.fy * camera[:, 1] / depth + intrinsics.cy
    behind = ~np.isfinite(depth) | (depth <= _MIN_DEPTH_M)
    return np.stack((np.where(behind, np.nan, u), np.where(behind, np.nan, v)), axis=-1)


def _jacobian(
    points: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> NDArray[np.float64]:
    """d(pixel) / d(reference-frame point), as `(N, 2, 3)`.

    The chain rule through the projection: with `Y = R X + t`,

        du/dY = [fx/Y2,  0,     -fx Y0 / Y2^2]
        dv/dY = [0,      fy/Y2, -fy Y1 / Y2^2]

    and `dY/dX = R`. The third column is where the conditioning lives -- it is
    what makes `J'J` nearly singular when a point sits far along an axis that
    neither camera resolves.
    """
    camera = points @ rotation.T + translation
    depth = camera[:, 2]
    safe = np.where(np.abs(depth) < _MIN_DEPTH_M, np.nan, depth)

    inner = np.zeros((points.shape[0], 2, 3), dtype=np.float64)
    inner[:, 0, 0] = intrinsics.fx / safe
    inner[:, 0, 2] = -intrinsics.fx * camera[:, 0] / (safe * safe)
    inner[:, 1, 1] = intrinsics.fy / safe
    inner[:, 1, 2] = -intrinsics.fy * camera[:, 1] / (safe * safe)
    return inner @ rotation


def stacked_jacobian(points: NDArray[np.float64], geometry: StereoGeometry) -> NDArray[np.float64]:
    """Both views' Jacobians as one `(N, 4, 3)` block.

    The four rows are the two cameras' u and v. Everything downstream -- the
    Gauss-Newton step, the covariance, the 3D velocity -- is a least-squares
    problem against this one matrix, which is why it is built once and shared
    rather than re-derived in three places that could disagree about a sign.
    """
    return np.concatenate(
        (
            _jacobian(points, np.eye(3), np.zeros(3), geometry.reference),
            _jacobian(points, geometry.rotation, geometry.translation, geometry.target),
        ),
        axis=1,
    )


def triangulate_linear(
    reference_px: NDArray[np.float64],
    target_px: NDArray[np.float64],
    geometry: StereoGeometry,
) -> NDArray[np.float64]:
    """Hartley's DLT: the null space of the stacked epipolar constraints.

    For each view and each detection `(u, v)`, a point `X` on the ray satisfies
    `u * P[2] . X = P[0] . X` and likewise for `v`, giving two rows of a 4x4
    homogeneous system whose null space is the point. Solved by SVD, batched over
    every landmark-frame at once.

    NaN in, NaN out. A landmark either camera did not supply cannot be
    triangulated, and returning a coordinate for it would make an absence look
    like a measurement -- the same rule `undistort_pixels` follows one layer down.
    """
    reference_px = np.asarray(reference_px, dtype=np.float64).reshape(-1, 2)
    target_px = np.asarray(target_px, dtype=np.float64).reshape(-1, 2)
    if reference_px.shape != target_px.shape:
        raise TriangulationError(
            f"The two views supplied {reference_px.shape[0]} and {target_px.shape[0]} "
            "points; triangulation pairs them one for one."
        )

    count = reference_px.shape[0]
    result = np.full((count, 3), np.nan, dtype=np.float64)
    usable = np.isfinite(reference_px).all(axis=1) & np.isfinite(target_px).all(axis=1)
    if not np.any(usable):
        return result

    first = geometry.projection_reference()
    second = geometry.projection_target()

    a = reference_px[usable]
    b = target_px[usable]
    system = np.empty((int(usable.sum()), 4, 4), dtype=np.float64)
    system[:, 0] = a[:, 0:1] * first[2] - first[0]
    system[:, 1] = a[:, 1:2] * first[2] - first[1]
    system[:, 2] = b[:, 0:1] * second[2] - second[0]
    system[:, 3] = b[:, 1:2] * second[2] - second[1]

    # Rows scaled to unit norm before the SVD. Without it the two views'
    # contributions are weighted by their pixel magnitudes, so a point near the
    # frame corner of a 4K clip outweighs one near the centre by a factor of a
    # thousand -- an arbitrary weighting that has nothing to do with either
    # measurement's quality.
    norms = np.linalg.norm(system, axis=2, keepdims=True)
    system = np.divide(system, norms, out=np.zeros_like(system), where=norms > 0.0)

    _, _, right = np.linalg.svd(system)
    homogeneous = right[:, 3, :]
    scale = homogeneous[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        points = homogeneous[:, :3] / scale[:, None]
    result[usable] = np.where(np.isfinite(points), points, np.nan)
    return result


def refine(
    points: NDArray[np.float64],
    reference_px: NDArray[np.float64],
    target_px: NDArray[np.float64],
    geometry: StereoGeometry,
    *,
    iterations: int = 10,
    tolerance_m: float = 1e-7,
) -> NDArray[np.float64]:
    """Gauss-Newton on the reprojection error in both views.

    Minimises the geometric quantity the DLT only approximates. Each step solves
    `(J'J) d = -J' r` for a 3-vector correction, batched over every point; the
    normal equations are used rather than a per-point least squares for the same
    reason `filtering/localpoly.py` uses them, and with the same fallback to the
    pseudo-inverse when a batch contains a singular system.

    A point that is behind either camera, or whose step does not resolve, keeps
    its seed rather than wandering: the refinement is an improvement on a working
    answer and is never allowed to turn one into a worse answer.
    """
    current = np.array(points, dtype=np.float64, copy=True)
    live = np.isfinite(current).all(axis=1)
    if not np.any(live):
        return current

    observed = np.concatenate((reference_px, target_px), axis=1)

    for _ in range(max(1, iterations)):
        active = np.flatnonzero(live)
        if active.size == 0:
            break

        candidate = current[active]
        first, second = geometry.project(candidate)
        residual = np.concatenate((first, second), axis=1) - observed[active]
        jacobian = stacked_jacobian(candidate, geometry)

        ok = np.isfinite(residual).all(axis=1) & np.isfinite(jacobian).all(axis=(1, 2))
        if not np.any(ok):
            break

        rows = active[ok]
        block = jacobian[ok]
        transposed = np.swapaxes(block, 1, 2)
        gram = transposed @ block
        rhs = transposed @ residual[ok][..., None]

        try:
            step = -np.linalg.solve(gram, rhs)[..., 0]
        except np.linalg.LinAlgError:
            # One ill-conditioned point must not take the clip's batch down. The
            # minimum-norm step is a sane answer for a direction the data does
            # not determine, and the convergence gate refuses the point anyway.
            step = -np.einsum("nij,nj->ni", np.linalg.pinv(gram), rhs[..., 0])

        moved = np.isfinite(step).all(axis=1)
        if not np.any(moved):
            break
        current[rows[moved]] += step[moved]
        if float(np.max(np.abs(step[moved]))) < tolerance_m:
            break

    # Anything the refinement pushed behind a camera reverts to its seed.
    first, second = geometry.project(current)
    broken = ~(np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1))
    current[broken & live] = points[broken & live]
    return current


def reprojection_errors(
    points: NDArray[np.float64],
    reference_px: NDArray[np.float64],
    target_px: NDArray[np.float64],
    geometry: StereoGeometry,
) -> NDArray[np.float64]:
    """Per-point RMS reprojection error over the two views, in pixels.

    RMS of the two Euclidean distances rather than their maximum, because both
    views are equally evidence and the optimiser distributed the miss between
    them; the maximum would report where the split happened to land.

    **What this number is not** is the accuracy of the point. See the module
    docstring: a detection displaced along the epipolar line produces a perfect
    residual at a wrong depth, so this measures the component of disagreement
    the geometry can express and nothing about the component it cannot.
    """
    first, second = geometry.project(points)
    a = np.linalg.norm(first - reference_px, axis=1)
    b = np.linalg.norm(second - target_px, axis=1)
    return np.sqrt((a * a + b * b) / 2.0)


def convergence_angles(
    points: NDArray[np.float64], geometry: StereoGeometry
) -> NDArray[np.float64]:
    """Angle between the two rays at each point, in degrees.

    **The gate.** The reference camera sits at the origin and the target at
    `geometry.target_centre`, so each is a direction from a known centre to the
    fitted point. Depth uncertainty scales as `1 / sin` of this angle: a pair
    converging at 90 degrees turns a pixel of invisible along-epipolar error into
    about a pixel's worth of depth error, and a pair at 5 degrees turns it into
    eleven times that.

    It is computed from the *fitted* point rather than from the cameras' optical
    axes because it is a property of where the point sat, not of where the
    cameras were aimed -- a point at the near edge of an overlapping field of
    view converges quite differently from one at the far edge.
    """
    towards_reference = -points
    towards_target = geometry.target_centre[None, :] - points

    first = np.linalg.norm(towards_reference, axis=1)
    second = np.linalg.norm(towards_target, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.einsum("ij,ij->i", towards_reference, towards_target) / (first * second)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def positional_uncertainty(
    points: NDArray[np.float64], geometry: StereoGeometry, sigma_px: float
) -> NDArray[np.float64]:
    """Standard deviation along the worst-determined direction, in metres.

    `sigma^2 (J'J)^-1` is the covariance of a least-squares fit whose
    observations carry independent noise of standard deviation `sigma`. Its
    largest eigenvalue is the variance along the worst direction, so the answer
    is `sigma / sqrt(lambda_min(J'J))` -- computed from the smallest eigenvalue
    directly rather than by inverting, which is both cheaper and better behaved
    exactly where it matters, on the nearly-singular systems a shallow
    convergence angle produces.

    Reported in the **worst** direction rather than as an average of the three,
    because the three are wildly unequal here. A stereo pair localises a point
    well across both rays and poorly along their bisector, and quoting the mean
    of a millimetre, a millimetre and four centimetres would describe a precision
    the reconstruction does not have in the direction anybody cares about.
    """
    jacobian = stacked_jacobian(points, geometry)
    finite = np.isfinite(jacobian).all(axis=(1, 2))

    result = np.full(points.shape[0], np.nan, dtype=np.float64)
    if not np.any(finite):
        return result

    gram = np.swapaxes(jacobian[finite], 1, 2) @ jacobian[finite]
    smallest = np.linalg.eigvalsh(gram)[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        result[finite] = sigma_px / np.sqrt(np.where(smallest > 0.0, smallest, np.nan))
    return result


def velocity_from_image(
    points: NDArray[np.float64],
    reference_velocity_px_s: NDArray[np.float64],
    target_velocity_px_s: NDArray[np.float64],
    geometry: StereoGeometry,
) -> NDArray[np.float64]:
    """3D velocity, in metres per second, from the two views' image velocities.

    **Not a finite difference of the reconstructed track**, and that is the same
    rule Phase 3 states: a derivative comes from the fit, never from
    differencing what the fit produced. Both views already carry a velocity the
    local polynomial fitted, so differentiating the least-squares condition
    `J' (proj(X) - x_obs) = 0` through time gives

        X_dot = (J'J)^-1 J' x_obs_dot

    which is the exact first-order velocity of the fitted point, assembled from
    quantities that were fitted rather than from a second numerical operation
    with its own noise response. It is consistent with the position by
    construction, which a difference of positions is not.

    The neglected term is `dJ/dt` acting on the residual, which is zero to the
    extent the fit converged -- so this is exact at a perfect reconstruction and
    degrades with the residual, not with the frame rate.
    """
    jacobian = stacked_jacobian(points, geometry)
    observed = np.concatenate((reference_velocity_px_s, target_velocity_px_s), axis=1)

    result = np.full_like(points, np.nan)
    finite = np.isfinite(jacobian).all(axis=(1, 2)) & np.isfinite(observed).all(axis=1)
    if not np.any(finite):
        return result

    block = jacobian[finite]
    transposed = np.swapaxes(block, 1, 2)
    gram = transposed @ block
    rhs = transposed @ observed[finite][..., None]
    try:
        result[finite] = np.linalg.solve(gram, rhs)[..., 0]
    except np.linalg.LinAlgError:
        result[finite] = np.einsum("nij,nj->ni", np.linalg.pinv(gram), rhs[..., 0])
    return result


def triangulate(
    reference_px: NDArray[np.float64],
    target_px: NDArray[np.float64],
    geometry: StereoGeometry,
    *,
    refine_points: bool = True,
    iterations: int = 10,
) -> NDArray[np.float64]:
    """Linear triangulation, refined unless asked not to. `(N, 2)` twice in, `(N, 3)` out."""
    seed = triangulate_linear(reference_px, target_px, geometry)
    if not refine_points:
        return seed
    return refine(
        seed,
        np.asarray(reference_px, dtype=np.float64).reshape(-1, 2),
        np.asarray(target_px, dtype=np.float64).reshape(-1, 2),
        geometry,
        iterations=iterations,
    )

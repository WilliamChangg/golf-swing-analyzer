/**
 * GENERATED FILE - DO NOT EDIT.
 *
 * Produced by scripts/gen_types.py from the Pydantic contracts in
 * python/analyzer/contracts/. To change these types, edit the Python models and
 * re-run `npm run gen:types`.
 */


export type { HealthStatus, EnvironmentReport, PlatformInfo, ComponentStatus, ComputeInfo } from "./EnvironmentReport";
export type { HashAlgorithm, TimestampSource, VideoMetadata, ContentKey, VideoStreamInfo, VideoTiming, IntervalStats } from "./VideoMetadata";
export type { PoseExtractionResult, PoseModelInfo, PoseExtractionStats } from "./PoseExtractionResult";
export type { ProgressUpdate } from "./ProgressUpdate";
export type { SwingEvent, SwingPhase, HandSource, SwingPhases, DetectedEvent, EventConfidence, DetectedPhase, HandSignalInfo, PhaseConfig } from "./SwingPhases";
export type { MetricName, MetricGroup, MetricUnit, MetricBasis, CameraView, BodySide, CalibrationStatus, MetricSet, Metric, MetricConfidence, RefusedMetric, ViewEstimate, LeadSide, RotationReference, FrameGeometry, MetricConfig } from "./MetricSet";
export type { SyncMethod, AnchorSource, SyncModel, ClipRef, TimeMap, SyncAnchor, AnchorResidual, SyncQuality, SyncConfidence, CorrelationReport, OverlapReport, SyncConfig } from "./SyncModel";
export type { CameraRole, DistortionModel, BoardFamily, CameraRig, CameraCalibration, CameraIntrinsics, CalibrationQuality, CoverageReport, DetectionReport, BoardSpec, BoardObservation, StereoCalibration, PairingReport, CalibrationConfig } from "./CameraRig";
export type { LandmarkSpace, ReconstructionReport, ReconstructionQuality, BoneConsistency, SymmetryCheck, PairingSummary, LandmarkReconstruction, ReconstructionConfig } from "./ReconstructionReport";
export type { ProjectList, Project, ProjectClip, ProjectSync } from "./ProjectList";
export type { Comparison, ThresholdMethod, FindingRefusal, PhrasingMode, CoachingReport, Finding, ThresholdSource, Evidence, RefusedFinding, PhrasingReport, GuardRejection, CoachingConfig } from "./CoachingReport";
export type { SeekIndex } from "./SeekIndex";
export type { Landmark, OverlayState, PoseOverlay, OverlayFrame, OverlayPoint, OverlayShaft } from "./PoseOverlay";
export type { RefusalReason, SceneCameraKind, ReconstructionScene, SceneFrame, ScenePoint, Vec3, PointUncertainty, SceneTrajectory, SceneCamera } from "./ReconstructionScene";
export type { Direction, DifferenceRefusal, TrajectoryChannel, SwingComparison, ClipSummary, PhaseClock, ClockKnot, CameraAgreement, MetricDifference, BracketTerm, RefusedDifference, TrajectoryComparison, ChannelSample, HandPathOverlay, PathSample, ComparisonConfig } from "./SwingComparison";

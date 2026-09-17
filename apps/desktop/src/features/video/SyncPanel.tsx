/**
 * Two-camera synchronisation: align a second clip against this one, then correct it.
 *
 * The panel is built around one picture — both clips drawn on a single clock —
 * because that is the only form in which an alignment can be agreed or disagreed
 * with. "Offset: −258 ms" is not a claim anyone can check by reading it; two
 * bars with the takeaway, top, impact and finish marked on each, lined up or
 * visibly not, is.
 *
 * **The residual is given as much room as the offset.** It is the one number
 * that answers a question the engine cannot otherwise answer at all: whether the
 * two clips show the same swing. Nothing here can see that two cameras were
 * pointed at one event, and a pair of different swings aligns perfectly happily
 * — what it cannot do is make their phase durations agree. So a residual well
 * above the frame-rate floor is displayed as a finding rather than as a
 * statistic, and the floor is shown beside it so the comparison needs no
 * arithmetic from the reader.
 *
 * **Manual anchors are offered after the automatic attempt, not instead of it.**
 * A person pinning the same instant in both clips is supplying the answer, and
 * the engine's job is then the error accounting rather than arbitration — but
 * the frame counts needed to bound a picker only arrive with the engine's first
 * reply, and a refusal carries them just as an alignment does.
 */

import type { EngineError, SyncModel, TimeMap } from "@gsa/types";
import { open } from "@tauri-apps/plugin-dialog";
import {
  ChevronLeft,
  ChevronRight,
  Link2,
  Loader2,
  Pin,
  Trash2,
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { EngineErrorPanel } from "@/components/engine-error-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { ManualAnchor } from "@/lib/ipc";
import { syncClips } from "@/lib/ipc";

const VIDEO_EXTENSIONS = ["mp4", "mov", "m4v", "avi", "mkv"];

type State =
  | { phase: "empty" }
  | { phase: "running"; targetPath: string }
  | { phase: "done"; result: SyncModel }
  | { phase: "failed"; targetPath: string; error: EngineError };

/**
 * The reference-clip time corresponding to a target-clip time.
 *
 * Mirrors `TimeMap.to_reference` in python/analyzer/contracts/sync.py. Kept to
 * the one direction the drawing needs rather than porting the pair: a second
 * copy of the map's algebra is a second place for its sign to be wrong, and
 * nothing in the WebView has any business inverting an uncertainty.
 */
function toReference(map: TimeMap, targetS: number): number {
  return map.pivot_s + (targetS - map.pivot_s - map.offset_s) / map.rate;
}

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

function formatMs(value: number | null | undefined, signed = false): string {
  if (value == null) return "—";
  return signed
    ? `${value >= 0 ? "+" : ""}${value.toFixed(1)} ms`
    : `${value.toFixed(1)} ms`;
}

function Field({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string | undefined;
}) {
  return (
    <div className="flex justify-between gap-4 text-sm">
      <dt className="text-muted-foreground shrink-0">{label}</dt>
      <dd className={`truncate font-mono ${tone ?? ""}`} title={value}>
        {value}
      </dd>
    </div>
  );
}

/**
 * Both clips on one axis, in reference-clip time.
 *
 * The target bar is drawn through the fitted map, so the picture *is* the
 * alignment rather than an illustration of it. Event markers come from the
 * anchors, which carry the instant in each clip's own clock; the target's are
 * mapped the same way the bar is, so an anchor that the fit does not explain
 * appears where it actually sits rather than where it was assumed to be.
 */
function AlignmentTimeline({ result }: { result: SyncModel }) {
  const map = result.time_map;
  if (!map) return null;

  const reference = result.reference;
  const target = result.target;

  const referenceStart = reference.start_s;
  const referenceEnd = reference.start_s + reference.duration_s;
  const targetStart = toReference(map, target.start_s);
  const targetEnd = toReference(map, target.start_s + target.duration_s);

  const start = Math.min(referenceStart, targetStart);
  const end = Math.max(referenceEnd, targetEnd);
  const span = end - start || 1;
  const percent = (value: number) => ((value - start) / span) * 100;

  const rows = [
    {
      key: "reference",
      name: reference.name,
      from: referenceStart,
      to: referenceEnd,
      bar: "bg-sky-400/70",
      marks: (result.anchors ?? []).map((anchor) => ({
        label: anchor.label,
        at: anchor.reference_s,
      })),
    },
    {
      key: "target",
      name: target.name,
      from: targetStart,
      to: targetEnd,
      bar: "bg-orange-400/70",
      marks: (result.anchors ?? []).map((anchor) => ({
        label: anchor.label,
        at: toReference(map, anchor.target_s),
      })),
    },
  ];

  return (
    <div className="space-y-3" data-testid="alignment-timeline">
      {rows.map((row) => (
        <div key={row.key} className="space-y-1">
          <div className="text-muted-foreground flex justify-between text-xs">
            <span className="truncate">{row.name}</span>
            <span className="font-mono">
              {row.from.toFixed(2)}–{row.to.toFixed(2)} s
            </span>
          </div>
          <div className="bg-muted relative h-7 overflow-hidden rounded">
            <div
              className={`absolute inset-y-0 ${row.bar}`}
              style={{
                left: `${String(percent(row.from))}%`,
                width: `${String(percent(row.to) - percent(row.from))}%`,
              }}
            />
            {row.marks.map((mark) => (
              <div
                key={`${row.key}-${mark.label}`}
                className="absolute inset-y-0 w-0.5 bg-slate-900/70"
                style={{ left: `${String(percent(mark.at))}%` }}
                title={`${mark.label} — ${mark.at.toFixed(3)} s`}
              />
            ))}
          </div>
        </div>
      ))}
      <p className="text-muted-foreground text-xs">
        Both clips in the reference clip&rsquo;s clock. Ticks are the instants
        the alignment was anchored on; where they line up vertically, the two
        cameras agree about when that instant happened.
      </p>
    </div>
  );
}

/** A frame stepper for one clip, bounded by the frame count the engine reported. */
function FramePicker({
  name,
  frames,
  value,
  onChange,
}: {
  name: string;
  frames: number;
  value: number;
  onChange: (next: number) => void;
}) {
  const clamp = (next: number) => Math.max(0, Math.min(frames - 1, next));
  return (
    <div className="space-y-1">
      <div className="text-muted-foreground truncate text-xs">{name}</div>
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          aria-label={`Previous frame in ${name}`}
          onClick={() => {
            onChange(clamp(value - 1));
          }}
          disabled={value <= 0}
        >
          <ChevronLeft />
        </Button>
        <input
          type="range"
          min={0}
          max={Math.max(0, frames - 1)}
          value={value}
          aria-label={`Frame in ${name}`}
          onChange={(event) => {
            onChange(clamp(Number(event.target.value)));
          }}
          className="flex-1"
        />
        <Button
          size="sm"
          variant="outline"
          aria-label={`Next frame in ${name}`}
          onClick={() => {
            onChange(clamp(value + 1));
          }}
          disabled={value >= frames - 1}
        >
          <ChevronRight />
        </Button>
        <span className="w-20 shrink-0 text-right font-mono text-sm">
          {value} / {frames - 1}
        </span>
      </div>
    </div>
  );
}

export function SyncPanel({
  referencePath,
  referenceSlowMotion = 1,
}: {
  referencePath: string;
  referenceSlowMotion?: number;
}) {
  const [state, setState] = useState<State>({ phase: "empty" });
  const [targetPath, setTargetPath] = useState<string | null>(null);
  const [targetSlowMotion, setTargetSlowMotion] = useState(1);
  const [anchors, setAnchors] = useState<ManualAnchor[]>([]);
  const [referenceFrame, setReferenceFrame] = useState(0);
  const [targetFrame, setTargetFrame] = useState(0);

  const clips = state.phase === "done" ? state.result : null;

  const run = useCallback(
    async (path: string, picks: ManualAnchor[]) => {
      setState({ phase: "running", targetPath: path });
      const result = await syncClips(referencePath, path, {
        referenceSlowMotion,
        targetSlowMotion,
        ...(picks.length > 0 ? { anchors: picks } : {}),
      });
      setState(
        result.ok
          ? { phase: "done", result: result.value }
          : { phase: "failed", targetPath: path, error: result.error },
      );
    },
    [referencePath, referenceSlowMotion, targetSlowMotion],
  );

  const choose = useCallback(async () => {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [{ name: "Video", extensions: VIDEO_EXTENSIONS }],
    });
    if (typeof selected !== "string") return;
    setTargetPath(selected);
    setAnchors([]);
    await run(selected, []);
  }, [run]);

  const pin = useCallback(() => {
    setAnchors((current) => [
      ...current,
      {
        // Numbered rather than named: the engine never interprets the label, and
        // asking for one before the instant is pinned is a dialog in the way of
        // a two-click action.
        label: `anchor ${String(current.length + 1)}`,
        referenceFrame,
        targetFrame,
      },
    ]);
  }, [referenceFrame, targetFrame]);

  const residualsByLabel = useMemo(() => {
    const map = new Map<string, number>();
    for (const entry of clips?.residuals ?? [])
      map.set(entry.label, entry.residual_ms);
    return map;
  }, [clips]);

  const quality = clips?.quality;
  const confidence = clips?.confidence;
  const map = clips?.time_map;

  // The comparison the whole panel exists to make legible: a residual measured
  // against what the two frame rates actually permit, rather than against zero.
  const residualRatio =
    quality?.residual_rms_ms != null && quality.quantisation_floor_ms > 0
      ? quality.residual_rms_ms / quality.quantisation_floor_ms
      : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>Second camera</span>
          {clips ? (
            <span
              className={`text-sm font-medium ${
                clips.aligned ? "text-status-ok" : "text-status-degraded"
              }`}
            >
              {clips.aligned ? "Aligned" : "Not aligned"}
            </span>
          ) : null}
        </CardTitle>
        <CardDescription>
          Relate a second clip&rsquo;s clock to this one. Both need their poses
          extracted first.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-end gap-3">
          <Button
            onClick={() => void choose()}
            size="sm"
            disabled={state.phase === "running"}
          >
            {state.phase === "running" ? (
              <Loader2 className="animate-spin" />
            ) : (
              <Link2 />
            )}
            {targetPath ? "Choose another clip" : "Choose second clip"}
          </Button>

          <label className="text-muted-foreground flex items-center gap-2 text-sm">
            Its slow motion
            <input
              type="number"
              min={1}
              step={1}
              value={targetSlowMotion}
              aria-label="Slow-motion factor of the second clip"
              onChange={(event) => {
                setTargetSlowMotion(Math.max(1, Number(event.target.value)));
              }}
              className="border-input w-20 rounded border px-2 py-1 font-mono"
            />
            ×
          </label>

          {targetPath ? (
            <span className="text-muted-foreground truncate text-sm">
              {basename(targetPath)}
            </span>
          ) : null}
        </div>

        {state.phase === "running" ? (
          <div className="text-muted-foreground flex items-center gap-3 py-6 text-sm">
            <Loader2 className="size-4 animate-spin" />
            Filtering both clips and locating their events.
          </div>
        ) : null}

        {state.phase === "failed" ? (
          <EngineErrorPanel
            error={state.error}
            onRetry={() => void run(state.targetPath, anchors)}
            retryLabel="Try again"
          />
        ) : null}

        {clips ? (
          <>
            {map ? <AlignmentTimeline result={clips} /> : null}

            <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
              <Field
                label="Offset"
                value={
                  map
                    ? `${formatMs(map.offset_s * 1000, true)}${
                        map.offset_uncertainty_s != null
                          ? ` ± ${formatMs(map.offset_uncertainty_s * 1000)}`
                          : ""
                      }`
                    : "refused"
                }
              />
              <Field label="Method" value={clips.method ?? "none"} />
              <Field
                label="Clock rate"
                value={
                  map?.rate_estimated
                    ? map.rate.toFixed(5)
                    : "1.0 (assumed, not measured)"
                }
                tone={map?.rate_estimated ? "text-status-degraded" : undefined}
              />
              <Field
                label="Frame-rate floor"
                value={formatMs(quality?.quantisation_floor_ms)}
              />
              <Field
                label="Anchor residual"
                value={
                  quality?.residual_rms_ms == null
                    ? "not measurable"
                    : `${formatMs(quality.residual_rms_ms)} rms${
                        residualRatio
                          ? ` (${residualRatio.toFixed(1)}× floor)`
                          : ""
                      }`
                }
                tone={
                  residualRatio && residualRatio > 3
                    ? "text-status-degraded"
                    : undefined
                }
              />
              <Field
                label="Confidence"
                value={
                  confidence
                    ? `${confidence.overall.toFixed(2)} = ${confidence.agreement.toFixed(2)} × ${confidence.anchors.toFixed(2)} × ${confidence.stability.toFixed(2)}`
                    : "—"
                }
              />
            </dl>

            {quality?.residual_rms_ms == null && clips.aligned ? (
              <p className="text-muted-foreground text-xs">
                No residual is reported because the fit has no spare degrees of
                freedom: it passes through every anchor by construction, so a
                residual of zero would mean nothing was checked.
              </p>
            ) : null}

            {clips.anchors && clips.anchors.length > 0 ? (
              <div className="space-y-1">
                <div className="text-muted-foreground text-xs">
                  Anchors, and how far each sits from the fitted map
                </div>
                <ul className="space-y-1 text-sm">
                  {clips.anchors.map((anchor) => (
                    <li
                      key={anchor.label}
                      className="flex justify-between gap-4 font-mono"
                    >
                      <span className="truncate">
                        {anchor.label}
                        <span className="text-muted-foreground">
                          {" "}
                          {anchor.reference_frame} → {anchor.target_frame}
                        </span>
                      </span>
                      <span>
                        {formatMs(residualsByLabel.get(anchor.label), true)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {clips.refusal ? (
              <p className="text-status-degraded text-sm">{clips.refusal}</p>
            ) : null}

            {clips.warnings && clips.warnings.length > 0 ? (
              <ul className="text-muted-foreground space-y-2 text-xs">
                {clips.warnings.map((warning) => (
                  <li key={warning} className="flex gap-2">
                    <span aria-hidden="true">&middot;</span>
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            ) : null}

            <div className="space-y-3 border-t pt-4">
              <div>
                <h4 className="text-sm font-medium">Correct it by hand</h4>
                <p className="text-muted-foreground mt-1 text-xs">
                  Step each clip to the same instant and pin it. Your picks
                  replace the detected events entirely — the engine keeps the
                  arithmetic and the error accounting, and does not second-guess
                  the instants.
                </p>
              </div>

              <FramePicker
                name={clips.reference.name}
                frames={clips.reference.frames}
                value={referenceFrame}
                onChange={setReferenceFrame}
              />
              <FramePicker
                name={clips.target.name}
                frames={clips.target.frames}
                value={targetFrame}
                onChange={setTargetFrame}
              />

              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="outline" onClick={pin}>
                  <Pin /> Pin this instant
                </Button>
                <Button
                  size="sm"
                  disabled={anchors.length === 0 || targetPath === null}
                  onClick={() => {
                    if (targetPath) void run(targetPath, anchors);
                  }}
                >
                  Re-align on {anchors.length} pick
                  {anchors.length === 1 ? "" : "s"}
                </Button>
                {anchors.length > 0 ? (
                  <Button
                    size="sm"
                    variant="outline"
                    aria-label="Clear pinned instants"
                    onClick={() => {
                      setAnchors([]);
                    }}
                  >
                    <Trash2 />
                  </Button>
                ) : null}
              </div>

              {anchors.length > 0 ? (
                <ul className="text-muted-foreground space-y-1 font-mono text-xs">
                  {anchors.map((anchor) => (
                    <li key={anchor.label}>
                      {anchor.label}: {anchor.referenceFrame} →{" "}
                      {anchor.targetFrame}
                    </li>
                  ))}
                </ul>
              ) : null}

              {anchors.length === 1 ? (
                <p className="text-muted-foreground text-xs">
                  One pick fixes the offset and nothing checks it. A second, far
                  from the first, gives the engine something to disagree with.
                </p>
              ) : null}
            </div>
          </>
        ) : null}

        {state.phase === "empty" ? (
          <p className="text-muted-foreground text-sm">
            Aligning two cameras is what Phases 8 and 9 build on: nothing can be
            triangulated from two views until they are known to be of the same
            instant.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

/**
 * Two clips, side by side, with everything that is not a swing difference removed.
 *
 * The sixth screen, and the second — after Three-D — whose subject is a pair.
 * It is separate from Swing for the reason Three-D is: the Swing screen works on
 * one loose file and everything on it is a statement about that file, whereas
 * every number here is a statement about two.
 *
 * **Both clips are chosen through the Rust picker, one at a time.** There is
 * still no command that takes a path, which is what keeps the asset scope Phase
 * 14 opened meaningful: the WebView cannot name a file it wants read. The second
 * pick replaces the second clip and leaves the first alone, so swapping one half
 * of a pair does not mean re-choosing both.
 *
 * **No video element.** Phase 14 put a player behind the frame indices because a
 * metric is checked against the frame it came from; two players would double
 * that and deliver neither, since the frames a difference cites are in two
 * different files and only one can be on screen. What this screen shows instead
 * is the frames, printed, so a reader can take them to the Swing screen — and the
 * trajectory plots, which are the part that genuinely needs two clips at once.
 *
 * Analysis is not automatic on the second pick, for the reason it is not on the
 * Swing screen: this is the longest call the app makes after a reconstruction,
 * because it runs the whole chain twice.
 */

import type { EngineError, ProgressUpdate, SwingComparison } from "@gsa/types";
import { FolderOpen, GitCompare, Loader2, Scale } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { EngineErrorPanel } from "@/components/engine-error-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { DifferencesPanel } from "@/features/compare/DifferencesPanel";
import { TrajectoryPlot } from "@/features/compare/TrajectoryPlot";
import {
  chooseClip,
  compareSwings,
  onProgress,
  progressFraction,
} from "@/lib/ipc";

type Side = "reference" | "target";

interface Inputs {
  windowS?: number;
  referenceSlowMotion: number;
  targetSlowMotion: number;
}

const DEFAULT_INPUTS: Inputs = {
  referenceSlowMotion: 1,
  targetSlowMotion: 1,
};

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

/**
 * One clip's knots, which is what the normalisation divided out.
 *
 * On screen rather than only in the contract, because the plots below have had
 * every timing difference removed from them by construction. A reader who sees
 * two curves reaching the top at the same place should be able to see, in the
 * same view, that one of them took half a second longer to get there.
 */
function ClockTable({ result }: { result: SwingComparison }) {
  const sides = [
    { label: basename(result.reference.path), summary: result.reference },
    { label: basename(result.target.path), summary: result.target },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2" data-testid="clock-table">
      {sides.map(({ label, summary }) => (
        <div key={summary.content_key.digest} className="space-y-2">
          <h4 className="truncate text-sm font-medium">{label}</h4>
          <dl className="text-muted-foreground space-y-1 text-xs">
            <div className="flex justify-between gap-2">
              <dt>view</dt>
              <dd className="font-mono">{summary.view}</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>slow motion</dt>
              <dd className="font-mono">{summary.slow_motion_factor}×</dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt>lens corrected</dt>
              <dd className="font-mono">
                {summary.calibration === "none" ? "no" : summary.calibration}
              </dd>
            </div>
          </dl>
          {summary.clock.usable ? (
            <table className="w-full text-xs">
              <thead className="text-muted-foreground">
                <tr>
                  <th className="text-left font-normal">event</th>
                  <th className="text-right font-normal">frame</th>
                  <th className="text-right font-normal">time</th>
                  <th className="text-right font-normal">±</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {(summary.clock.knots ?? []).map((knot) => (
                  <tr key={knot.event}>
                    <td className="font-sans">{knot.event}</td>
                    <td className="text-right">{knot.frame_index}</td>
                    <td className="text-right">
                      {knot.timestamp_s.toFixed(3)} s
                    </td>
                    <td className="text-right">
                      {(knot.ambiguity_s * 1000).toFixed(0)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-status-degraded text-xs">
              {summary.clock.methodology}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

export function CompareScreen() {
  const [paths, setPaths] = useState<Record<Side, string | null>>({
    reference: null,
    target: null,
  });
  const [inputs, setInputs] = useState<Inputs>(DEFAULT_INPUTS);
  const [result, setResult] = useState<SwingComparison | null>(null);
  const [busy, setBusy] = useState<{
    label: string;
    update: ProgressUpdate | null;
  } | null>(null);
  const [error, setError] = useState<EngineError | null>(null);

  const unlisten = useRef<(() => void) | null>(null);
  useEffect(
    () => () => {
      unlisten.current?.();
    },
    [],
  );

  const choose = useCallback(async (side: Side) => {
    setError(null);
    const chosen = await chooseClip();
    if (!chosen.ok) {
      setError(chosen.error);
      return;
    }
    if (chosen.value == null) return;
    setPaths((current) => ({ ...current, [side]: chosen.value }));
    setResult(null);
  }, []);

  const run = useCallback(async () => {
    const { reference, target } = paths;
    if (reference === null || target === null) return;

    setBusy({ label: "Analysing both clips", update: null });
    setError(null);
    unlisten.current = await onProgress((update) => {
      setBusy((current) => (current ? { ...current, update } : current));
    });
    try {
      const outcome = await compareSwings(reference, target, {
        ...(inputs.windowS === undefined ? {} : { windowS: inputs.windowS }),
        referenceSlowMotion: inputs.referenceSlowMotion,
        targetSlowMotion: inputs.targetSlowMotion,
      });
      if (outcome.ok) {
        setResult(outcome.value);
      } else {
        setError(outcome.error);
      }
    } catch (cause) {
      setError({
        kind: "transport",
        message: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      unlisten.current?.();
      unlisten.current = null;
      setBusy(null);
    }
  }, [inputs, paths]);

  const ready = paths.reference !== null && paths.target !== null;
  const percent = busy?.update ? progressFraction(busy.update) : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-lg font-semibold tracking-tight">Compare</h2>
          <p className="text-muted-foreground mt-1 text-sm">
            Two recordings on one clock. Everything that differs because of the
            camera rather than the swing is refused by name.
          </p>
        </div>
        <Button
          size="sm"
          onClick={() => void run()}
          disabled={!ready || busy !== null}
        >
          {busy ? <Loader2 className="animate-spin" /> : <GitCompare />}
          Compare
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>The pair</CardTitle>
          <CardDescription>
            Order decides the sign of every difference and nothing else. The
            smoothing window is shared, because smoothing two clips differently
            would move the features the comparison keys on; the slow-motion
            factors are per clip, because two recordings of one player are
            routinely not both slowed.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            {(["reference", "target"] as const).map((side) => (
              <div key={side} className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium capitalize">{side}</span>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => void choose(side)}
                    disabled={busy !== null}
                    data-testid={`choose-${side}`}
                  >
                    <FolderOpen /> Choose
                  </Button>
                </div>
                <p className="text-muted-foreground truncate text-sm">
                  {paths[side] === null
                    ? "No clip chosen."
                    : basename(paths[side])}
                </p>
                <label className="block text-xs">
                  <span className="text-muted-foreground">
                    Slow-motion factor
                  </span>
                  <input
                    type="number"
                    step="0.5"
                    min="0.1"
                    className="border-input mt-1 w-24 rounded-md border px-2 py-1 font-mono"
                    value={
                      side === "reference"
                        ? inputs.referenceSlowMotion
                        : inputs.targetSlowMotion
                    }
                    disabled={busy !== null}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      setInputs((current) =>
                        side === "reference"
                          ? { ...current, referenceSlowMotion: value }
                          : { ...current, targetSlowMotion: value },
                      );
                    }}
                  />
                </label>
              </div>
            ))}
          </div>

          <label className="block text-sm">
            <span className="text-muted-foreground block text-xs">
              Smoothing window (s), shared
            </span>
            <input
              type="number"
              step="0.01"
              min="0.01"
              className="border-input mt-1 w-28 rounded-md border px-2 py-1 font-mono"
              value={inputs.windowS ?? 0.1}
              disabled={busy !== null}
              onChange={(event) => {
                setInputs((current) => ({
                  ...current,
                  windowS: Number(event.target.value),
                }));
              }}
            />
          </label>
        </CardContent>
      </Card>

      {busy ? (
        <Card>
          <CardContent className="space-y-2 py-6">
            <div className="text-muted-foreground flex justify-between text-sm">
              <span>{busy.label}</span>
              <span className="font-mono">
                {percent === null
                  ? "—"
                  : `${String(Math.round(percent * 100))}%`}
              </span>
            </div>
            <div
              className="bg-muted h-2 w-full overflow-hidden rounded-full"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={
                percent === null ? undefined : Math.round(percent * 100)
              }
            >
              <div
                className="bg-primary h-full transition-[width] duration-200"
                style={{
                  width: `${String(Math.round((percent ?? 0) * 100))}%`,
                }}
              />
            </div>
          </CardContent>
        </Card>
      ) : null}

      {error ? (
        <EngineErrorPanel error={error} onRetry={() => void run()} />
      ) : null}

      {result === null && !busy ? (
        <Card>
          <CardContent className="text-muted-foreground flex flex-col items-center gap-3 py-12 text-center text-sm">
            <Scale className="size-8 opacity-40" aria-hidden="true" />
            <p className="max-w-lg">
              Choose two clips. Both need their poses extracted first — the
              Swing screen does that — and both need a detected swing with all
              four events, because the four events are the only instants two
              recordings are known to share.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {result ? (
        <Card>
          <CardHeader>
            <CardTitle>The clocks</CardTitle>
            <CardDescription>
              What the normalisation divided out. Putting both swings on one
              axis forces the four events to coincide, so nothing in the plots
              below can say that one player reached the top later — that
              difference is here, and in the timing rows of the panel
              underneath.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ClockTable result={result} />
          </CardContent>
        </Card>
      ) : null}

      {result?.computed ? (
        <Card>
          <CardHeader>
            <CardTitle>Trajectories</CardTitle>
            <CardDescription>
              Both swings on one axis: 0 at the takeaway, 1 at the top, 2 at
              impact, 3 at the finish. The shaded band is how far the curve can
              move on one frame of ambiguity at each event, so a gap inside it
              is not a difference between the swings.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            {(result.trajectories ?? []).map((channel) =>
              channel.refusal === null || channel.refusal === undefined ? (
                <div key={channel.channel} className="space-y-1">
                  <h4 className="text-sm font-medium">{channel.label}</h4>
                  <TrajectoryPlot
                    channel={channel}
                    referenceLabel={basename(result.reference.path)}
                    targetLabel={basename(result.target.path)}
                  />
                </div>
              ) : (
                <div key={channel.channel} className="space-y-1">
                  <h4 className="text-sm font-medium">{channel.label}</h4>
                  <p className="text-status-degraded text-sm">
                    {channel.reason}
                  </p>
                </div>
              ),
            )}
          </CardContent>
        </Card>
      ) : null}

      {result?.computed ? <DifferencesPanel result={result} /> : null}

      {result && !result.computed ? (
        <Card className="border-status-degraded/40">
          <CardHeader>
            <CardTitle className="text-status-degraded text-base">
              These two clips cannot be put on one clock
            </CardTitle>
            <CardDescription>
              A result, not a failure. A comparison needs all four events in
              both clips: three knots cannot map the phase they do not bound,
              and a fourth invented here would decide where every sample lands.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              {(result.warnings ?? []).map((warning) => (
                <li key={warning} className="flex gap-2">
                  <span aria-hidden="true">·</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

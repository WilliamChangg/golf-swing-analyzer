/**
 * Environment health screen.
 *
 * Renders exactly what the analysis engine measured on this machine and nothing
 * else: no component is shown as healthy unless the engine said so, and a
 * failed call renders as a failure rather than an empty (and therefore falsely
 * reassuring) list.
 */

import type {
  ComponentStatus,
  EngineError,
  EngineResult,
  EnvironmentReport,
} from "@gsa/types";
import { Loader2, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { EngineErrorPanel } from "@/components/engine-error-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { StatusBadge } from "@/components/ui/status-badge";
import { doctor } from "@/lib/ipc";

import { ModelsPanel } from "./ModelsPanel";

type LoadState =
  | { phase: "loading" }
  | { phase: "loaded"; report: EnvironmentReport }
  | { phase: "failed"; error: EngineError };

/** Map an engine result onto view state. Pure, so both the mount effect and the
 *  re-check handler can share it without duplicating the mapping. */
function toLoadState(result: EngineResult<EnvironmentReport>): LoadState {
  return result.ok
    ? { phase: "loaded", report: result.value }
    : { phase: "failed", error: result.error };
}

function ComponentRow({ component }: { component: ComponentStatus }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3 py-3">
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="font-mono text-sm font-medium">
            {component.name}
          </span>
          {component.version ? (
            <span className="text-muted-foreground font-mono text-xs">
              {component.version}
            </span>
          ) : null}
        </div>
        <p className="text-muted-foreground mt-1 text-sm break-words">
          {component.detail}
        </p>
        {component.remediation ? (
          <p className="mt-1 text-sm break-words text-amber-600 dark:text-amber-400">
            Fix: {component.remediation}
          </p>
        ) : null}
      </div>
      <StatusBadge status={component.status} />
    </div>
  );
}

function ComputePanel({ report }: { report: EnvironmentReport }) {
  const { compute, platform } = report;

  // Presented as explicit label/value pairs rather than a single "GPU: yes"
  // indicator, because torch and MediaPipe use different backends and
  // collapsing them would misrepresent where inference actually runs.
  const rows: Array<[string, string]> = [
    ["PyTorch auto preference", compute.selected_device],
    ["PyTorch", compute.torch_version ?? "not available"],
    [
      "Metal (MPS)",
      `${compute.mps_available ? "available" : "unavailable"} / ${
        compute.mps_built ? "built" : "not built"
      }`,
    ],
    ["CUDA", compute.cuda_available ? "available" : "unavailable"],
    [
      "CPU fallback",
      compute.cpu_fallback_available ? "available" : "unavailable",
    ],
    ["MediaPipe delegate", compute.mediapipe_delegate],
    [
      "FFmpeg hwaccel",
      compute.ffmpeg_hwaccels?.length
        ? compute.ffmpeg_hwaccels.join(", ")
        : "none detected",
    ],
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Compute</CardTitle>
        <CardDescription>
          {platform.system} {platform.release} &middot; {platform.machine}
          {platform.cpu_count === null ? "" : ` · ${platform.cpu_count} CPUs`}
          {` · Python ${platform.python_version}`}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
          {rows.map(([label, value]) => (
            <div key={label} className="flex justify-between gap-4 text-sm">
              <dt className="text-muted-foreground shrink-0">{label}</dt>
              <dd className="truncate font-mono" title={value}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}

export function HealthScreen() {
  const [state, setState] = useState<LoadState>({ phase: "loading" });

  // Runs once on mount. State already starts as "loading", so nothing is set
  // synchronously here -- doing so would cause a cascading render.
  useEffect(() => {
    let cancelled = false;
    void doctor().then((result) => {
      if (!cancelled) setState(toLoadState(result));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Event handler rather than an effect, so setting "loading" up front is fine.
  const runCheck = useCallback(async () => {
    setState({ phase: "loading" });
    setState(toLoadState(await doctor()));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Environment</h2>
          <p className="text-muted-foreground mt-1 text-sm">
            What the analysis engine measured on this machine.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {state.phase === "loaded" ? (
            <StatusBadge status={state.report.overall_status} />
          ) : null}
          <Button
            onClick={() => void runCheck()}
            variant="outline"
            size="sm"
            disabled={state.phase === "loading"}
          >
            {state.phase === "loading" ? (
              <Loader2 className="animate-spin" />
            ) : (
              <RefreshCw />
            )}
            Re-check
          </Button>
        </div>
      </div>

      <ModelsPanel />

      {state.phase === "loading" ? (
        <Card>
          <CardContent className="text-muted-foreground flex items-center gap-3 py-8 text-sm">
            <Loader2 className="size-4 animate-spin" />
            Probing the analysis engine. The first run loads PyTorch and
            MediaPipe, which takes a few seconds.
          </CardContent>
        </Card>
      ) : null}

      {state.phase === "failed" ? (
        <EngineErrorPanel error={state.error} onRetry={() => void runCheck()} />
      ) : null}

      {state.phase === "loaded" ? (
        <div className="space-y-6">
          {state.report.warnings?.length ? (
            <Card className="border-status-degraded/40">
              <CardHeader>
                <CardTitle className="text-status-degraded text-base">
                  Caveats
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {state.report.warnings.map((warning) => (
                    <li key={warning} className="flex gap-2">
                      <span aria-hidden="true">&middot;</span>
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ) : null}

          <ComputePanel report={state.report} />

          <Card>
            <CardHeader>
              <CardTitle>Components</CardTitle>
              <CardDescription>
                {state.report.components.length} checks, measured at{" "}
                {new Date(state.report.generated_at).toLocaleTimeString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {state.report.components.map((component, index) => (
                <div key={component.name}>
                  {index > 0 ? <Separator /> : null}
                  <ComponentRow component={component} />
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      ) : null}
    </div>
  );
}

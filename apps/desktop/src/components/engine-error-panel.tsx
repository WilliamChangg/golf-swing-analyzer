/**
 * Presentation for a failed engine call.
 *
 * Shared by every screen that talks to the engine, so that a given failure
 * always reads the same way. The `kind` distinctions are the reason this is a
 * component rather than a message string: "uv is not installed", "the worker
 * crashed", and "you picked an audio file" call for different actions from the
 * user, and collapsing them into one apology helps nobody.
 */

import type { EngineError } from "@gsa/types";
import { RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { remediationOf } from "@/lib/ipc";

/** Human-readable explanation for each way an engine call can fail. */
const ERROR_HEADLINE: Record<EngineError["kind"], string> = {
  spawn: "The analysis engine could not be started",
  transport: "Lost contact with the analysis engine",
  protocol: "The analysis engine spoke an unexpected protocol",
  method: "The analysis engine reported an error",
  timeout: "The analysis engine did not respond in time",
};

const ERROR_HINT: Record<EngineError["kind"], string> = {
  spawn:
    "Check that uv is installed and that `uv sync` has been run in the python/ directory.",
  transport: "The worker process may have crashed. Try again.",
  protocol:
    "The desktop app and the analysis engine are likely different versions. Reinstall both from the same commit.",
  method: "See the message above for the specific failure.",
  timeout: "The first run loads PyTorch and MediaPipe, which can take time.",
};

export function EngineErrorPanel({
  error,
  onRetry,
  retryLabel = "Retry",
}: {
  error: EngineError;
  onRetry?: () => void;
  retryLabel?: string;
}) {
  // The engine attaches a remedy when it knows one — it is the layer that can
  // tell a missing file from a truncated one — so prefer it over the generic
  // per-kind hint.
  const remediation = remediationOf(error);

  return (
    <Card className="border-status-error/40">
      <CardHeader>
        <CardTitle className="text-status-error">
          {ERROR_HEADLINE[error.kind]}
        </CardTitle>
        <CardDescription>
          {remediation ?? ERROR_HINT[error.kind]}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <pre className="bg-muted overflow-x-auto rounded-md p-3 font-mono text-xs whitespace-pre-wrap">
          {error.message}
          {error.code === undefined ? "" : `\n\n(code ${error.code})`}
        </pre>
        {onRetry ? (
          <Button onClick={onRetry} variant="outline" size="sm">
            <RefreshCw /> {retryLabel}
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}

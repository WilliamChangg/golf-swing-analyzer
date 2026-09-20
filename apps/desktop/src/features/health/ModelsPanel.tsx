import type { EngineError, ModelInventory, ProgressUpdate } from "@gsa/types";
import { useState } from "react";

import { EngineErrorPanel } from "@/components/engine-error-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  installModel,
  listModels,
  normalizeError,
  onProgress,
  progressFraction,
} from "@/lib/ipc";

export function ModelsPanel() {
  const [inventory, setInventory] = useState<ModelInventory | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<EngineError | null>(null);
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);

  async function run(name?: string) {
    setBusy(name ?? "verify");
    setError(null);
    setProgress(null);
    let unlisten: (() => void) | undefined;
    try {
      if (name)
        unlisten = await onProgress(setProgress, { task: "install_model" });
      const result = await (name ? installModel(name) : listModels());
      if (result.ok) setInventory(result.value);
      else setError(result.error);
    } catch (cause) {
      setError(normalizeError(cause));
    } finally {
      unlisten?.();
      setBusy(null);
    }
  }

  const fraction = progress ? progressFraction(progress) : null;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Local models</CardTitle>
        <CardDescription>
          Downloads and updates install the version pinned by this application.
          Model files are verified before replacing an installed version.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Button
          variant="outline"
          size="sm"
          disabled={busy !== null}
          onClick={() => void run()}
        >
          {inventory ? "Verify models" : "Manage models"}
        </Button>
        {busy ? (
          <p role="status" className="text-sm">
            {busy === "verify"
              ? "Verifying local files…"
              : `Installing ${busy}…`}
            {progress
              ? ` ${progress.stage}${fraction === null ? "" : ` ${Math.round(fraction * 100)}%`}`
              : ""}
          </p>
        ) : null}
        {error ? <EngineErrorPanel error={error} /> : null}
        {inventory?.models.map((model) => (
          <article
            key={model.name}
            aria-label={model.name}
            className="space-y-2 border-t pt-4"
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="font-mono text-sm font-medium">
                {model.name}
                {model.name === inventory.default_pose_model
                  ? " · default"
                  : ""}
              </h3>
              <Button
                variant="outline"
                size="sm"
                disabled={busy !== null}
                onClick={() => void run(model.name)}
              >
                {model.state === "missing"
                  ? "Download"
                  : model.state === "mismatch"
                    ? "Update to pinned version"
                    : "Reinstall"}
              </Button>
            </div>
            <p className="text-sm">
              {model.backend} · {model.device} ·{" "}
              {(model.size_bytes / 1048576).toFixed(1)} MB
            </p>
            <p className="text-sm">{model.detail}</p>
            <p className="text-muted-foreground text-xs break-all">
              Pinned SHA-256: {model.version}
            </p>
            {model.installed_sha256 &&
            model.installed_sha256 !== model.version ? (
              <p className="text-xs break-all">
                Installed SHA-256: {model.installed_sha256}
              </p>
            ) : null}
            <ul className="text-muted-foreground list-inside list-disc text-xs">
              {model.input_requirements.map((requirement) => (
                <li key={requirement}>{requirement}</li>
              ))}
            </ul>
          </article>
        ))}
      </CardContent>
    </Card>
  );
}

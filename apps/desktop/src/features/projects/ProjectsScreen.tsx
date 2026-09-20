/**
 * Sessions, which remember which clips belong together.
 *
 * A project is the first state in this system that cannot be recomputed. Every
 * other artifact — poses, filters, phases, metrics — is a function of a video
 * file and a configuration, and can be thrown away and rebuilt. Which two clips
 * are two views of the same swing is a fact only a person knows, so it is
 * stored, and it lives in SQLite under the data directory rather than in the
 * cache for exactly that reason.
 *
 * **Clips are identified by content, not by path.** A file that has been moved
 * or renamed still matches; a different recording that happens to have the same
 * name does not. That is what makes a stored alignment survive tidying up a
 * folder, and it is why adding a clip probes it rather than taking the path on
 * trust.
 *
 * The role is asked for and recorded as a **declaration**. Phase 6 measures the
 * view from the footage and may disagree — it did on `iron_dtl.mp4`, which was
 * filmed from in front of the player despite its name — and that disagreement
 * is worth being able to state, which it cannot be unless the declaration was
 * recorded separately from the measurement.
 */

import type { CameraRole, EngineError, Project, ProjectList } from "@gsa/types";
import { FolderPlus, Loader2, Plus, Trash2 } from "lucide-react";
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
import {
  addClip,
  chooseClip,
  createProject,
  deleteProject,
  listProjects,
  removeClip,
} from "@/lib/ipc";

/**
 * Camera positions offered, as an exhaustive record.
 *
 * Exhaustive so a role added to the Python contract is a type error here until
 * it is given a label, rather than appearing in the picker as a raw enum name.
 */
const ROLE_LABEL: Record<CameraRole, string> = {
  face_on: "Face-on",
  down_the_line: "Down the line",
  other: "Other",
};

const ROLES = Object.keys(ROLE_LABEL) as CameraRole[];

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

function ProjectCard({
  project,
  onChanged,
  onError,
}: {
  project: Project;
  onChanged: (project: Project) => void;
  onError: (error: EngineError) => void;
}) {
  const [role, setRole] = useState<CameraRole>("face_on");
  const [busy, setBusy] = useState(false);

  const attach = useCallback(async () => {
    setBusy(true);
    try {
      const chosen = await chooseClip();
      if (!chosen.ok) {
        onError(chosen.error);
        return;
      }
      if (chosen.value == null) return;

      const result = await addClip(project.id, chosen.value, role);
      if (result.ok) onChanged(result.value);
      else onError(result.error);
    } finally {
      setBusy(false);
    }
  }, [onChanged, onError, project.id, role]);

  const detach = useCallback(
    async (clipId: number) => {
      setBusy(true);
      try {
        const result = await removeClip(project.id, clipId);
        if (result.ok) onChanged(result.value);
        else onError(result.error);
      } finally {
        setBusy(false);
      }
    },
    [onChanged, onError, project.id],
  );

  const clips = project.clips ?? [];

  return (
    <div className="space-y-3 rounded-md border p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="font-medium">{project.name}</h4>
        <span className="text-muted-foreground font-mono text-xs">
          #{project.id}
        </span>
      </div>

      {project.notes ? (
        <p className="text-muted-foreground text-sm">{project.notes}</p>
      ) : null}

      {clips.length === 0 ? (
        <p className="text-muted-foreground text-sm">
          No clips yet. Two views of one swing is what makes calibration,
          synchronisation and three-dimensional reconstruction possible; one
          clip is still a usable project and measures everything a single camera
          supports.
        </p>
      ) : (
        <ul className="space-y-1.5 text-sm">
          {clips.map((entry) => (
            <li
              key={entry.id}
              className="flex flex-wrap items-baseline justify-between gap-2"
            >
              <span className="flex items-baseline gap-2">
                <span className="font-mono text-xs">
                  {ROLE_LABEL[entry.role]}
                </span>
                <span>{basename(entry.path)}</span>
                {entry.slow_motion_factor !== 1 ? (
                  <span className="text-muted-foreground text-xs">
                    {entry.slow_motion_factor}&times; slow
                  </span>
                ) : null}
              </span>
              <Button
                variant="ghost"
                size="sm"
                aria-label={`Remove ${basename(entry.path)}`}
                disabled={busy}
                onClick={() => void detach(entry.id)}
              >
                <Trash2 />
              </Button>
            </li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <label className="text-sm">
          <span className="sr-only">Camera position</span>
          <select
            value={role}
            aria-label="Camera position"
            className="border-input rounded-md border px-2 py-1 text-sm"
            onChange={(event) => {
              setRole(event.target.value as CameraRole);
            }}
          >
            {ROLES.map((entry) => (
              <option key={entry} value={entry}>
                {ROLE_LABEL[entry]}
              </option>
            ))}
          </select>
        </label>
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => void attach()}
        >
          {busy ? <Loader2 className="animate-spin" /> : <Plus />}
          Add clip
        </Button>
      </div>
    </div>
  );
}

export function ProjectsScreen() {
  const [projects, setProjects] = useState<ProjectList | null>(null);
  const [error, setError] = useState<EngineError | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const result = await listProjects();
    if (result.ok) {
      setProjects(result.value);
      setError(null);
    } else {
      setError(result.error);
    }
  }, []);

  // The initial load. Written out rather than calling `refresh` so the state
  // lands in the promise's callback and is dropped if the screen has gone --
  // which is both what the effect rules ask for and a real fix: `refresh`
  // resolves after an await, and a screen switched away from in the meantime
  // would otherwise be setting state on a component nobody is looking at.
  useEffect(() => {
    let live = true;
    void listProjects().then((result) => {
      if (!live) return;
      if (result.ok) {
        setProjects(result.value);
        setError(null);
      } else {
        setError(result.error);
      }
    });
    return () => {
      live = false;
    };
  }, []);

  const create = useCallback(async () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    setBusy(true);
    try {
      const result = await createProject(trimmed);
      if (result.ok) {
        setName("");
        await refresh();
      } else {
        setError(result.error);
      }
    } finally {
      setBusy(false);
    }
  }, [name, refresh]);

  const remove = useCallback(async (projectId: number) => {
    setBusy(true);
    try {
      const result = await deleteProject(projectId);
      if (result.ok) setProjects(result.value);
      else setError(result.error);
    } finally {
      setBusy(false);
    }
  }, []);

  const entries = projects?.projects ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Sessions</h2>
        <p className="text-muted-foreground mt-1 text-sm">
          Which clips belong together. The only thing here that cannot be
          recomputed from the footage.
        </p>
      </div>

      {error ? (
        <EngineErrorPanel error={error} onRetry={() => void refresh()} />
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>New session</CardTitle>
          <CardDescription>
            The name is for you; it is not unique, and the id is the identity.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="flex flex-wrap items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void create();
            }}
          >
            <label className="flex-1">
              <span className="sr-only">Session name</span>
              <input
                value={name}
                placeholder="Range session, 14 September"
                aria-label="Session name"
                className="border-input w-full rounded-md border px-3 py-1.5 text-sm"
                onChange={(event) => {
                  setName(event.target.value);
                }}
              />
            </label>
            <Button type="submit" size="sm" disabled={busy || !name.trim()}>
              <FolderPlus /> Create
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between gap-4">
            <span>Sessions</span>
            <span className="text-muted-foreground text-sm font-normal">
              {entries.length}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {entries.length === 0 ? (
            <p className="text-muted-foreground text-sm">No sessions yet.</p>
          ) : (
            entries.map((project) => (
              <div key={project.id} className="space-y-2">
                <ProjectCard
                  project={project}
                  onChanged={() => void refresh()}
                  onError={setError}
                />
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  onClick={() => void remove(project.id)}
                >
                  <Trash2 /> Delete session
                </Button>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Session management.
 *
 * The behaviour worth pinning is that a clip is attached with a **declared**
 * camera role, and that cancelling the picker is not an error — both of which
 * are easy to get wrong in a way nothing visibly breaks.
 */

import type { Project, ProjectList } from "@gsa/types";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as IpcModule from "@/lib/ipc";

const listProjectsMock = vi.hoisted(() => vi.fn());
const createProjectMock = vi.hoisted(() => vi.fn());
const deleteProjectMock = vi.hoisted(() => vi.fn());
const addClipMock = vi.hoisted(() => vi.fn());
const chooseClipMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  ...(await importOriginal<typeof IpcModule>()),
  listProjects: listProjectsMock,
  createProject: createProjectMock,
  deleteProject: deleteProjectMock,
  addClip: addClipMock,
  chooseClip: chooseClipMock,
}));

const { ProjectsScreen } = await import("./ProjectsScreen");

function project(overrides: Partial<Project> = {}): Project {
  return {
    schema_version: 1,
    id: 1,
    name: "Range session",
    notes: "",
    created_at: "2026-09-18T12:00:00Z",
    clips: [],
    syncs: [],
    rig: null,
    ...overrides,
  };
}

function list(projects: Project[]): ProjectList {
  return {
    schema_version: 1,
    projects,
    database_path: "/Users/example/Library/Application Support/gsa/projects.db",
  };
}

beforeEach(() => {
  listProjectsMock.mockReset().mockResolvedValue({ ok: true, value: list([]) });
  createProjectMock.mockReset();
  deleteProjectMock.mockReset();
  addClipMock.mockReset();
  chooseClipMock.mockReset();
});

describe("ProjectsScreen", () => {
  it("lists the sessions the engine reports", async () => {
    listProjectsMock.mockResolvedValue({
      ok: true,
      value: list([project(), project({ id: 2, name: "Lesson" })]),
    });

    render(<ProjectsScreen />);

    expect(await screen.findByText("Range session")).toBeVisible();
    expect(screen.getByText("Lesson")).toBeVisible();
  });

  it("attaches a clip with the declared camera role", async () => {
    // The declaration is recorded separately from Phase 6's measurement of the
    // view precisely so the two can disagree -- which they did on
    // `iron_dtl.mp4`, filmed from in front of the player despite its name.
    listProjectsMock.mockResolvedValue({ ok: true, value: list([project()]) });
    chooseClipMock.mockResolvedValue({ ok: true, value: "/data/dtl.mp4" });
    addClipMock.mockResolvedValue({ ok: true, value: project() });

    render(<ProjectsScreen />);
    await screen.findByText("Range session");

    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Camera position" }),
      "down_the_line",
    );
    await userEvent.click(screen.getByRole("button", { name: /Add clip/ }));

    await waitFor(() => {
      expect(addClipMock).toHaveBeenCalledWith(
        1,
        "/data/dtl.mp4",
        "down_the_line",
      );
    });
  });

  it("treats a cancelled picker as nothing happening", async () => {
    // `chooseClip` returns null on cancel, which is not an error and must not
    // be reported as one.
    listProjectsMock.mockResolvedValue({ ok: true, value: list([project()]) });
    chooseClipMock.mockResolvedValue({ ok: true, value: null });

    render(<ProjectsScreen />);
    await screen.findByText("Range session");
    await userEvent.click(screen.getByRole("button", { name: /Add clip/ }));

    await waitFor(() => {
      expect(chooseClipMock).toHaveBeenCalled();
    });
    expect(addClipMock).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows a clip's slow-motion factor when it is not one", async () => {
    // A supplied factor multiplies every duration and speed measured from that
    // clip, so it belongs where the clip is listed rather than buried.
    listProjectsMock.mockResolvedValue({
      ok: true,
      value: list([
        project({
          clips: [
            {
              id: 7,
              path: "/data/rory.mp4",
              name: "rory.mp4",
              content_key: {
                algorithm: "sha256-sampled-v1",
                digest: "b".repeat(64),
                size_bytes: 10,
              },
              role: "face_on",
              slow_motion_factor: 7,
              label: "",
              added_at: "2026-09-18T12:00:00Z",
              exists: true,
            },
          ],
        }),
      ]),
    });

    render(<ProjectsScreen />);

    expect(await screen.findByText(/7×\s*slow/)).toBeVisible();
  });

  it("refuses to create a session with a blank name", async () => {
    render(<ProjectsScreen />);

    expect(
      await screen.findByRole("button", { name: /Create/ }),
    ).toBeDisabled();
    expect(createProjectMock).not.toHaveBeenCalled();
  });

  it("creates a session and reloads the list", async () => {
    createProjectMock.mockResolvedValue({ ok: true, value: project() });

    render(<ProjectsScreen />);
    await userEvent.type(
      screen.getByRole("textbox", { name: "Session name" }),
      "Range session",
    );
    await userEvent.click(screen.getByRole("button", { name: /Create/ }));

    await waitFor(() => {
      expect(createProjectMock).toHaveBeenCalledWith("Range session");
    });
    expect(listProjectsMock).toHaveBeenCalledTimes(2);
  });

  it("reports an engine failure rather than showing an empty list", async () => {
    listProjectsMock.mockResolvedValue({
      ok: false,
      error: {
        kind: "spawn",
        message: "The analysis engine could not be started.",
      },
    });

    render(<ProjectsScreen />);

    // The headline, not the raw message: the two differ, and a spawn failure is
    // the one case where the app can say something more useful than the
    // engine's own wording because the engine is what failed to start.
    expect(
      await screen.findByText("The analysis engine could not be started"),
    ).toBeVisible();
    expect(screen.getByText(/uv is installed/)).toBeVisible();
  });
});

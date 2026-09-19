/**
 * The screen where the geometry and the footage are put on one number.
 *
 * What is worth pinning here is **not** that a video element and a 3D view stay
 * in step — they cannot drift, because there is only one frame index and both
 * read it. What is worth pinning is everything around that: which number it is,
 * whether the app can tell the video is the right recording, and what it does
 * when it is not.
 *
 * The "scrub stays in sync" claim itself is checked where it means something, in
 * `e2e/scene.spec.ts`, against a real decoder reporting the frame it painted.
 */

import type { Project, ProjectList, ReconstructionScene } from "@gsa/types";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as IpcModule from "@/lib/ipc";
import {
  EMPTY_FRAME,
  FIRST_FRAME,
  FRAME_COUNT,
  PARTIAL_FRAME,
  matchingSeekIndex,
  sceneFixture,
} from "@/features/scene/scene.fixture";

const listProjectsMock = vi.hoisted(() => vi.fn());
const reconstructSceneMock = vi.hoisted(() => vi.fn());
const chooseClipMock = vi.hoisted(() => vi.fn());
const probeVideoMock = vi.hoisted(() => vi.fn());
const seekIndexMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/ipc", async (importOriginal) => ({
  ...(await importOriginal<typeof IpcModule>()),
  listProjects: listProjectsMock,
  reconstructScene: reconstructSceneMock,
  chooseClip: chooseClipMock,
  probeVideo: probeVideoMock,
  seekIndex: seekIndexMock,
  onProgress: () => Promise.resolve(() => undefined),
  clipSource: (path: string) => `asset://${path}`,
}));

const { SceneScreen } = await import("./SceneScreen");

const PROJECT: Project = {
  schema_version: 1,
  id: 7,
  name: "Range session",
  notes: "",
  created_at: "2026-09-18T12:00:00Z",
  clips: [],
  syncs: [],
  rig: null,
};

const PROJECTS: ProjectList = {
  schema_version: 1,
  projects: [PROJECT],
  database_path: "/data/projects.db",
};

const METADATA = {
  schema_version: 1,
  path: "/data/face_on.mp4",
  content_key: matchingSeekIndex().content_key,
  stream: { display_width: 1920, display_height: 1080 },
} as unknown as Awaited<ReturnType<typeof IpcModule.probeVideo>> extends {
  value: infer T;
}
  ? T
  : never;

beforeEach(() => {
  listProjectsMock.mockReset().mockResolvedValue({ ok: true, value: PROJECTS });
  reconstructSceneMock
    .mockReset()
    .mockResolvedValue({ ok: true, value: sceneFixture() });
  chooseClipMock
    .mockReset()
    .mockResolvedValue({ ok: true, value: "/data/face_on.mp4" });
  probeVideoMock.mockReset().mockResolvedValue({ ok: true, value: METADATA });
  seekIndexMock
    .mockReset()
    .mockResolvedValue({ ok: true, value: matchingSeekIndex() });
});

function scrubTo(frameIndex: number): void {
  fireEvent.change(screen.getByLabelText("Frame"), {
    target: { value: String(frameIndex) },
  });
}

async function reconstruct(): Promise<void> {
  render(<SceneScreen />);
  await screen.findByRole("option", { name: "Range session" });
  await userEvent.click(screen.getByRole("button", { name: /Reconstruct/ }));
  await screen.findByTestId("scene-viewport");
}

describe("before anything is reconstructed", () => {
  it("offers the sessions and draws no viewport", async () => {
    render(<SceneScreen />);
    await screen.findByRole("option", { name: "Range session" });
    expect(screen.queryByTestId("scene-viewport")).toBeNull();
  });

  it("explains what a reconstruction needs when there are no sessions", async () => {
    listProjectsMock.mockResolvedValue({
      ok: true,
      value: { ...PROJECTS, projects: [] },
    });
    render(<SceneScreen />);
    // A reconstruction needs a pair, a calibration and an alignment. None of the
    // three is recoverable from a loose file, and saying so is more use than an
    // empty viewport.
    expect(
      await screen.findByText(/two clips of\s+one swing/),
    ).toBeInTheDocument();
  });

  it("surfaces the engine's refusal as a result", async () => {
    reconstructSceneMock.mockResolvedValue({
      ok: false,
      error: {
        kind: "engine",
        message:
          "This project's cameras have not been calibrated as a stereo pair.",
      },
    });
    render(<SceneScreen />);
    await screen.findByRole("option", { name: "Range session" });
    await userEvent.click(screen.getByRole("button", { name: /Reconstruct/ }));
    expect(
      await screen.findByText(/have not been calibrated as a stereo pair/),
    ).toBeInTheDocument();
  });
});

describe("the frame the viewport draws", () => {
  it("starts at the scene's own first frame, not at zero", async () => {
    // The reconstruction spans the frames where the two clips overlap, which is
    // rarely the whole clip. A viewport that opened at frame zero would open
    // outside its own data.
    await reconstruct();
    expect(screen.getByTestId("scene-viewport")).toHaveAttribute(
      "data-frame",
      String(FIRST_FRAME),
    );
    expect(screen.getByTestId("scene-frame")).toHaveTextContent(
      String(FIRST_FRAME),
    );
  });

  it("follows the scrubber with no clip loaded", async () => {
    // The viewport is usable on its own. A reconstruction is worth looking at
    // before the reference clip has been picked, and needing the video first
    // would make the 3D view conditional on a permission grant it does not use.
    await reconstruct();
    const slider = screen.getByLabelText("Frame");
    expect(slider).toHaveAttribute("min", String(FIRST_FRAME));
    expect(slider).toHaveAttribute(
      "max",
      String(FIRST_FRAME + FRAME_COUNT - 1),
    );

    scrubTo(FIRST_FRAME + 1);
    await waitFor(() => {
      expect(screen.getByTestId("scene-viewport")).toHaveAttribute(
        "data-frame",
        String(FIRST_FRAME + 1),
      );
    });
  });

  it("draws nothing on a frame the reconstruction refused entirely", async () => {
    await reconstruct();
    scrubTo(EMPTY_FRAME);
    await waitFor(() => {
      expect(screen.getByTestId("scene-viewport")).toHaveAttribute(
        "data-reconstructed",
        "0",
      );
    });
    expect(screen.getByTestId("scene-empty")).toBeInTheDocument();
  });
});

describe("pairing the scene with a clip", () => {
  it("accepts the clip the reconstruction was built from", async () => {
    await reconstruct();
    await userEvent.click(
      screen.getByRole("button", { name: /Choose the clip/ }),
    );
    expect(await screen.findByTestId("scene-clip-matched")).toBeInTheDocument();
    expect(screen.queryByTestId("scene-clip-mismatch")).toBeNull();
  });

  it("**refuses a different recording rather than scrubbing it anyway**", async () => {
    // The failure this guards against does not look like an error. The video
    // plays, the skeleton moves, the frame numbers agree — and the body on
    // screen has nothing to do with the footage behind it.
    const other = matchingSeekIndex();
    seekIndexMock.mockResolvedValue({
      ok: true,
      value: {
        ...other,
        content_key: { ...other.content_key, digest: "f".repeat(64) },
      },
    });
    await reconstruct();
    await userEvent.click(
      screen.getByRole("button", { name: /Choose the clip/ }),
    );

    const mismatch = await screen.findByTestId("scene-clip-mismatch");
    expect(mismatch).toHaveTextContent(
      /not the clip this reconstruction was built from/,
    );
    expect(screen.queryByTestId("scene-clip-matched")).toBeNull();
  });

  it("is not an error when the picker is cancelled", async () => {
    chooseClipMock.mockResolvedValue({ ok: true, value: null });
    await reconstruct();
    await userEvent.click(
      screen.getByRole("button", { name: /Choose the clip/ }),
    );
    expect(screen.queryByTestId("scene-clip-matched")).toBeNull();
    expect(screen.queryByTestId("scene-clip-mismatch")).toBeNull();
    expect(probeVideoMock).not.toHaveBeenCalled();
  });
});

describe("the viewpoint panel", () => {
  it("states the calibration, the conditioning and what the view hides", async () => {
    await reconstruct();
    const panel = screen.getByTestId("viewpoint-panel");
    expect(panel).toHaveTextContent(/calibrated as a pair/);
    expect(panel).toHaveTextContent(/88°/);
    // The fixture's ellipsoid points down the reference camera's axis, so the
    // default view hides most of it — and the panel says so rather than letting
    // the picture speak for itself.
    expect(screen.getByTestId("viewpoint-warning")).toHaveTextContent(
      /pointing along the line of sight/,
    );
  });

  it("stops warning once the reader has orbited across the error", async () => {
    await reconstruct();
    expect(screen.getByTestId("viewpoint-warning")).toBeInTheDocument();

    const viewport = screen.getByTestId("scene-viewport");
    // A drag of 225 px at 0.4 degrees per pixel is 90 degrees round.
    viewport.dispatchEvent(
      new PointerEvent("pointerdown", {
        bubbles: true,
        clientX: 0,
        clientY: 0,
      }),
    );
    viewport.dispatchEvent(
      new PointerEvent("pointermove", {
        bubbles: true,
        clientX: 225,
        clientY: 0,
      }),
    );
    viewport.dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));

    await waitFor(() => {
      expect(screen.queryByTestId("viewpoint-warning")).toBeNull();
    });
  });

  it("names the refusal reason for a landmark that produced nothing", async () => {
    // Not "one landmark missing": the fix for a shallow intersection is to move
    // a camera, and none of the other four reasons has that fix.
    await reconstruct();
    scrubTo(PARTIAL_FRAME);
    await waitFor(() => {
      expect(screen.getByTestId("scene-refusals")).toHaveTextContent(
        /rays too shallow to intersect/,
      );
    });
  });
});

describe("scene warnings", () => {
  it("shows what the engine warned about", async () => {
    const warned: ReconstructionScene = sceneFixture({
      warnings: [
        "This pair has no stored alignment, so one was fitted for this scene.",
      ],
    });
    reconstructSceneMock.mockResolvedValue({ ok: true, value: warned });
    await reconstruct();
    expect(screen.getByText(/no stored alignment/)).toBeInTheDocument();
  });
});

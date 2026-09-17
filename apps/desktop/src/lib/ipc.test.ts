import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.hoisted(() => vi.fn());
const listenMock = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));
vi.mock("@tauri-apps/api/event", () => ({ listen: listenMock }));

const {
  doctor,
  extractPoses,
  normalizeError,
  onProgress,
  probeVideo,
  progressFraction,
  remediationOf,
} = await import("./ipc");

/** A minimal, valid progress payload; tests override what they care about. */
function update(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    request_id: 1,
    task: "extract_poses",
    stage: "estimating",
    current: 25,
    total: 100,
    elapsed_s: 1,
    detail: null,
    ...overrides,
  };
}

describe("normalizeError", () => {
  it("treats a plain string rejection as a transport failure", () => {
    expect(normalizeError("boom")).toEqual({
      kind: "transport",
      message: "boom",
    });
  });

  it("unwraps a JS Error", () => {
    expect(normalizeError(new Error("exploded"))).toEqual({
      kind: "transport",
      message: "exploded",
    });
  });

  it("preserves a structured engine error", () => {
    expect(
      normalizeError({
        kind: "method",
        message: "Unknown method 'nope'",
        code: -32601,
      }),
    ).toEqual({
      kind: "method",
      message: "Unknown method 'nope'",
      code: -32601,
    });
  });

  it("downgrades an unrecognised kind rather than trusting it", () => {
    expect(normalizeError({ kind: "bogus", message: "x" }).kind).toBe(
      "transport",
    );
  });

  it("never produces an empty message", () => {
    for (const input of [null, undefined, 42, {}, []]) {
      expect(normalizeError(input).message.length).toBeGreaterThan(0);
    }
  });
});

describe("doctor", () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  it("returns ok with the report on success", async () => {
    const report = { schema_version: 1, components: [] };
    invokeMock.mockResolvedValue(report);

    const result = await doctor();

    expect(invokeMock).toHaveBeenCalledWith("doctor");
    expect(result).toEqual({ ok: true, value: report });
  });

  it("returns a normalised error instead of throwing", async () => {
    invokeMock.mockRejectedValue({ kind: "spawn", message: "no python" });

    const result = await doctor();

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.kind).toBe("spawn");
      expect(result.error.message).toBe("no python");
    }
  });

  it("does not leak a rejection when the engine dies mid-call", async () => {
    invokeMock.mockRejectedValue(new Error("worker exited"));
    await expect(doctor()).resolves.toMatchObject({ ok: false });
  });
});

describe("probeVideo", () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  it("passes the path and an explicit refresh flag", async () => {
    invokeMock.mockResolvedValue({ path: "/tmp/a.mov" });

    await probeVideo("/tmp/a.mov");

    expect(invokeMock).toHaveBeenCalledWith("probe_video", {
      path: "/tmp/a.mov",
      refresh: false,
    });
  });

  it("forwards refresh when asked to re-read", async () => {
    invokeMock.mockResolvedValue({});

    await probeVideo("/tmp/a.mov", { refresh: true });

    expect(invokeMock).toHaveBeenCalledWith("probe_video", {
      path: "/tmp/a.mov",
      refresh: true,
    });
  });

  it("returns an unusable file as a handled error, not a rejection", async () => {
    invokeMock.mockRejectedValue({
      kind: "method",
      message: "clip.m4a contains no video stream.",
      code: -31001,
      data: { remediation: "Select a video recording." },
    });

    const result = await probeVideo("/tmp/clip.m4a");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error.code).toBe(-31001);
      expect(remediationOf(result.error)).toBe("Select a video recording.");
    }
  });
});

describe("remediationOf", () => {
  it("returns null when the engine attached no remedy", () => {
    expect(remediationOf({ kind: "transport", message: "gone" })).toBeNull();
  });

  it("returns null rather than a non-string, so it cannot be rendered raw", () => {
    expect(
      remediationOf({
        kind: "method",
        message: "x",
        data: { remediation: 42 },
      }),
    ).toBeNull();
  });

  it("ignores an empty remedy", () => {
    expect(
      remediationOf({
        kind: "method",
        message: "x",
        data: { remediation: "" },
      }),
    ).toBeNull();
  });
});

describe("extractPoses", () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  it("passes the path and an explicit null model", async () => {
    /* Null rather than omitted: the engine forbids unexpected parameters, so
       what goes on the wire is worth keeping exact. */
    invokeMock.mockResolvedValue({});

    await extractPoses("/tmp/a.mov");

    expect(invokeMock).toHaveBeenCalledWith("extract_poses", {
      path: "/tmp/a.mov",
      model: null,
    });
  });

  it("forwards a chosen model", async () => {
    invokeMock.mockResolvedValue({});

    await extractPoses("/tmp/a.mov", { model: "pose_landmarker_lite" });

    expect(invokeMock).toHaveBeenCalledWith("extract_poses", {
      path: "/tmp/a.mov",
      model: "pose_landmarker_lite",
    });
  });

  it("returns a missing model as a handled error with its remedy", async () => {
    invokeMock.mockRejectedValue({
      kind: "method",
      message: "Model 'pose_landmarker_heavy' is not downloaded.",
      code: -31001,
      data: { remediation: "Run `python scripts/download_models.py`." },
    });

    const result = await extractPoses("/tmp/a.mov");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(remediationOf(result.error)).toBe(
        "Run `python scripts/download_models.py`.",
      );
    }
  });
});

describe("onProgress", () => {
  beforeEach(() => {
    listenMock.mockReset();
    listenMock.mockResolvedValue(() => undefined);
  });

  /** The callback Tauri's `listen` was registered with, so a test can fire it. */
  function emitter(): (event: unknown) => void {
    const call = listenMock.mock.calls[0];
    if (!call) throw new Error("nothing subscribed to the event channel");
    return call[1] as (event: unknown) => void;
  }

  it("listens on the engine progress channel", async () => {
    await onProgress(() => undefined);
    expect(listenMock).toHaveBeenCalledWith(
      "engine://progress",
      expect.any(Function),
    );
  });

  it("delivers updates to the handler", async () => {
    const seen: number[] = [];
    await onProgress((u) => seen.push(u.current));

    const emit = emitter();
    emit({ payload: update({ current: 7 }) });

    expect(seen).toEqual([7]);
  });

  it("filters out other tasks sharing the channel", async () => {
    const seen: string[] = [];
    await onProgress((u) => seen.push(u.task), { task: "extract_poses" });

    const emit = emitter();
    emit({ payload: update({ task: "something_else" }) });
    emit({ payload: update({ task: "extract_poses" }) });

    expect(seen).toEqual(["extract_poses"]);
  });

  it("passes everything through when no task is given", async () => {
    const seen: string[] = [];
    await onProgress((u) => seen.push(u.task));

    const emit = emitter();
    emit({ payload: update({ task: "anything" }) });

    expect(seen).toEqual(["anything"]);
  });

  it("returns the unsubscribe function", async () => {
    const unlisten = vi.fn();
    listenMock.mockResolvedValue(unlisten);

    (await onProgress(() => undefined))();

    expect(unlisten).toHaveBeenCalled();
  });
});

describe("progressFraction", () => {
  it("is the completed proportion", () => {
    expect(progressFraction(update())).toBeCloseTo(0.25);
  });

  it("is null without a total", () => {
    expect(progressFraction(update({ total: null }))).toBeNull();
  });

  it("is null for a zero total", () => {
    expect(progressFraction(update({ total: 0 }))).toBeNull();
  });

  it("is clamped to one", () => {
    expect(progressFraction(update({ current: 500 }))).toBe(1);
  });
});

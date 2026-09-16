import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

const { doctor, normalizeError, probeVideo, remediationOf } =
  await import("./ipc");

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

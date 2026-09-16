import { beforeEach, describe, expect, it, vi } from "vitest";

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

const { doctor, normalizeError } = await import("./ipc");

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

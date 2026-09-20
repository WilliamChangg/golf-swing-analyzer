import type * as Ipc from "@/lib/ipc";
import type { ModelInventory, ProgressUpdate } from "@gsa/types";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  install: vi.fn(),
  listen: vi.fn(),
  unlisten: vi.fn(),
}));
vi.mock("@/lib/ipc", async (original) => ({
  ...(await original<typeof Ipc>()),
  listModels: mocks.list,
  installModel: mocks.install,
  onProgress: mocks.listen,
}));
const { ModelsPanel } = await import("./ModelsPanel");
const inventory: ModelInventory = {
  default_pose_model: "full",
  models: [
    {
      name: "full",
      version: "a".repeat(64),
      backend: "mediapipe_tasks",
      device: "cpu",
      input_requirements: ["RGB video"],
      size_bytes: 100,
      required: true,
      state: "missing",
      detail: "Not installed.",
    },
  ],
};
beforeEach(() => {
  vi.resetAllMocks();
  mocks.list.mockResolvedValue({ ok: true, value: inventory });
  mocks.listen.mockResolvedValue(mocks.unlisten);
});

it("verifies files and displays backend, device and input requirements", async () => {
  render(<ModelsPanel />);
  await userEvent.click(screen.getByRole("button", { name: "Manage models" }));
  expect(await screen.findByText(/mediapipe_tasks · cpu/)).toBeInTheDocument();
  expect(screen.getByText("RGB video")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Verify models" }));
  expect(mocks.list).toHaveBeenCalledTimes(2);
});

it("subscribes before download, disables actions, then displays verified result", async () => {
  let finish: (value: unknown) => void = () => {};
  mocks.install.mockReturnValue(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  mocks.listen.mockImplementation(
    (handler: (event: ProgressUpdate) => void) => {
      handler({
        task: "install_model",
        stage: "download",
        current: 50,
        total: 100,
        elapsed_s: 1,
      });
      return Promise.resolve(mocks.unlisten);
    },
  );
  render(<ModelsPanel />);
  await userEvent.click(screen.getByRole("button", { name: "Manage models" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Download" }),
  );
  expect(mocks.install).toHaveBeenCalledWith("full");
  expect(screen.getByRole("status")).toHaveTextContent("download 50%");
  expect(screen.getByRole("button", { name: "Download" })).toBeDisabled();
  finish({
    ok: true,
    value: {
      ...inventory,
      models: [
        { ...inventory.models[0], state: "verified", detail: "Verified." },
      ],
    },
  });
  expect(await screen.findByText("Verified.")).toBeInTheDocument();
  expect(mocks.unlisten).toHaveBeenCalledOnce();
  expect(screen.getByRole("button", { name: "Reinstall" })).toBeEnabled();
});

it("preserves inventory and offers another update after a failed download", async () => {
  mocks.list.mockResolvedValue({
    ok: true,
    value: {
      ...inventory,
      models: [{ ...inventory.models[0], state: "mismatch" }],
    },
  });
  mocks.install.mockResolvedValue({
    ok: false,
    error: {
      kind: "method",
      message: "SHA-256 mismatch; installed file preserved.",
    },
  });
  render(<ModelsPanel />);
  await userEvent.click(screen.getByRole("button", { name: "Manage models" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Update to pinned version" }),
  );
  expect(await screen.findByText(/SHA-256 mismatch/)).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Update to pinned version" }),
  ).toBeEnabled();
  await waitFor(() => expect(mocks.unlisten).toHaveBeenCalledOnce());
});

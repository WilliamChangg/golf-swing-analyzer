import { expect, test } from "@playwright/test";
import { stubEngine } from "./fixtures";

const model = {
  name: "pose_landmarker_full",
  version: "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
  backend: "mediapipe_tasks",
  device: "cpu",
  input_requirements: ["RGB video frames in timestamp order"],
  size_bytes: 9398198,
  required: true,
  state: "missing",
  detail: "Not installed.",
};
const inventory = { default_pose_model: model.name, models: [model] };

test("download and verify a model through the Environment screen", async ({
  page,
}) => {
  await stubEngine(page, {
    list_models: { result: inventory },
    install_model: {
      result: {
        ...inventory,
        models: [
          {
            ...model,
            state: "verified",
            detail: "Size and SHA-256 match the pinned version.",
          },
        ],
      },
    },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Environment" }).click();
  await page.getByRole("button", { name: "Manage models" }).click();
  await expect(page.getByText(/mediapipe_tasks · cpu/)).toBeVisible();
  await page.getByRole("button", { name: "Download", exact: true }).click();
  await expect(
    page.getByText("Size and SHA-256 match the pinned version."),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Reinstall" })).toBeEnabled();
});

test("failed update keeps the model and retry action visible", async ({
  page,
}) => {
  await stubEngine(page, {
    list_models: {
      result: {
        ...inventory,
        models: [
          { ...model, state: "mismatch", installed_sha256: "b".repeat(64) },
        ],
      },
    },
    install_model: {
      error: {
        kind: "method",
        message: "Downloaded SHA-256 differs; installed file preserved.",
      },
    },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Environment" }).click();
  await page.getByRole("button", { name: "Manage models" }).click();
  await page.getByRole("button", { name: "Update to pinned version" }).click();
  await expect(page.getByText(/installed file preserved/)).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Update to pinned version" }),
  ).toBeEnabled();
  await expect(page.getByText(/Installed SHA-256:/)).toBeVisible();
});

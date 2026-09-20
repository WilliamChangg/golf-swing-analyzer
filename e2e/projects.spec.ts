/** Browser session workflows; persistence itself is exercised by pytest/SQLite. */
import { expect, test } from "@playwright/test";

import {
  NO_PROJECTS,
  ONE_PROJECT,
  VFR_SEEK_INDEX,
  stubEngine,
} from "./fixtures";

const project = { ...ONE_PROJECT.projects[0], name: "Sunday practice" };
const attached = {
  ...project,
  clips: [
    {
      id: 7,
      path: VFR_SEEK_INDEX.path,
      role: "down_the_line",
      slow_motion_factor: 1,
    },
  ],
};

test("creates a session, attaches a declared camera, removes it and deletes the session", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await stubEngine(page, {
    list_projects: {
      sequence: [
        // React StrictMode checks the initial subscription twice.
        NO_PROJECTS,
        NO_PROJECTS,
        { projects: [project] },
        { projects: [attached] },
        { projects: [project] },
      ].map((result) => ({ result })),
    },
    create_project: { result: project },
    choose_clip: { result: VFR_SEEK_INDEX.path },
    add_clip: { result: attached },
    remove_clip: { result: project },
    delete_project: { result: NO_PROJECTS },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Sessions", exact: true }).click();
  await expect(page.getByText("No sessions yet.")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Create", exact: true }),
  ).toBeDisabled();
  await page
    .getByRole("textbox", { name: "Session name" })
    .fill("  Sunday practice  ");
  await page.getByRole("button", { name: "Create", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Sunday practice" }),
  ).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Session name" })).toHaveValue(
    "",
  );
  await page
    .getByRole("combobox", { name: "Camera position" })
    .selectOption("down_the_line");
  await page.getByRole("button", { name: "Add clip" }).click();
  await expect(
    page.getByText("vfr_30_to_15fps.mp4", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Down the line", { exact: true }).first(),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Remove vfr_30_to_15fps.mp4" })
    .click();
  await expect(page.getByText(/No clips yet/)).toBeVisible();
  await page.getByRole("button", { name: "Delete session" }).click();
  await expect(page.getByText("No sessions yet.")).toBeVisible();
  const calls = await page.evaluate(
    () => (window as unknown as Record<string, unknown>).__GSA_CALLS__,
  );
  expect(calls).toEqual(
    expect.arrayContaining([
      {
        command: "create_project",
        args: { name: "Sunday practice", notes: null },
      },
      {
        command: "add_clip",
        args: {
          projectId: 3,
          path: VFR_SEEK_INDEX.path,
          role: "down_the_line",
          slowMotionFactor: null,
          label: null,
        },
      },
      { command: "remove_clip", args: { projectId: 3, clipId: 7 } },
      { command: "delete_project", args: { projectId: 3 } },
    ]),
  );
  expect(errors).toEqual([]);
});

test("cancelled attachment leaves the session unchanged", async ({ page }) => {
  await stubEngine(page, {
    list_projects: { result: { projects: [project] } },
    choose_clip: { result: null },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Sessions", exact: true }).click();
  await page.getByRole("button", { name: "Add clip" }).click();
  await expect(page.getByRole("button", { name: "Add clip" })).toBeEnabled();
  await expect(page.getByText(/No clips yet/)).toBeVisible();
  const calls = await page.evaluate(
    () => (window as unknown as Record<string, unknown>).__GSA_CALLS__,
  );
  expect(calls).not.toEqual(
    expect.arrayContaining([expect.objectContaining({ command: "add_clip" })]),
  );
});

test("failed creation keeps the entered name and can be retried", async ({
  page,
}) => {
  await stubEngine(page, {
    list_projects: {
      sequence: [
        { result: NO_PROJECTS },
        { result: NO_PROJECTS },
        { result: { projects: [project] } },
      ],
    },
    create_project: {
      sequence: [
        {
          error: { kind: "engine", message: "Session database is unavailable" },
        },
        { result: project },
      ],
    },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Sessions", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Session name" })
    .fill("Sunday practice");
  await page.getByRole("button", { name: "Create", exact: true }).click();
  await expect(page.getByText("Session database is unavailable")).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Session name" })).toHaveValue(
    "Sunday practice",
  );
  await page.getByRole("button", { name: "Create", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Sunday practice" }),
  ).toBeVisible();
  await expect(page.getByText("Session database is unavailable")).toHaveCount(
    0,
  );
});

/**
 * The comparison screen, in a real browser, against a real comparison.
 *
 * The component tests in `DifferencesPanel.test.tsx` and `paths.test.ts` check
 * what the panel says and what the path builder emits. What they cannot check is
 * the part that only exists once a browser has laid the SVG out: that the two
 * curves and the band are actually **on** the plot, that a break in a series is a
 * break in the drawn geometry, and that a refused channel draws nothing at all.
 *
 * `comparison.fixture.json` is a real engine output, not a hand-built one. It
 * comes from `compare()` over two synthetic swings that differ in shape *and*
 * tempo — so the durations differ, the normalised curves differ, and two of the
 * four channels refuse. Generated at 25 samples rather than the default 121 to
 * keep a committed fixture readable; nothing in the drawing depends on the count.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";

import { stubEngine } from "./fixtures";

function fixture(name: string): unknown {
  return JSON.parse(
    readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8"),
  );
}

/** Two swings from one camera: the curves are comparable and some of them differ. */
const COMPARISON = fixture("./comparison.fixture.json");

/**
 * The same swing from two cameras thirty degrees apart.
 *
 * Also a real engine output, and the reason the previous test cannot be written
 * as a mixed fixture: the camera gate is a fact about the **pair**, so it refuses
 * every channel or none. A reader offered three refused curves and one drawn one
 * would read the drawn one as having passed a test the others failed, when in
 * truth it was never asked a different question.
 */
const REFUSED = fixture("./comparison-refused.fixture.json");

/** The action button, not the navigation tab of the same name. */
function compareButton(page: Page) {
  return page.getByRole("main").getByRole("button", { name: /Compare/ });
}

/** Choose both clips through the stubbed picker, then compare. */
async function runComparison(page: Page): Promise<void> {
  await page.getByTestId("choose-reference").click();
  await page.getByTestId("choose-target").click();
  await compareButton(page).click();
  await expect(page.getByText("The clocks")).toBeVisible();
}

async function openCompare(page: Page, result: unknown): Promise<void> {
  await stubEngine(page, {
    choose_clip: { result: "/clips/before.mov" },
    compare_swings: { result },
  });
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Screens" })
    .getByRole("button", { name: "Compare" })
    .click();
}

test.beforeEach(async ({ page }) => {
  await openCompare(page, COMPARISON);
});

test("asks for two clips before it will compare anything", async ({ page }) => {
  await expect(compareButton(page)).toBeDisabled();
  await expect(page.getByText(/Choose two clips/)).toBeVisible();
});

test("draws both curves and the bracket band on a comparable channel", async ({
  page,
}) => {
  await runComparison(page);

  const plot = page.getByTestId("plot-hand_speed");
  await expect(plot).toBeVisible();

  // Both series are drawn, and neither is empty — an empty `d` renders as a
  // valid SVG path that occupies nothing, which a visibility check would pass.
  for (const id of ["reference-path", "target-path", "bracket-band"]) {
    const drawn = await plot.getByTestId(id).getAttribute("d");
    expect(drawn ?? "").toMatch(/^M[\d.]/);
  }

  // The band has real area: its bounding box is taller than a hairline.
  const band = await plot.getByTestId("bracket-band").boundingBox();
  expect(band?.height ?? 0).toBeGreaterThan(2);
});

test("marks the positions where the difference clears its bracket", async ({
  page,
}) => {
  await runComparison(page);

  // The fixture's two swings genuinely differ in shape, so some positions
  // resolve and some do not. Both being true is the point: a plot where every
  // position resolved would mean the bracket was doing nothing.
  const ticks = page
    .getByTestId("plot-hand_speed")
    .getByTestId("resolved-tick");
  const marked = await ticks.count();
  expect(marked).toBeGreaterThan(0);
  expect(marked).toBeLessThan(25);
});

test("draws a channel that is comparable but shows no resolved difference", async ({
  page,
}) => {
  // Not the same thing as a refusal. The shoulder line was measured in both
  // clips and the curves are drawn; what the pair cannot do is tell them apart.
  await runComparison(page);

  const plot = page.getByTestId("plot-shoulder_angle");
  await expect(plot).toBeVisible();
  await expect(plot.getByTestId("resolved-tick")).toHaveCount(0);
});

test("draws no curve at all when the camera gate refused the pair", async ({
  page,
}) => {
  await openCompare(page, REFUSED);
  await runComparison(page);

  // Every channel, not a selection: two curves on one axis is a comparison
  // whatever the caption says, so a refused pair draws none of them.
  await expect(page.locator('[data-testid^="plot-"]')).toHaveCount(0);
  await expect(page.getByText("Hand speed")).toBeVisible();
  await expect(
    page.getByText(/disagree about where the camera stood/).first(),
  ).toBeVisible();
});

test("shows both clocks, which is the timing the plots had removed", async ({
  page,
}) => {
  await runComparison(page);

  const clocks = page.getByTestId("clock-table");
  await expect(clocks.getByText("takeaway")).toHaveCount(2);

  // The two backswings really do differ — that is what the normalisation
  // divided out, and the only place on this screen it is still visible.
  const times = await clocks
    .locator("tbody tr td:nth-child(3)")
    .allInnerTexts();
  expect(new Set(times).size).toBeGreaterThan(1);
});

test("puts no ranking of the two swings on screen", async ({ page }) => {
  await runComparison(page);

  const body = (await page.locator("main").innerText()).toLowerCase();
  for (const word of ["better", "worse", "grade", "rating", "improved"]) {
    expect(body).not.toContain(word);
  }
  expect(body).toContain("there is no score");
});

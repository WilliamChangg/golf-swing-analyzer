/**
 * What the measurements support, and the much longer list of what they do not.
 *
 * This panel is laid out against its own contents. Across four reference clips
 * the engine produces at most two findings and refuses ten or eleven, and on the
 * 30 fps amateur clip it produces **none** — so a design that put findings first
 * and refusals in a footnote would be a mostly-empty screen with the
 * information hidden underneath it. The refusals are the panel.
 *
 * Every finding carries its evidence, and every piece of evidence is a jump
 * target into the player. That is the argument Phase 13 made for citing frames
 * at all, and it only pays off here: **a finding that carries its frames is a
 * finding somebody can discover is wrong.** A sentence of advice is not. So the
 * frames are clickable rather than printed, because the video is on the same
 * screen and following them is one click.
 *
 * `observation` is always shown and `phrased` never replaces it. A language
 * model that reworded a finding could describe the first finding with the
 * second's numbers and pass every check the guard makes, so the engine's own
 * sentence stays on screen beside any rewriting of it.
 */

import type {
  CoachingReport,
  Evidence,
  Finding,
  RefusedFinding,
} from "@gsa/types";
import { CircleSlash, Quote } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * Why a rule did not fire, in the reader's terms.
 *
 * Exhaustive, so a refusal added to the Python contract is a TypeScript error
 * here until it is given words — the alternative being a raw enum name on
 * screen, which is how a system that refuses eleven of twelve rules ends up
 * looking broken rather than careful.
 */
const REFUSAL_NOTE: Record<RefusedFinding["refusal"], string> = {
  no_swing:
    "no swing was detected in this clip, so there is nothing to compare",
  no_metric: "this clip did not produce the measurement the rule compares",
  no_threshold:
    "nobody has published a number for this, or published how it was measured",
  basis_not_permitted:
    "the published figure was measured on a different kind of quantity than this clip can produce",
  view_mismatch: "the camera position does not contain this measurement",
  supplied_timebase:
    "the clip's slow-motion factor was supplied rather than measured, so a duration in seconds is a guess multiplied by a number",
  low_confidence:
    "the measurement is there but too weakly determined to compare",
  no_uncertainty:
    "how well this quantity is measured has never been quantified, so a comparison cannot clear a bracket",
  unresolved:
    "the comparison cannot clear the bracket this clip's frame rate imposes",
};

function EvidenceRow({
  evidence,
  onSeek,
}: {
  evidence: Evidence;
  onSeek: (frame: number) => void;
}) {
  const frames = evidence.frames;
  return (
    <li className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-xs">
      <span className="font-medium">{evidence.label}</span>
      <span className="font-mono">
        {evidence.value.toFixed(2)}
        {evidence.uncertainty == null
          ? ""
          : ` ± ${evidence.uncertainty.toFixed(1)}`}
      </span>
      <span className="text-muted-foreground">{evidence.basis}</span>
      {frames.length > 0 ? (
        <button
          type="button"
          className="text-primary font-mono hover:underline"
          onClick={() => {
            onSeek(frames[0] ?? 0);
          }}
        >
          {frames.length === 1
            ? `frame ${String(frames[0])}`
            : `frames ${String(frames[0])}–${String(frames[frames.length - 1])}`}
        </button>
      ) : (
        <span className="text-muted-foreground font-mono">no frames cited</span>
      )}
    </li>
  );
}

function FindingCard({
  finding,
  onSeek,
}: {
  finding: Finding;
  onSeek: (frame: number) => void;
}) {
  const band =
    finding.band_low == null && finding.band_high == null
      ? null
      : `${finding.band_low?.toFixed(2) ?? "—"} to ${finding.band_high?.toFixed(2) ?? "—"}`;

  return (
    <li className="space-y-3 rounded-md border p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="font-medium">{finding.title}</h4>
        <span className="text-muted-foreground font-mono text-xs">
          {finding.rule_id}
        </span>
      </div>

      <p className="text-sm">{finding.observation}</p>

      {finding.phrased ? (
        // Beside the observation, never instead of it. The guard can tell that a
        // rewording invents no numbers; it cannot tell that the rewording is
        // about this finding rather than the one below it.
        <p className="text-muted-foreground flex gap-2 text-sm italic">
          <Quote className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
          {finding.phrased}
        </p>
      ) : null}

      <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
        <div className="flex justify-between gap-2">
          <dt className="text-muted-foreground">Measured</dt>
          <dd className="font-mono">{finding.value.toFixed(2)}</dd>
        </div>
        {band ? (
          <div className="flex justify-between gap-2">
            <dt className="text-muted-foreground">Published band</dt>
            <dd className="font-mono">{band}</dd>
          </div>
        ) : null}
        <div className="flex justify-between gap-2">
          {/* The bracket is what the comparison had to clear before it could be
              reported as a comparison at all. On the 30 fps clip one frame at
              the top is worth 0.74 of a band 1.37 wide, which is why that clip
              produces no findings — so this number is the one that explains the
              panel's emptiness elsewhere. */}
          <dt className="text-muted-foreground">Bracket it cleared</dt>
          <dd className="font-mono">{finding.bracket.toFixed(2)}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-muted-foreground">Margin</dt>
          <dd className="font-mono">{finding.margin.toFixed(2)}</dd>
        </div>
      </dl>

      <div className="space-y-1">
        <h5 className="text-xs font-medium">Evidence</h5>
        <ul className="space-y-1">
          {finding.evidence.map((entry) => (
            <EvidenceRow
              key={`${entry.metric}-${entry.event ?? entry.phase ?? "clip"}`}
              evidence={entry}
              onSeek={onSeek}
            />
          ))}
        </ul>
      </div>

      <p className="text-muted-foreground text-xs">
        Compared against {finding.source.citation}
        {finding.source.population ? ` — ${finding.source.population}` : ""}
        {finding.source.sample_size == null
          ? ""
          : `, n = ${String(finding.source.sample_size)}`}
        . Measured by {finding.source.method.replace(/_/g, " ")}.
      </p>
    </li>
  );
}

export function FindingsPanel({
  report,
  onSeek,
}: {
  report: CoachingReport;
  onSeek: (frame: number) => void;
}) {
  // The list fields default to empty on the Python side, so they arrive as
  // optional. Defaulted once here rather than guarded at each use, which is
  // where one of them eventually gets forgotten.
  const findings = report.findings ?? [];
  const refused = report.refused ?? [];
  const warnings = report.warnings ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>Findings</span>
          <span className="text-muted-foreground text-sm font-normal">
            {findings.length} of {report.rules_considered} rules
          </span>
        </CardTitle>
        <CardDescription>
          There is no score, no severity and no grade. Each finding carries the
          comparison it made, the bracket it had to clear and the frames it was
          measured from.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {warnings.length > 0 ? (
          <ul className="text-status-degraded space-y-2 text-sm">
            {warnings.map((warning) => (
              <li key={warning} className="flex gap-2">
                <span aria-hidden="true">&middot;</span>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        ) : null}

        {findings.length === 0 ? (
          // An empty result is the ordinary outcome, and saying so is the
          // difference between a system that found nothing and one that is
          // broken. The reader is pointed at the refusals, where the reasons are.
          <p className="text-muted-foreground flex items-start gap-3 text-sm">
            <CircleSlash
              className="mt-0.5 size-4 shrink-0 opacity-50"
              aria-hidden="true"
            />
            <span>
              No rule reached a conclusion this clip&rsquo;s measurements
              support. That is a result, not a failure — every rule that was
              considered is listed below with the reason it stopped.
            </span>
          </p>
        ) : (
          <ul className="space-y-4">
            {findings.map((finding) => (
              <FindingCard
                key={finding.rule_id}
                finding={finding}
                onSeek={onSeek}
              />
            ))}
          </ul>
        )}

        {refused.length > 0 ? (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">
              Not concluded ({refused.length})
            </h4>
            <ul className="space-y-2 text-xs">
              {refused.map((entry) => (
                <li key={entry.rule_id} className="space-y-0.5">
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <span className="font-medium">{entry.title}</span>
                    <span className="text-muted-foreground font-mono">
                      {entry.refusal}
                    </span>
                  </div>
                  <p className="text-muted-foreground">
                    {REFUSAL_NOTE[entry.refusal]}. {entry.reason}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

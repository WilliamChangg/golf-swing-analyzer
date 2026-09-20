/**
 * Play, pause, rate, frame step, and the jump targets.
 *
 * The event buttons are the reason this is a component rather than four
 * controls: an analysis whose interesting instants are "top" and "impact" is
 * much easier to check if those are one click away than if they are a frame
 * number to scrub for. Which is the same argument Phase 4 made for building an
 * inspector at all.
 *
 * The frame readout shows **what is on screen**, not what was asked for, and
 * shows the difference when there is one. A player that displayed the requested
 * number would look correct on every clip including the ones it is wrong on.
 */

import type { SwingPhases } from "@gsa/types";
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  PLAYBACK_RATES,
  type FramePlayer,
} from "@/features/player/useFramePlayer";

const EVENT_LABEL: Record<string, string> = {
  takeaway: "Takeaway",
  top: "Top",
  impact: "Impact",
  finish: "Finish",
};

export function Transport({
  player,
  phases,
  lastFrame,
}: {
  player: FramePlayer;
  phases: SwingPhases | null;
  lastFrame: number;
}) {
  const shown = player.residual?.landed ?? player.frame;
  const drifted = player.residual !== null && player.residual.delta !== 0;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="icon"
          aria-label="Back to start"
          onClick={() => {
            player.seekTo(0);
          }}
        >
          <SkipBack />
        </Button>
        <Button
          size="icon"
          aria-label={player.playing ? "Pause" : "Play"}
          onClick={player.togglePlay}
        >
          {player.playing ? <Pause /> : <Play />}
        </Button>
        <Button
          variant="outline"
          size="icon"
          aria-label="Previous frame"
          onClick={() => {
            player.step(-1);
          }}
        >
          <ChevronLeft />
        </Button>
        <Button
          variant="outline"
          size="icon"
          aria-label="Next frame"
          onClick={() => {
            player.step(1);
          }}
        >
          <ChevronRight />
        </Button>

        <div
          className="flex items-center gap-1"
          role="group"
          aria-label="Playback rate"
        >
          {PLAYBACK_RATES.map((rate) => (
            <Button
              key={rate}
              size="sm"
              variant={player.rate === rate ? "default" : "outline"}
              aria-pressed={player.rate === rate}
              onClick={() => {
                player.setRate(rate);
              }}
            >
              {rate}&times;
            </Button>
          ))}
        </div>
      </div>

      <input
        type="range"
        min={0}
        max={lastFrame}
        value={shown}
        aria-label="Frame"
        className="w-full"
        onChange={(event) => {
          player.seekTo(Number(event.target.value));
        }}
      />

      <div
        role="status"
        aria-live="polite"
        className="bg-muted/40 flex flex-wrap items-baseline gap-x-6 gap-y-1 rounded-md p-3 text-sm"
      >
        <span className="font-mono">
          frame <span className="font-semibold">{shown}</span> of {lastFrame}
        </span>
        {!player.observable ? (
          // Said rather than hidden. Without `requestVideoFrameCallback` the
          // seeks still happen and nothing can confirm them, and a residual of
          // zero here would be a number the app never measured.
          <span className="text-status-degraded">
            this browser cannot report which frame it painted — seeks are
            unverified
          </span>
        ) : drifted ? (
          <span className="text-status-degraded font-mono">
            asked for {player.residual?.requested} — landed{" "}
            {(player.residual?.delta ?? 0) > 0 ? "+" : ""}
            {player.residual?.delta}
          </span>
        ) : (
          <span className="text-status-ok">
            seek landed on the frame asked for
          </span>
        )}
      </div>

      {phases?.detected ? (
        <div className="flex flex-wrap gap-2">
          {(phases.events ?? []).map((entry) => (
            <Button
              key={entry.event}
              variant="outline"
              size="sm"
              onClick={() => {
                player.seekTo(entry.frame_index);
              }}
            >
              {EVENT_LABEL[entry.event] ?? entry.event}
              <span className="text-muted-foreground font-mono text-xs">
                {entry.frame_index}
              </span>
            </Button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

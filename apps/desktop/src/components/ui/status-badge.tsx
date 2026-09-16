import { AlertTriangle, CircleAlert, CircleCheck, CircleX } from "lucide-react";
import type { ComponentProps } from "react";

import { cn } from "@/lib/utils";

/**
 * Status values mirrored from the engine's `HealthStatus` contract.
 *
 * Kept as a local union rather than importing the generated type so this
 * presentational component has no dependency on the IPC layer; the health
 * screen is responsible for mapping engine values onto it. The exhaustive
 * record below means adding a status to the contract fails the type check here
 * until it is given a presentation.
 */
export type Status = "ok" | "degraded" | "missing" | "error";

interface StatusPresentation {
  label: string;
  icon: typeof CircleCheck;
  className: string;
}

const PRESENTATION: Record<Status, StatusPresentation> = {
  ok: {
    label: "OK",
    icon: CircleCheck,
    className: "text-status-ok border-status-ok/40 bg-status-ok/10",
  },
  degraded: {
    label: "Degraded",
    icon: AlertTriangle,
    className:
      "text-status-degraded border-status-degraded/40 bg-status-degraded/10",
  },
  missing: {
    label: "Missing",
    icon: CircleAlert,
    className:
      "text-status-missing border-status-missing/40 bg-status-missing/10",
  },
  error: {
    label: "Error",
    icon: CircleX,
    className: "text-status-error border-status-error/40 bg-status-error/10",
  },
};

export interface StatusBadgeProps extends ComponentProps<"span"> {
  status: Status;
}

export function StatusBadge({ status, className, ...props }: StatusBadgeProps) {
  const { label, icon: Icon, className: statusClass } = PRESENTATION[status];

  return (
    <span
      data-slot="status-badge"
      data-status={status}
      className={cn(
        "inline-flex w-fit shrink-0 items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium",
        statusClass,
        className,
      )}
      {...props}
    >
      {/* The icon carries the same meaning as the text, so it is hidden from
          assistive tech rather than announced twice. */}
      <Icon className="size-3.5" aria-hidden="true" />
      {label}
    </span>
  );
}

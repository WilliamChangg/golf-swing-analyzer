import { HealthScreen } from "@/features/health/HealthScreen";

/**
 * Application root.
 *
 * Phase 0 has a single screen. Routing is deliberately absent rather than
 * scaffolded with placeholder routes: an empty route that renders nothing
 * would imply features that do not exist yet.
 */
export function App() {
  return <HealthScreen />;
}

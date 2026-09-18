import {
  Activity,
  FileVideo,
  FlagTriangleRight,
  FolderKanban,
} from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { SwingScreen } from "@/features/analysis/SwingScreen";
import { HealthScreen } from "@/features/health/HealthScreen";
import { ProjectsScreen } from "@/features/projects/ProjectsScreen";
import { VideoScreen } from "@/features/video/VideoScreen";

/**
 * Application root.
 *
 * Four screens, still switched by local state rather than by a router. Phase 14
 * was where a router was expected to earn its place, and it did not: the thing
 * that would justify one is a URL worth addressing — a link to a project, a
 * swing, a frame — and nothing here is addressed from outside the window. There
 * is no second window, no deep link and no browser history to be wrong about.
 * A router would be indirection with nothing on the other side of it, which is
 * the same conclusion Phase 0 reached for a different reason.
 *
 * The Swing screen is the workflow; Video remains the panel-by-panel view of
 * one clip, which is where an individual stage is checked in isolation.
 */

const SCREENS = [
  { id: "swing", label: "Swing", icon: FlagTriangleRight },
  { id: "projects", label: "Sessions", icon: FolderKanban },
  { id: "video", label: "Video", icon: FileVideo },
  { id: "health", label: "Environment", icon: Activity },
] as const;

type ScreenId = (typeof SCREENS)[number]["id"];

export function App() {
  const [screen, setScreen] = useState<ScreenId>("swing");

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">
          Golf Swing Analyzer
        </h1>
        <nav className="mt-4 flex gap-2" aria-label="Screens">
          {SCREENS.map(({ id, label, icon: Icon }) => (
            <Button
              key={id}
              onClick={() => {
                setScreen(id);
              }}
              variant={screen === id ? "default" : "outline"}
              size="sm"
              aria-current={screen === id ? "page" : undefined}
            >
              <Icon /> {label}
            </Button>
          ))}
        </nav>
      </header>

      <main>
        {screen === "swing" ? <SwingScreen /> : null}
        {screen === "projects" ? <ProjectsScreen /> : null}
        {screen === "video" ? <VideoScreen /> : null}
        {screen === "health" ? <HealthScreen /> : null}
      </main>
    </div>
  );
}

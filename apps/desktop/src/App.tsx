import { Activity, FileVideo } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { HealthScreen } from "@/features/health/HealthScreen";
import { VideoScreen } from "@/features/video/VideoScreen";

/**
 * Application root.
 *
 * Two screens, switched by local state rather than a router. A router earns its
 * place when there are URLs worth addressing — deep links into a project, a
 * swing, a frame — which arrives with project management in Phase 14. Until
 * then it would be indirection with nothing on the other side of it.
 */

const SCREENS = [
  { id: "health", label: "Environment", icon: Activity },
  { id: "video", label: "Video", icon: FileVideo },
] as const;

type ScreenId = (typeof SCREENS)[number]["id"];

export function App() {
  const [screen, setScreen] = useState<ScreenId>("health");

  return (
    <div className="mx-auto w-full max-w-4xl px-6 py-10">
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
        {screen === "health" ? <HealthScreen /> : null}
        {screen === "video" ? <VideoScreen /> : null}
      </main>
    </div>
  );
}

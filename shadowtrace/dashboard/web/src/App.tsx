import { useState } from "react";
import Overview from "./pages/Overview";
import Archetypes from "./pages/Archetypes";
import Recommendations from "./pages/Recommendations";
import System from "./pages/System";
import Health from "./pages/Health";

const TABS = ["Overview", "Archetypes", "Recommendations", "Health", "System"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Overview");

  return (
    <div className="max-w-5xl mx-auto px-4 py-6 sm:p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">shadowtrace</h1>
        <p className="text-sm text-slate-500">Personal shadow tracing & model recommender</p>
      </header>

      <nav className="flex gap-2 mb-6 border-b border-slate-200 dark:border-slate-700 overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-2 text-sm whitespace-nowrap ${
              tab === t
                ? "border-b-2 border-blue-500 font-medium"
                : "text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
            }`}
          >
            {t}
          </button>
        ))}
      </nav>

      <main>
        {tab === "Overview" && <Overview />}
        {tab === "Archetypes" && <Archetypes />}
        {tab === "Recommendations" && <Recommendations />}
        {tab === "Health" && <Health />}
        {tab === "System" && <System />}
      </main>
    </div>
  );
}

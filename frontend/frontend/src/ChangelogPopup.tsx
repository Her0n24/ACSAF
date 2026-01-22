import { useState } from "react";

const changelogEntries = [
    {
    date: "2025.12.23",
    changes: [
      " Main Script: Logic overhaul of the cloud cover score",
      "Display: Now correctly displays cloud base level inferred from LCL if cloud base level cannot be determined from cloud cover",
    ],
  },
  {
    date: "2025.12.24",
    changes: [
      " Main Script: Fixed an issue where final scores used wrong AOD value. AOD scores and displayed AOD were global averages instead of city specific values",
      "Main Script: Changed AOD score logic"
    ],
  },
  {
    date: "2026.1.11",
    changes: [
      "Main Script: Changed the assumed elevation angle that can be viewed by an observer used in the computation of actual afterglow time from 15 to 5 degrees",
      "Main Script: Increased RH threshold for determining cloud base for liquid phase clouds (cloud layer where temperature is above 0 deg) from 85 to 95%. For supposed ice phase clouds, RH threshold remains at 80%",
      "Display: Now, score won’t be displayed when there is no cloud cover",
      "Display: Emphasis this is a score for CLOUD afterglow in the home page"
    ],
  }
];

export default function ChangelogPopup() {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: "fixed", top: 24, right: 24, zIndex: 1000 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="rounded-full bg-transparent px-4 py-2 text-xs font-bold text-white shadow-lg hover:scale-105 transition border border-white/30"
        aria-label="Show changelog"
      >
        {open ? "Close Changelog" : "Changelog"}
      </button>
      {open && (
        <div className="mt-2 w-80 rounded-2xl bg-black/90 p-4 text-white shadow-2xl border border-white/20 animate-fade-in">
          <h3 className="text-lg font-semibold mb-2">Changelog</h3>
          <ul className="space-y-3">
            {changelogEntries.map((entry) => (
              <li key={entry.date}>
                <div className="text-xs text-white/60 mb-1">{entry.date}</div>
                <ul className="list-disc ml-5 text-sm">
                  {entry.changes.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

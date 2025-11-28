import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { isAxiosError } from "axios";
import { fetchForecast } from "../api/client";

type ForecastDoc = Record<string, unknown> & { city?: string };
type DayKey = "tdy" | "tmr";
type Mode = "sunset" | "sunrise";
type OutlookRow = {
  metric: string;
  tdy?: string;
  tmr?: string;
};

const DAY_LABEL: Record<DayKey, string> = {
  tdy: "Today",
  tmr: "Tomorrow",
};
const EMPTY_FORECAST: ForecastDoc = {};

export default function Forecast() {
  const { city } = useParams<{ city: string }>();
  const [data, setData] = useState<ForecastDoc | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("sunset");

  useEffect(() => {
    if (!city) {
      setError("Missing city parameter");
      setLoading(false);
      return;
    }

    fetchForecast(city)
      .then(setData)
      .catch((err: unknown) => {
        if (isAxiosError(err)) {
          setError(err.response?.data?.error ?? err.message);
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Failed to load forecast");
        }
      })
      .finally(() => setLoading(false));
  }, [city]);

  const doc = data ?? EMPTY_FORECAST;
  const overviewItems = useMemo(() => buildOverview(doc), [doc]);
  const sunsetRows = useMemo(() => buildOutlookRows(doc, "sunset"), [doc]);
  const sunriseRows = useMemo(() => buildOutlookRows(doc, "sunrise"), [doc]);
  const activeRows = mode === "sunset" ? sunsetRows : sunriseRows;
  const likelihoodToday = getLikelihood(doc, mode, "tdy");
  const likelihoodTomorrow = getLikelihood(doc, mode, "tmr");
  const possibleColors = getPossibleColors(doc, mode, "tdy");
  const gradient = buildGradient(possibleColors);
  const modelRun = doc["run_time"] ? formatValue(doc["run_time"]) : null;

  if (loading) {
    return <FullScreenMessage message="Loading forecast…" />;
  }

  if (error || !data) {
    return (
      <FullScreenMessage
        message={error ?? "No forecast available."}
        actionLabel="Back to search"
        actionHref="/"
      />
    );
  }

  return (
    <main
      className="min-h-screen px-4 py-10 lg:py-16"
      style={gradient ? { backgroundImage: gradient } : undefined}
    >
      <div className="mx-auto flex max-w-4xl flex-col gap-8">
        <div className="flex flex-col gap-3">
          <Link to="/" className="text-sm text-white/70 hover:text-white">
            ← Back to search
          </Link>
          <p className="text-xs uppercase tracking-[0.5em] text-white/60">City Forecast</p>
          <h1 className="text-4xl font-semibold text-white">{data.city}</h1>
        </div>

        <section className="rounded-[32px] border border-white/15 bg-white/10 p-6 shadow-[0_30px_80px_rgba(0,0,0,0.45)] backdrop-blur-3xl">
          <div className="flex flex-col gap-6">
            <ModeSlider active={mode} onChange={setMode} />

            <div className="grid gap-4 md:grid-cols-2">
              <LikelihoodCard label="Today" value={likelihoodToday} />
              <LikelihoodCard label="Tomorrow" value={likelihoodTomorrow} subtle />
            </div>

            {(possibleColors.length > 0 || modelRun) && (
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                {possibleColors.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {possibleColors.map((color) => (
                      <span
                        key={color}
                        className="rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs uppercase tracking-wide text-white/80"
                      >
                        {color}
                      </span>
                    ))}
                  </div>
                )}

                {modelRun && (
                  <p className="text-xs uppercase tracking-[0.3em] text-white/60 sm:text-right">
                    Model run <span className="ml-2 text-white/90">{modelRun}</span>
                  </p>
                )}
              </div>
            )}
          </div>
        </section>

        {overviewItems.length > 0 && (
          <section className="rounded-[32px] border border-white/10 bg-white/5 p-6 backdrop-blur-2xl">
            <h2 className="mb-4 text-2xl font-semibold text-white">Timing</h2>
            <dl className="grid gap-4 sm:grid-cols-2">
              {overviewItems.map(({ label, value }) => (
                <div key={label} className="rounded-2xl border border-white/10 bg-black/20 p-4">
                  <dt className="text-xs uppercase tracking-wide text-white/60">{label}</dt>
                  <dd className="mt-2 text-lg font-medium text-white">{value}</dd>
                </div>
              ))}
            </dl>
          </section>
        )}

        <section className="rounded-[32px] border border-white/10 bg-white/5 p-6 backdrop-blur-2xl">
          <header className="mb-4 flex flex-col gap-1">
            <p className="text-xs uppercase tracking-[0.3em] text-white/60">{mode === "sunset" ? "Sunset" : "Sunrise"} metrics</p>
            <h2 className="text-2xl font-semibold text-white">Atmospheric Condition</h2>
          </header>
          <OutlookTable
            rows={activeRows}
            emptyMessage={`No ${mode} metrics available.`}
          />
        </section>
      </div>
    </main>
  );
}

function FullScreenMessage({
  message,
  actionLabel,
  actionHref,
}: {
  message: string;
  actionLabel?: string;
  actionHref?: string;
}) {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-4 px-4 text-center text-white/80">
      <p className="text-lg">{message}</p>
      {actionLabel && actionHref && (
        <Link
          to={actionHref}
          className="rounded-full bg-gradient-to-r from-[#f6a34d] via-[#f470a6] to-[#bf63f9] px-4 py-2 text-sm font-semibold text-slate-950"
        >
          {actionLabel}
        </Link>
      )}
    </main>
  );
}

function buildOverview(doc: ForecastDoc): Array<{ label: string; value: string }> {
  if (!doc) {
    return [];
  }

  const candidateKeys: Array<{ key: string; label: string }> = [
    { key: "forecast_time", label: "Forecast time" },
    { key: "sunrise_time_tdy", label: "Sunrise (today)" },
    { key: "sunrise_time_tmr", label: "Sunrise (tomorrow)" },
    { key: "sunset_time_tdy", label: "Sunset (today)" },
    { key: "sunset_time_tmr", label: "Sunset (tomorrow)" },
  ];

  return candidateKeys
    .filter(({ key }) => key in doc)
    .map(({ key, label }) => ({ label, value: formatValue(doc[key]) }));
}

function buildOutlookRows(doc: ForecastDoc, prefix: "sunset" | "sunrise"): OutlookRow[] {
  const rows = new Map<string, OutlookRow>();

  Object.entries(doc).forEach(([key, value]) => {
    if (!key.startsWith(`${prefix}_`)) {
      return;
    }

    const trimmed = key.replace(`${prefix}_`, "");
    const dayMatch = trimmed.match(/_(tdy|tmr)$/);
    const metricKey = dayMatch ? trimmed.replace(/_(tdy|tmr)$/, "") : trimmed;
    const day = (dayMatch?.[1] as DayKey | undefined) ?? "tdy";

    const current = rows.get(metricKey) ?? { metric: toTitle(metricKey) };
    current[day] = Array.isArray(value) ? value.map(formatValue).join(", ") : formatValue(value);
    rows.set(metricKey, current);
  });

  return Array.from(rows.values()).sort((a, b) => a.metric.localeCompare(b.metric));
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "string") {
    if (/^\d{4}-\d{2}-\d{2}T/.test(value)) {
      const date = new Date(value);
      if (!Number.isNaN(date.getTime())) {
        return date.toLocaleString();
      }
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatValue(item)).join(", ");
  }
  return JSON.stringify(value);
}

function toTitle(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function OutlookTable({ rows, emptyMessage }: { rows: OutlookRow[]; emptyMessage: string }) {
  if (rows.length === 0) {
    return <p className="text-sm text-white/70">{emptyMessage}</p>;
  }

  return (
    <div className="overflow-hidden rounded-3xl border border-white/10 bg-white/5">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="bg-white/10 text-xs uppercase tracking-wide text-white/70">
          <tr>
            <th scope="col" className="px-4 py-3 font-semibold">
              Metric
            </th>
            <th scope="col" className="px-4 py-3 font-semibold">
              {DAY_LABEL.tdy}
            </th>
            <th scope="col" className="px-4 py-3 font-semibold">
              {DAY_LABEL.tmr}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ metric, tdy, tmr }, index) => (
            <tr key={metric} className={index % 2 === 0 ? "bg-black/10" : "bg-transparent"}>
              <th scope="row" className="px-4 py-3 text-white">
                {metric}
              </th>
              <td className="px-4 py-3 text-white/90">{tdy ?? "—"}</td>
              <td className="px-4 py-3 text-white/90">{tmr ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function getLikelihood(doc: ForecastDoc, mode: Mode, day: DayKey): number | null {
  const key = `${mode}_likelihood_index_${day}`;
  const value = doc[key];
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string" && !Number.isNaN(Number(value))) {
    return Number(value);
  }
  return null;
}

function getPossibleColors(doc: ForecastDoc, mode: Mode, day: DayKey): string[] {
  const key = `${mode}_possible_colors_${day}`;
  const value = doc[key];
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  return [];
}

function buildGradient(colors: string[]): string | null {
  if (!colors.length) {
    return null;
  }

  const mapped = colors
    .slice(0, 3)
    .map((color, index) => `${mapColorName(color)} ${index === 0 ? "0%" : index === 1 ? "50%" : "100%"}`)
    .join(", ");

  return `linear-gradient(135deg, ${mapped})`;
}

function mapColorName(name: string): string {
  const presets: Record<string, string> = {
    "orange-red": "#e97b3ce8",
    "dark-red": "#991e1ea1",
    magenta: "#c155a4c3",
    "orange-yellow": "#987715c1",
    "golden-yellow": "#d8953d7c",
    "sunset-gold": "#9c843c2d",
    "golden-orange": "#b46a35d7",
  };
  return presets[name] ?? name;
}

function ModeSlider({ active, onChange }: { active: Mode; onChange: (mode: Mode) => void }) {
  const options: Mode[] = ["sunset", "sunrise"];
  return (
    <div className="relative flex rounded-full border border-white/15 bg-white/5 p-1 text-sm text-white/70">
      {options.map((option) => {
        const selected = option === active;
        return (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={`flex-1 rounded-full px-4 py-2 font-semibold transition ${
              selected
                ? "bg-white/30 text-slate-900 shadow-[0_10px_30px_rgba(0,0,0,0.25)]"
                : "text-white/70 hover:text-white"
            }`}
          >
            {option === "sunset" ? "Sunset" : "Sunrise"}
          </button>
        );
      })}
    </div>
  );
}

function LikelihoodCard({ label, value, subtle }: { label: string; value: number | null; subtle?: boolean }) {
  return (
    <div
      className={`rounded-[28px] border border-white/15 p-5 backdrop-blur-2xl ${
        subtle ? "bg-white/5" : "bg-gradient-to-br from-[#f6a34d]/35 via-[#f470a6]/30 to-[#bf63f9]/30"
      }`}
    >
      <p className="text-sm uppercase tracking-[0.3em] text-white/70">{label}</p>
      <div className="mt-3 flex items-baseline gap-3">
        <span className="text-6xl font-semibold text-white">{value ?? "—"}</span>
        <span className="text-sm text-white/60">Vibrancy Index</span>
      </div>
    </div>
  );
}

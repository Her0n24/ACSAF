import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { isAxiosError } from "axios";
import { fetchForecast } from "../api/client";
import { useLanguage } from "../context/LanguageContext";
import { t } from "../translations";

type DayKey = "tdy" | "tmr";
type Mode = "sunset" | "sunrise";
type LayerKey = "lcc" | "mcc" | "hcc" | "tcc";
type CloudProfilePayload = {
  distance_km?: unknown;
  lcc?: unknown;
  mcc?: unknown;
  hcc?: unknown;
  tcc?: unknown;
};
type ProfileKey = `${Mode}_cloud_profiles_${DayKey}`;
type ForecastDoc = Record<string, unknown> & {
  city?: string;
  country?: string;
} & {
  [key in ProfileKey]?: CloudProfilePayload;
};
type OutlookRow = {
  metric: string;
  tdy?: string;
  tmr?: string;
};

const METRIC_ORDER: string[] = [
  "afterglow_time",
  "cloud_present",
  "cloud_base_lvl",
  "cloud_local_cover",
  "avg_path",
  "cloud_layer_key",
  "azimuth",
  "geom_condition",
  "geom_condition_LCL_used",
  "hcc_condition",
];

const METRIC_LABEL_OVERRIDES: Record<string, string> = {
  afterglow_time: "metric.afterglowTime",
  cloud_present: "metric.cloudPresent",
  cloud_base_lvl: "metric.cloudBaseLevel",
  cloud_local_cover: "metric.cloudLocalCover",
  avg_path: "metric.avgPath",
  cloud_layer_key: "metric.cloudLayerReasoning",
  azimuth: "metric.azimuth",
  geom_condition: "metric.geomCondition",
  hcc_condition: "metric.hccCondition",
  geom_condition_LCL_used: "metric.geomConditionLcl"
};

const DAY_LABEL: Record<DayKey, string> = {
  tdy: "forecast.today",
  tmr: "forecast.tomorrow",
};
const EMPTY_FORECAST: ForecastDoc = {};

export default function Forecast() {
  const { city } = useParams<{ city: string }>();
  const [data, setData] = useState<ForecastDoc | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("sunset");
  const [profileDay, setProfileDay] = useState<DayKey>("tdy");
  const { language } = useLanguage();

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
  const overviewItems = useMemo(() => buildOverview(doc, language), [doc, language]);
  const sunsetRows = useMemo(() => buildOutlookRows(doc, "sunset", language), [doc, language]);
  const sunriseRows = useMemo(() => buildOutlookRows(doc, "sunrise", language), [doc, language]);
  const activeRows = mode === "sunset" ? sunsetRows : sunriseRows;
  const likelihoodToday = getLikelihood(doc, mode, "tdy");
  const likelihoodTomorrow = getLikelihood(doc, mode, "tmr");
  const cloudPresentToday = getCloudPresent(doc, mode, "tdy");
  const cloudPresentTomorrow = getCloudPresent(doc, mode, "tmr");
  const possibleColors = getPossibleColors(doc, mode, "tdy");
  const gradient = buildGradient(possibleColors);
  const modelRun = doc["run_time"] ? formatValue(doc["run_time"]) : null;
  const activeProfile = useMemo(() => getCloudProfile(doc, mode, profileDay), [doc, mode, profileDay]);
  const activeLayerKey = useMemo(() => getCloudLayerKey(doc, mode, profileDay), [doc, mode, profileDay]);

  if (loading) {
    return <FullScreenMessage message={t("forecast.loadingForecast", language)} />;
  }

  if (error || !data) {
    return (
      <FullScreenMessage
        message={error ?? t("forecast.noForecast", language)}
        actionLabel={t("forecast.backToSearch", language)}
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
            ← {t("forecast.backToSearch", language)}
          </Link>
          <p className="text-xs uppercase tracking-[0.5em] text-white/60">{t("forecast.cityForecast", language)}</p>
          <h1 className="text-4xl font-semibold text-white">{data.city}, {data.country}</h1>
        </div>

        <section className="rounded-[32px] border border-white/15 bg-white/10 p-6 shadow-[0_30px_80px_rgba(0,0,0,0.45)] backdrop-blur-3xl">
          <div className="flex flex-col gap-6">
            <ModeSlider active={mode} onChange={setMode} language={language} />

            <div className="grid gap-4 md:grid-cols-2">
              <LikelihoodCard label={t("forecast.today", language)} value={likelihoodToday} noClouds={cloudPresentToday === false} language={language} />
              <LikelihoodCard label={t("forecast.tomorrow", language)} value={likelihoodTomorrow} subtle noClouds={cloudPresentTomorrow === false} language={language} />
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
                        {translateColorName(color, language)}
                      </span>
                    ))}
                  </div>
                )}

                {modelRun && (
                  <p className="text-xs uppercase tracking-[0.15em] text-white/60 sm:text-right">
                    {t("forecast.modelRun", language)}
                    <span className="ml-2 text-white/90">{modelRun}Z</span>
                  </p>
                )}
              </div>
            )}
          </div>
        </section>

        {overviewItems.length > 0 && (
          <section className="rounded-[32px] border border-white/10 bg-white/5 p-6 backdrop-blur-2xl">
            <h2 className="mb-4 text-2xl font-semibold text-white">{t("forecast.timingLocalTime", language)}</h2>
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
            <p className="text-xs uppercase tracking-[0.3em] text-white/60">{mode === "sunset" ? t("forecast.sunset", language) : t("forecast.sunrise", language)} {t("forecast.metrics", language)}</p>
            <h2 className="text-2xl font-semibold text-white">{t("forecast.atmosphericCondition", language)}</h2>
          </header>
          <OutlookTable
            rows={activeRows}
            emptyMessage={`No ${mode} ${t("forecast.metrics", language).toLowerCase()} available.`}
            language={language}
          />
        </section>

        <section className="rounded-[32px] border border-white/10 bg-white/5 p-6 backdrop-blur-2xl">
          <header className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-white/60">{mode === "sunset" ? t("forecast.sunset", language) : t("forecast.sunrise", language)} {t("forecast.cloudProfile", language)}</p>
              <h2 className="text-2xl font-semibold text-white">{t("forecast.cloudCoverAlongAzimuth", language)}</h2>
            </div>
            <DayToggle value={profileDay} onChange={setProfileDay} language={language} />
          </header>

          {activeProfile ? (
            <>
              <CloudProfileChart profile={activeProfile} mode={mode} day={profileDay} language={language} layerKey={activeLayerKey} />
              <p className="mt-5 rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-white/80">
                {t("forecast.cloudInfo", language)}
                <p>&nbsp;</p>
              </p>
            </>
          ) : (
            <p className="rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-white/70">
              {t("forecast.noCloudProfile", language)} {t(DAY_LABEL[profileDay], language).toLowerCase()}.
            </p>
          )}
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

type CloudProfile = {
  distance: number[];
  series: Partial<Record<LayerKey, number[]>>;
};

function getCloudProfile(doc: ForecastDoc, mode: Mode, day: DayKey): CloudProfile | null {
  const key = `${mode}_cloud_profiles_${day}` as ProfileKey;
  const raw = doc[key];
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const distance = toNumberArray((raw as CloudProfilePayload).distance_km);
  if (!distance.length) {
    return null;
  }

  const series: Partial<Record<LayerKey, number[]>> = {};
  ( ["lcc", "mcc", "hcc", "tcc"] as LayerKey[] ).forEach((layer) => {
    const values = toNumberArray((raw as CloudProfilePayload)[layer]);
    if (values.length) {
      series[layer] = values.slice(0, distance.length);
    }
  });

  const hasAny = Object.values(series).some((arr) => arr && arr.length);
  return hasAny ? { distance, series } : null;
}

function toNumberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((entry) => {
      if (typeof entry === "number") {
        return entry;
      }
      if (typeof entry === "string" && entry.trim() !== "" && !Number.isNaN(Number(entry))) {
        return Number(entry);
      }
      return null;
    })
    .filter((entry): entry is number => entry !== null);
}

function DayToggle({ value, onChange, language }: { value: DayKey; onChange: (day: DayKey) => void; language: "en" | "zh" }) {
  return (
    <div className="inline-flex rounded-full border border-white/15 bg-white/5 p-1 text-xs text-white/80">
      {( ["tdy", "tmr"] as DayKey[] ).map((day) => {
        const selected = day === value;
        return (
          <button
            key={day}
            type="button"
            onClick={() => onChange(day)}
            className={`rounded-full px-4 py-1 font-semibold transition ${
              selected ? "bg-white/30 text-slate-900" : "text-white/70 hover:text-white"
            }`}
          >
            {t(DAY_LABEL[day], language)}
          </button>
        );
      })}
    </div>
  );
}

function CloudProfileChart({ profile, mode, day, language, layerKey }: { profile: CloudProfile; mode: Mode; day: DayKey; language: "en" | "zh"; layerKey?: LayerKey | null }) {
  const width = 900;
  const height = 420;
  const padding = 40;
  const innerShift = 12;
  const MAX_DISTANCE = 700; // km
  const leftBound = padding + innerShift;
  const rightBound = width - padding;
  const usableWidth = rightBound - leftBound;
  const usableHeight = height - padding * 2;

  const ALT_MAX_M = 9000; // vertical axis extent in meters
  const xScale = (distance: number) => leftBound + (Math.min(distance, MAX_DISTANCE) / MAX_DISTANCE) * usableWidth;
  const yScaleAltToSvg = (altM: number) => padding + usableHeight * (1 - Math.min(Math.max(altM, 0), ALT_MAX_M) / ALT_MAX_M);

  // sample positions for ray polylines (dense for smooth curves)
  const RAY_SAMPLES = 200;
  const sampleXsKm = Array.from({ length: RAY_SAMPLES }, (_, i) => (i / (RAY_SAMPLES - 1)) * MAX_DISTANCE);
  const sampleXsM = sampleXsKm.map((d) => d * 1000);

  // representative layer heights (meters)
  const LAYER_HEIGHTS: Record<LayerKey, number> = { lcc: 1000, mcc: 4000, hcc: 7500, tcc: 9000 };
  const N = profile.distance.length;
  const vals_lcc = profile.series.lcc ?? new Array(N).fill(0);
  const vals_mcc = profile.series.mcc ?? new Array(N).fill(0);
  const vals_hcc = profile.series.hcc ?? new Array(N).fill(0);

  // Vertical resolution for 'contourf' approximation
  const V_SAMPLES = 80;
  const alts = Array.from({ length: V_SAMPLES }, (_, i) => (i / (V_SAMPLES - 1)) * ALT_MAX_M);

  // build a discrete grid [vIndex][xIndex] by interpolating layer fractions by altitude
  const grid = useMemo(() => {
    const g: number[][] = Array.from({ length: V_SAMPLES }, () => new Array(N).fill(0));
    const hts = [LAYER_HEIGHTS.lcc, LAYER_HEIGHTS.mcc, LAYER_HEIGHTS.hcc];
    for (let xi = 0; xi < N; xi++) {
      const vvals = [vals_lcc[xi] ?? 0, vals_mcc[xi] ?? 0, vals_hcc[xi] ?? 0];
      for (let vi = 0; vi < V_SAMPLES; vi++) {
        const alt = alts[vi];
        let val = 0;
        if (alt <= hts[0]) {
          val = vvals[0];
        } else if (alt >= hts[2]) {
          val = vvals[2];
        } else {
          for (let k = 0; k < hts.length - 1; k++) {
            if (alt >= hts[k] && alt <= hts[k + 1]) {
              const tt = (alt - hts[k]) / (hts[k + 1] - hts[k]);
              val = (1 - tt) * vvals[k] + tt * vvals[k + 1];
              break;
            }
          }
        }
        g[vi][xi] = val;
      }
    }
    return g;
  }, [vals_lcc, vals_mcc, vals_hcc, profile.distance]);

  // No interactive hover state (interaction removed per user request)

  // Ray constants and generator
  const ALPHA_COEFF = -5.14e-5;
  const TIMESTEP_SECONDS = 60;
  const TIMESTEP_ARRAY = Array.from({ length: Math.floor(1080 / TIMESTEP_SECONDS) + 1 }, (_, i) => i * TIMESTEP_SECONDS);
  const R_EARTH_M = 6.371e6;

  function parabolicRay(xM: number, m: number, alpha: number, r: number, H: number) {
    return (xM - m) * Math.tan(alpha) + (0.5 * (xM - m) * (xM - m)) / r + H;
  }

  const H_for_rays = (layerKey && LAYER_HEIGHTS[layerKey]) || 0;
  const rays = TIMESTEP_ARRAY.map((t) => {
    const alpha = ALPHA_COEFF * t;
    const ys = sampleXsM.map((xM) => parabolicRay(xM, 0, alpha, R_EARTH_M, H_for_rays));
    return { t, alpha, ys };
  });

  // legend will be constructed from LAYER_BOUNDS/LAYER_COLOR below

  // interaction removed: cloud sampling helper not required when hover is disabled

  // Interaction removed: no mouse handlers or tooltip

  return (
    <div className="space-y-3">
      <div className="rounded-3xl border border-white/10 bg-black/20 p-4">
        <div style={{ position: "relative" }}>
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${t("forecast.cloudProfile", language)} for ${mode} ${t(DAY_LABEL[day], language)}`} className="w-full">
          <defs>
            <linearGradient id="gridFade" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(255, 255, 255, 0.75)" />
              <stop offset="100%" stopColor="rgba(255, 255, 255, 0.5)" />
            </linearGradient>
          </defs>
          <rect x={leftBound} y={padding} width={usableWidth} height={usableHeight} fill="url(#gridFade)" opacity={0.12} />
          {[0, 2000, 4000, 6000, 8000, 9000].map((alt) => (
            <g key={alt}>
              <line x1={leftBound} x2={rightBound} y1={yScaleAltToSvg(alt)} y2={yScaleAltToSvg(alt)} stroke="rgba(255, 255, 255, 0.51)" strokeDasharray="4 6" />
              <text x={leftBound - 20} y={yScaleAltToSvg(alt) + 4} fill="rgba(255,255,255,0.6)" fontSize="10" textAnchor="end">
                {alt >= 1000 ? `${alt / 1000} km` : `${alt} m`}
              </text>
            </g>
          ))}
          <text x={rightBound} y={height - padding + 20} textAnchor="end" fill="rgba(255,255,255,0.7)" fontSize="11">
            {MAX_DISTANCE} km
          </text>
          {/* removed cloud-percent reference line; axis now displays altitude */}

          {/* colorfill grid (approximate contourf) */}
          {grid.map((col, vi) => {
            const altTop = alts[vi + 1] ?? alts[vi];
            const altBottom = alts[vi];
            const yTop = yScaleAltToSvg(altTop);
            const cellHeight = Math.max(1, yScaleAltToSvg(altBottom) - yTop);
            return (
              <g key={`row-${vi}`}>
                {col.map((val, xi) => {
                  const x0 = xScale(profile.distance[xi] ?? 0);
                  const x1 = xScale(profile.distance[Math.min(xi + 1, profile.distance.length - 1)] ?? (profile.distance[xi] ?? 0));
                  const w = Math.max(1, x1 - x0);
                  // compute smooth grayscale fill per-cell based on fraction and vertical level
                  const frac = Math.min(Math.max(val, 0), 100) / 100;
                  const levelFactor = vi / Math.max(1, V_SAMPLES - 1);
                  // base intensity: higher fraction => darker (smaller number)
                  let gray = Math.round(300 - 180 * frac - 40 * levelFactor);
                  if (frac >= 0.9) {
                    // emphasize very high concentrations by darkening further
                    gray = Math.max(6, Math.round(gray * 0.9));
                  }
                  const grayClamped = Math.max(6, Math.min(230, gray));
                  const baseColor = `rgb(${grayClamped},${grayClamped},${grayClamped})`;
                  const opacity = Math.min(0.92, 0.04 + frac * 0.6 + levelFactor * 0.12);
                  return (
                    <rect
                      key={`cell-${vi}-${xi}`}
                      x={x0}
                      y={yTop}
                      width={w}
                      height={cellHeight}
                      fill={baseColor}
                      fillOpacity={opacity}
                      strokeOpacity={0}
                    />
                  );
                })}
              </g>
            );
          })}

          {/* parabolic rays overlay: skip any ray that goes below ground; cut off at ALT_MAX_M instead of clamping */}
          {rays.map((ray, ri) => {
            // skip entire ray if any point is below ground
            if (ray.ys.some((y) => y <= 0)) return null;

            // find first index where ray exceeds top altitude; draw up to (but not including) that index
            const firstAbove = ray.ys.findIndex((y) => y > ALT_MAX_M);
            const endIndex = firstAbove === -1 ? ray.ys.length : firstAbove;
            if (endIndex <= 0) return null;

            const pts: string[] = [];
            for (let i = 0; i < endIndex; i++) {
              const xKm = sampleXsKm[i];
              const x = xScale(xKm);
              const y = yScaleAltToSvg(ray.ys[i]);
              pts.push(`${x},${y}`);
            }

            return (
              <polyline
                key={`ray-${ri}`}
                points={pts.join(" ")}
                fill="none"
                stroke="rgba(168, 200, 255, 0.65)"
                strokeWidth={1}
                strokeLinecap="round"
                strokeLinejoin="round"
                pointerEvents="none"
              />
            );
          })}

          <line x1={leftBound} y1={height - padding} x2={rightBound} y2={height - padding} stroke="rgba(255, 255, 255, 0.51)" />
          <line x1={leftBound} y1={padding} x2={leftBound} y2={height - padding} stroke="rgba(255, 255, 255, 0.5)" />
          {/* x-axis ticks every 175 km */}
          {[0, 175, 350, 525, 700].map((d) => (
            <g key={`xtick-${d}`}>
              <line x1={xScale(d)} x2={xScale(d)} y1={height - padding} y2={height - padding + 6} stroke="rgba(255,255,255,0.5)" />
              <text x={xScale(d)} y={height - padding + 22} textAnchor="middle" fill="rgba(255,255,255,0.75)" fontSize="10">
                {d}
              </text>
            </g>
          ))}
          <text x={(leftBound + rightBound) / 2} y={height - 6} textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize="12">
            Distance along azimuth (km)
          </text>
          </svg>
          {/* tooltip removed (interaction disabled) */}
        </div>
      </div>
      {/* legend removed per user request */}
    </div>
  );
}

function buildOverview(doc: ForecastDoc, language: "en" | "zh"): Array<{ label: string; value: string }> {
  if (!doc) {
    return [];
  }

  const candidateKeys: Array<{ key: string; label: string }> = [
    { key: "forecast_time", label: "forecast.forecastTime" },
    { key: "sunrise_time_tdy", label: "forecast.sunriseToday" },
    { key: "sunset_time_tdy", label: "forecast.sunsetToday" },
  ];

  return candidateKeys
    .filter(({ key }) => key in doc)
    .map(({ key, label }) => ({ label: t(label, language), value: formatValue(doc[key]) }));
}

function buildOutlookRows(doc: ForecastDoc, prefix: "sunset" | "sunrise", language: "en" | "zh"): OutlookRow[] {
  const excluded = new Set([
    "cloud_profiles",
    "lf_ma",
    "likelihood_index",
    "possible_colors",
    "time",
  ]);
  const rows = new Map<string, OutlookRow>();

  Object.entries(doc).forEach(([key, value]) => {
    if (!key.startsWith(`${prefix}_`)) {
      return;
    }

    const trimmed = key.replace(`${prefix}_`, "");
    const dayMatch = trimmed.match(/_(tdy|tmr)$/);
    const metricKey = dayMatch ? trimmed.replace(/_(tdy|tmr)$/ , "") : trimmed;

    if (excluded.has(metricKey)) {
      return;
    }

    const day = (dayMatch?.[1] as DayKey | undefined) ?? "tdy";

    const displayLabel = METRIC_LABEL_OVERRIDES[metricKey] ? t(METRIC_LABEL_OVERRIDES[metricKey], language) : toTitle(metricKey);
    const current = rows.get(metricKey) ?? { metric: displayLabel };
    current[day] = Array.isArray(value)
      ? value.map((v) => formatMetricValue(metricKey, v, language)).join(", ")
      : formatMetricValue(metricKey, value, language);
    rows.set(metricKey, current);
  });

  const orderRank = new Map(METRIC_ORDER.map((key, index) => [key, index]));

  return Array.from(rows.entries())
    .sort(([keyA, rowA], [keyB, rowB]) => {
      const rankA = orderRank.get(keyA) ?? Number.MAX_SAFE_INTEGER;
      const rankB = orderRank.get(keyB) ?? Number.MAX_SAFE_INTEGER;
      if (rankA === rankB) {
        return rowA.metric.localeCompare(rowB.metric);
      }
      return rankA - rankB;
    })
    .map(([, row]) => row);
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
    // If ISO string with time, strip timezone and microseconds for 'local' time as in JSON
    const isoMatch = value.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})/);
    if (isoMatch) {
      return `${isoMatch[1]} ${isoMatch[2]}`;
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => formatValue(item)).join(", ");
  }
  return JSON.stringify(value);
}

function isNumeric(value: unknown): boolean {
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value === "string") return value.trim() !== "" && !Number.isNaN(Number(value));
  return false;
}

function formatMetricValue(metricKey: string, value: unknown, language: "en" | "zh"): string {
  if (value === null || value === undefined) return "—";

  // Cloud base level rounded to nearest 100 m
  if (metricKey === "cloud_base_lvl") {
    if (isNumeric(value)) {
      const num = Math.round(Number(value) / 100) * 100;
      return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(num);
    }
    return formatValue(value);
  }

  // Cloud cover percentages and average path rounded to nearest integer
  if (metricKey === "cloud_local_cover" || metricKey === "avg_path") {
    if (isNumeric(value)) {
      return String(Math.round(Number(value)));
    }
    return formatValue(value);
  }

  // Azimuth rounded to nearest integer
  if (metricKey === "azimuth") {
    if (isNumeric(value)) {
      return String(Math.round(Number(value)));
    }
    return formatValue(value);
  }

  // Afterglow duration: display as minutes:seconds
  if (metricKey === "afterglow_time") {
    if (isNumeric(value)) {
      const total = Math.max(0, Math.round(Number(value)));
      const mins = Math.floor(total / 60);
      const secs = total % 60;
      return `${mins}:${secs.toString().padStart(2, "0")}`;
    }
    return formatValue(value);
  }

  if (metricKey === "cloud_layer_key") {
    // Accept either a string key (e.g. 'lcc') or an object mapping
    // like { lcc: true, mcc: false, hcc: false } and pick the
    // highest-priority present layer.
    if (typeof value === "string") {
      return translateColorName(value, language);
    }
    if (value && typeof value === "object") {
      try {
        const priority = ["hcc", "mcc", "lcc"];
        for (const k of priority) {
          if (Object.prototype.hasOwnProperty.call(value, k) && (value as any)[k]) {
            return translateColorName(k, language);
          }
        }
        // If values are numeric scores, pick the key with the largest value
        const entries = Object.entries(value as Record<string, any>);
        const numeric = entries.filter(([, v]) => typeof v === "number");
        if (numeric.length > 0) {
          const best = numeric.reduce((a, b) => (a[1] >= b[1] ? a : b));
          return translateColorName(best[0], language);
        }
      } catch (e) {
        /* fallthrough to default formatting */
      }
    }
    return formatValue(value);
  }

  if (Array.isArray(value)) return value.map((v) => formatMetricValue(metricKey, v, language)).join(", ");
  return formatValue(value);
}

function toTitle(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function OutlookTable({ rows, emptyMessage, language }: { rows: OutlookRow[]; emptyMessage: string; language?: "en" | "zh" }) {
  if (rows.length === 0) {
    return <p className="text-sm text-white/70">{emptyMessage}</p>;
  }

  return (
    <div className="overflow-hidden rounded-3xl border border-white/10 bg-white/5">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="bg-white/10 text-xs uppercase tracking-wide text-white/70">
          <tr>
            <th scope="col" className="px-4 py-3 font-semibold">
              {language ? t("forecast.metricHeader", language) : "Metric"}
            </th>
            <th scope="col" className="px-4 py-3 font-semibold">
              {language ? t(DAY_LABEL.tdy, language) : DAY_LABEL.tdy}
            </th>
            <th scope="col" className="px-4 py-3 font-semibold">
              {language ? t(DAY_LABEL.tmr, language) : DAY_LABEL.tmr}
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

function getCloudPresent(doc: ForecastDoc, mode: Mode, day: DayKey): boolean | null {
  const key = `${mode}_cloud_present_${day}`;
  const value = doc[key];
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const v = value.trim().toLowerCase();
    if (v === "no" || v === "false") return false;
    if (v === "yes" || v === "true") return true;
    return null;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  return null;
}

function getCloudLayerKey(doc: ForecastDoc, mode: Mode, day: DayKey): LayerKey | null {
  const key = `${mode}_cloud_layer_key_${day}`;
  const value = doc[key];
  if (typeof value === "string") {
    const v = value.trim().toLowerCase();
    if (v === "lcc" || v === "mcc" || v === "hcc" || v === "tcc") return v as LayerKey;
    return null;
  }
  if (value && typeof value === "object") {
    try {
      const priority: LayerKey[] = ["hcc", "mcc", "lcc"];
      for (const k of priority) {
        if (Object.prototype.hasOwnProperty.call(value, k) && (value as any)[k]) return k;
      }
      const entries = Object.entries(value as Record<string, any>).filter(([, v]) => typeof v === "number");
      if (entries.length > 0) {
        const best = entries.reduce((a, b) => (a[1] >= b[1] ? a : b));
        if (["lcc", "mcc", "hcc", "tcc"].includes(best[0])) return best[0] as LayerKey;
      }
    } catch (e) {
      /* ignore */
    }
  }
  return null;
}

function describeGlow(value: number | null, language: "en" | "zh"): string {
  if (value === null || Number.isNaN(value)) {
    return t("forecast.awaiting", language);
  }
  if (value <= 0) {
    return t("forecast.noGlow", language);
  }
  if (value <= 15) {
    return t("forecast.dullGlow", language);
  }
  if (value <= 40) {
    return t("forecast.moderateGlow", language);
  }
  if (value <= 60) {
    return t("forecast.vividGlow", language);
  }
  if (value <= 75) {
    return t("forecast.flamingGlow", language);
  }
  return t("forecast.flamboyantGlow", language);
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

function translateColorName(name: string, language: "en" | "zh"): string {
  const key = `color.${name.toLowerCase()}`;
  const translated = t(key, language);
  return translated === key ? name : translated;
}

function ModeSlider({ active, onChange, language }: { active: Mode; onChange: (mode: Mode) => void; language: "en" | "zh" }) {
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
            {option === "sunset" ? t("forecast.sunset", language) : t("forecast.sunrise", language)}
          </button>
        );
      })}
    </div>
  );
}

function LikelihoodCard({ label, value, subtle, noClouds, language }: { label: string; value: number | null; subtle?: boolean; noClouds?: boolean; language: "en" | "zh" }) {
  const describeText = noClouds ? t("forecast.noClouds", language) : describeGlow(value, language);
  return (
    <div
      className={`rounded-[28px] border border-white/15 p-5 backdrop-blur-2xl ${
        subtle ? "bg-white/5" : "bg-gradient-to-br from-[#f6a34d]/35 via-[#f470a6]/30 to-[#bf63f9]/30"
      }`}
    >
      <p className="text-sm uppercase tracking-[0.3em] text-white/70">{label}</p>
      <div className="mt-3 flex items-baseline gap-3">
        <span className="text-6xl font-semibold text-white">
          {noClouds ? "--" : (value ?? "—-")}
        </span>
        <span className="text-sm text-white/60">{describeText}</span>
      </div>
    </div>
  );
}

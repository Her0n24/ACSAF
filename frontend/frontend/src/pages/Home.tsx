import { useEffect, useMemo, useState } from "react";
import ChangelogPopup from "../ChangelogPopup";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { isAxiosError } from "axios";
import { fetchCities, type CityRecord } from "../api/client";

const formatCityLabel = (option: CityRecord) =>
  option.country ? `${option.city}, ${option.country}` : option.city;

export default function Home() {
  const [cities, setCities] = useState<CityRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchCities()
      .then(setCities)
      .catch((err: unknown) => {
        if (isAxiosError(err)) {
          setError(err.response?.data?.error ?? err.message);
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Failed to load cities");
        }
      })
      .finally(() => setLoading(false));
  }, []);

  const filteredCities = useMemo(() => {
    if (!query.trim()) {
      return cities;
    }
    const lower = query.trim().toLowerCase();
    return cities.filter((option) => formatCityLabel(option).toLowerCase().includes(lower));
  }, [cities, query]);

  const suggestions = useMemo(() => filteredCities.slice(0, 8), [filteredCities]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    const normalized = query.trim().toLowerCase();
    const bestMatch =
      filteredCities[0] ??
      cities.find(
        (option) =>
          formatCityLabel(option).toLowerCase() === normalized || option.city.toLowerCase() === normalized
      );
    if (bestMatch) {
      navigate(`/forecast/${encodeURIComponent(bestMatch.city)}`);
      setQuery(formatCityLabel(bestMatch));
      setFocused(false);
    }
  };

  const handleSuggestionClick = (option: CityRecord) => {
    navigate(`/forecast/${encodeURIComponent(option.city)}`);
    setQuery(formatCityLabel(option));
    setFocused(false);
  };

  if (loading) {
    return <FullScreenMessage message="Loading cities…" />;
  }

  if (error) {
    return <FullScreenMessage message={`Error: ${error}`} />;
  }

  return (
    <>
      <ChangelogPopup />
      <main className="min-h-screen px-4 pb-16">
      <section className="mx-auto flex min-h-[90vh] max-w-5xl items-center justify-center pt-12">
        <div className="w-full max-w-3xl space-y-10 rounded-[32px] border border-white/15 bg-white/10 p-10 shadow-[0_30px_80px_rgba(0,0,0,0.45)] backdrop-blur-3xl">
          <header className="text-center space-y-4">
            <p className="text-xs uppercase tracking-[0.4em] text-white/70">Project ACSAF</p>
            <h1 className="text-4xl font-semibold text-white">Cloud Afterglow Forecast</h1>
            <p className="text-base text-white/80">
              Catch the next brilliant dusk/ dawn cloud glow with AI-powered NWP forecast!
            </p>
          </header>

          <form className="relative" onSubmit={handleSubmit} autoComplete="off">
            <label htmlFor="city-search" className="sr-only">
              Search cities
            </label>
            <div className="flex gap-3 rounded-2xl border border-white/20 bg-white/5 px-5 py-4 backdrop-blur">
              <div className="flex flex-col flex-1">
                <input
                  id="city-search"
                  type="text"
                  placeholder="Start typing a city…"
                  className="bg-transparent text-lg text-white placeholder:text-white/40 focus:outline-none"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onFocus={() => setFocused(true)}
                  onBlur={() => setTimeout(() => setFocused(false), 120)}
                />
                <p className="text-xs text-white/60">{cities.length} cities available</p>
              </div>
              <button
                type="submit"
                disabled={!query.trim()}
                className="rounded-xl bg-gradient-to-r from-[#f6a34d] via-[#f470a6] to-[#bf63f9] px-5 py-2 font-semibold text-slate-950 transition disabled:opacity-50"
              >
                Go
              </button>
            </div>

            {query && focused && suggestions.length > 0 && (
              <ul className="absolute left-0 right-0 mt-3 max-h-72 overflow-auto rounded-2xl border border-white/15 bg-slate-900/80 p-2 shadow-2xl backdrop-blur">
                {suggestions.map((option) => (
                  <li key={`${option.city}-${option.country ?? ""}`}>
                    <button
                      type="button"
                      className="w-full rounded-xl px-4 py-3 text-left text-white hover:bg-white/10"
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => handleSuggestionClick(option)}
                    >
                      {formatCityLabel(option)}
                    </button>
                  </li>
                ))}
              </ul>
            )}

            {query && filteredCities.length === 0 && (
              <p className="mt-2 text-sm text-rose-200">No matching cities found.</p>
            )}
          </form>
        </div>
      </section>

      <section className="mx-auto mt-35 flex max-w-5xl flex-col gap-12 rounded-[36px] border border-white/10 bg-gradient-to-br from-white/10 via-white/5 to-white/0 p-10 text-white shadow-[0_30px_80px_rgba(0,0,0,0.55)] backdrop-blur-2xl">
        <div className="space-y-4 text-center">
          <h2 className="text-4xl font-semibold">ACSAF: The Aerosol and Cloud-Based Sunset/ Sunrise cloud Afterglow Forecaster</h2>
          <p className="mx-auto max-w-3xl text-base text-white/80">
            ACSAF is a quick and handy reference system to landscape photographers, cloud lovers, or anyone captivated by clouds' fleeting colors around the world, leveraging 
            AI-powered Numerical Weather Prediction (NWP) outputs. Daily cloud glow displays are rated out of 100. Save travel time and effort for the next best cloud glows!
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {["ECMWF AIFS data", "CAMS aerosol data", "Update twice daily"].map((title, index) => {
            const body = [
              "Gold standard global cloud cover, temperature and humidity data for computing cloud geometry effects.",
              "High resolution total aerosol and dust measurements to assist in determining afterglow color and vibrancy.",
              "Get the latest forecast updates twice daily well ahead of sunset and sunrise times to plan for your trip.",
            ][index];
            return (
              <article
                key={title}
                className="rounded-3xl border border-white/10 bg-black/30 p-5 shadow-[0_20px_40px_rgba(0,0,0,0.45)]"
              >
                <h3 className="text-xl font-semibold">{title}</h3>
                <p className="mt-3 text-sm text-white/80">{body}</p>
              </article>
            );
          })}
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <article className="rounded-3xl border border-white/10 bg-white/5 p-6">
            <p className="text-xs uppercase tracking-[0.4em] text-white/60">How it works</p>
            <h3 className="mt-2 text-2xl font-semibold">Outline</h3>
            <ul className="mt-4 space-y-3 text-white/80">
              <li>
                <span className="font-semibold text-white">1.</span> Score ranges 0-100. Higher indicates more favourable 
                cloud afterglow conditions for occurence and intense colors
              </li>
              <li>
                <span className="font-semibold text-white">2.</span> Afterglow time and formation condition based on Earth geometry and cloud cover
              </li>
              <li>
                <span className="font-semibold text-white">3.</span> Possible colours are based on the dominant cloud layer present and the aerosol content
                near the city
              </li>
            </ul>
          </article>
          <article className="rounded-3xl border border-white/10 bg-white/5 p-6">
            <p className="text-xs uppercase tracking-[0.4em] text-white/60">About this tool</p>
            <h3 className="mt-2 text-2xl font-semibold">Limitations</h3>
            <p className="mt-4 text-white/80">
              The computed scores are valid only for stratiform cloud layer. Convective clouds exhibit vertical structure, which 
              generally allows afterglow to appear more easily. Raw NWP ouput should be interpreted with caution as they are uncalibrated.  
            </p>
            <p className="mt-3 text-white/80">
              ACSAF scores are subjective
          and still lacks a holistic quantitative forecast verification. 
            </p>
            <p className="mt-3 text-white/80">
              AIFS is not designed for short term or high resolution forecasting with its 6
              hours temporal resolution. 
            </p>
          </article>
        </div>
      </section>

      <section className="mx-auto mt-16 max-w-5xl rounded-3xl border border-white/10 bg-black/30 p-8 text-white/80 shadow-[0_20px_50px_rgba(0,0,0,0.6)] backdrop-blur">
        <p className="text-xs uppercase tracking-[0.4em] text-white/60">Disclaimer</p>
        <p className="mt-3 text-sm leading-relaxed">
          Forecast guidance is experimental and for discussion only. 
          ACSAF assumes no responsibility for errors or omissions in the content that directly or indirectly result in any damange or loss caused or alleged to be caused
          by or in connection with use on the content.
        </p>
      </section>
      </main>
    </>
  );
}

function FullScreenMessage({ message }: { message: string }) {
  return (
    <main className="min-h-screen flex items-center justify-center px-4">
      <p className="text-lg text-white/80">{message}</p>
    </main>
  );
}
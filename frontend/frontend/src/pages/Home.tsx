import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { isAxiosError } from "axios";
import { fetchCities } from "../api/client";

export default function Home() {
  const [cities, setCities] = useState<string[]>([]);
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
    return cities.filter((city) => city.toLowerCase().includes(lower));
  }, [cities, query]);

  const suggestions = useMemo(() => filteredCities.slice(0, 8), [filteredCities]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) {
      return;
    }
    const bestMatch = filteredCities[0] ?? cities.find((city) => city.toLowerCase() === query.toLowerCase());
    if (bestMatch) {
      navigate(`/forecast/${encodeURIComponent(bestMatch)}`);
      setFocused(false);
    }
  };

  const handleSuggestionClick = (city: string) => {
    navigate(`/forecast/${encodeURIComponent(city)}`);
    setQuery(city);
    setFocused(false);
  };

  if (loading) {
    return <FullScreenMessage message="Loading cities…" />;
  }

  if (error) {
    return <FullScreenMessage message={`Error: ${error}`} />;
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-4 py-16">
      <div className="w-full max-w-2xl space-y-10 rounded-[32px] border border-white/15 bg-white/10 p-10 shadow-[0_30px_80px_rgba(0,0,0,0.45)] backdrop-blur-3xl">
        <header className="text-center space-y-4">
          <p className="text-xs uppercase tracking-[0.5em] text-white/70">Project ACSAF</p>
          <h1 className="text-4xl font-semibold text-white">Afterglow Forecast</h1>
          <p className="text-base text-white/80">
            Discover the latest forecast of sunset and sunrise glow score out of 100.
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
              {suggestions.map((city) => (
                <li key={city}>
                  <button
                    type="button"
                    className="w-full rounded-xl px-4 py-3 text-left text-white hover:bg-white/10"
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => handleSuggestionClick(city)}
                  >
                    {city}
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
    </main>
  );
}

function FullScreenMessage({ message }: { message: string }) {
  return (
    <main className="min-h-screen flex items-center justify-center px-4">
      <p className="text-lg text-white/80">{message}</p>
    </main>
  );
}

import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:5000",
  timeout: 8000,
});

export type CityRecord = {
  city: string;
  country?: string;
};

type LegacyCityResponse = { cities?: Array<string | CityRecord>; available_cities?: string[] };

export const fetchCities = async (): Promise<CityRecord[]> => {
  const response = await api.get<LegacyCityResponse>("/cities");
  const rawList = response.data.cities ?? response.data.available_cities ?? [];

  return rawList.map((entry) => {
    if (typeof entry === "string") {
      return { city: entry };
    }
    return {
      city: entry.city,
      country: entry.country ?? undefined,
    };
  });
};

export const fetchForecast = async (city: string) => {
  const response = await api.get("/forecast", { params: { city } });
  return response.data;
};

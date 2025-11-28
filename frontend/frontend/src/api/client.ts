import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:5000",
  timeout: 8000,
});

export const fetchCities = async () => {
  const response = await api.get<{ cities?: string[]; available_cities?: string[] }>("/cities");
  return response.data.cities ?? response.data.available_cities ?? [];
};

export const fetchForecast = async (city: string) => {
  const response = await api.get("/forecast", { params: { city } });
  return response.data;
};

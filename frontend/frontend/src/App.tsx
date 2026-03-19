import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Forecast from "./pages/Forecast";
import { LanguageProvider } from "./context/LanguageContext";

export default function App() {
  return (
    <LanguageProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/forecast/:city" element={<Forecast />} />
        </Routes>
      </BrowserRouter>
    </LanguageProvider>
  );
}

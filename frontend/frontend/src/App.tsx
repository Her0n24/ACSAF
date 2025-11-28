import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Forecast from "./pages/Forecast";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/forecast/:city" element={<Forecast />} />
      </Routes>
    </BrowserRouter>
  );
}

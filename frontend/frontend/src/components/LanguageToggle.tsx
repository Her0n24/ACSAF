import { useLanguage } from "../context/LanguageContext";

export default function LanguageToggle() {
  const { language, setLanguage } = useLanguage();

  return (
    <div style={{ position: "fixed", top: 24, left: 24, zIndex: 1000 }}>
      <button
        onClick={() => setLanguage(language === "en" ? "zh" : "en")}
        className="rounded-full bg-transparent px-4 py-2 text-xs font-bold text-white shadow-lg hover:scale-105 transition border border-white/30"
        aria-label="Toggle language"
      >
        {language === "en" ? "繁中" : "English"}
      </button>
    </div>
  );
}

import { useEffect, useState } from "react";
import { Github, Activity, BrainCircuit } from "lucide-react";

import UploadCard from "./components/UploadCard";
import ResultCard from "./components/ResultCard";
import ModelInfo from "./components/ModelInfo";
import { getModelInfo, recognizeFace } from "./services/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch(() =>
        setError("Start the FastAPI backend to load model information.")
      );
  }, []);

  async function handleRecognize() {
    if (!file) return;
    setLoading(true);
    setError("");

    try {
      const response = await recognizeFace(file);
      setResult(response.result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <nav className="navbar">
        <div className="brand">
          <BrainCircuit size={26} />
          <span>PCA Face Recognition</span>
        </div>
        <a
          href="https://github.com/"
          target="_blank"
          rel="noreferrer"
          className="icon-link"
        >
          <Github size={20} />
        </a>
      </nav>

      <header className="hero">
        <div>
          <p className="eyebrow">Computer Vision • PCA • Eigenfaces</p>
          <h1>Understand a face through its principal components.</h1>
          <p className="hero-text">
            Upload an image and explore dimensionality reduction,
            eigenfaces and nearest-neighbor recognition.
          </p>
        </div>
        <div className="status-pill">
          <Activity size={16} />
          FastAPI + React
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="content-grid">
        <UploadCard
          file={file}
          setFile={setFile}
          onRecognize={handleRecognize}
          loading={loading}
        />
        <ResultCard result={result} />
      </div>

      <div className="single-section">
        <ModelInfo info={info} />
      </div>

      <section className="card methodology">
        <p className="eyebrow">Methodology</p>
        <h2>How the system solves the problem</h2>
        <div className="pipeline">
          {[
            ["01", "Preprocess", "Grayscale, face detection and 64×64 resize"],
            ["02", "PCA", "Center data and calculate principal eigenvectors"],
            ["03", "Project", "Map the face into eigenface space"],
            ["04", "Recognize", "Compare projections with Euclidean distance"]
          ].map(([number, title, text]) => (
            <div className="pipeline-step" key={number}>
              <span>{number}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
          ))}
        </div>
      </section>

      <footer>PCA Face Recognition • Academic Project • Persons 1–4</footer>
    </main>
  );
}

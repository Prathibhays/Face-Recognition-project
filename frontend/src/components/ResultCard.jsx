import { CheckCircle2, ScanFace } from "lucide-react";

export default function ResultCard({ result }) {
  if (!result) {
    return (
      <section className="card empty-state">
        <ScanFace size={38} />
        <h2>Recognition result</h2>
        <p>Upload an image to see the PCA recognition result.</p>
      </section>
    );
  }

  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Result</p>
          <h2>Face recognized</h2>
        </div>
        <CheckCircle2 size={22} />
      </div>

      <div className="result-person">
        <span>Predicted person</span>
        <strong>Person {result.person}</strong>
      </div>

      <div className="metric-grid">
        <div className="metric">
          <span>Similarity</span>
          <strong>{result.similarity}%</strong>
        </div>
        <div className="metric">
          <span>Distance</span>
          <strong>{result.distance}</strong>
        </div>
        <div className="metric">
          <span>PCA components</span>
          <strong>{result.components}</strong>
        </div>
        
      </div>

      <h3 className="neighbors-title">Nearest neighbors</h3>
      <div className="neighbors">
        {result.neighbors.map((n, i) => (
          <div className="neighbor" key={`${n.person}-${i}`}>
            <span>Person {n.person}</span>
            <span>{n.distance.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

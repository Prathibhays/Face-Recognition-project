import { useRef } from "react";
import { Upload } from "lucide-react";

export default function UploadCard({ file, setFile, onRecognize, loading }) {
  const inputRef = useRef(null);

  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Recognition</p>
          <h2>Upload a face image</h2>
        </div>
        <Upload size={22} />
      </div>

      <div className="dropzone" onClick={() => inputRef.current?.click()}>
        <Upload size={34} />
        <strong>{file ? file.name : "Choose a JPG or PNG image"}</strong>
        <span>Face detection is attempted automatically.</span>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          hidden
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </div>

      <button
        className="primary-button"
        disabled={!file || loading}
        onClick={onRecognize}
      >
        {loading ? "Recognizing..." : "Recognize Face"}
      </button>
    </section>
  );
}

export default function ModelInfo({ info }) {
  if (!info) return null;

  const items = [
    ["Dataset", info.dataset],
    ["Image size", info.image_size],
    ["Training samples", info.training_samples],
    ["Testing samples", info.testing_samples],
    ["Original features", info.original_features],
    ["PCA components", info.pca_components],
    ["Validation accuracy", `${info.validation_accuracy}%`],
    ["Variance retained", `${info.explained_variance}%`]
  ];

  return (
    <section className="card">
      <p className="eyebrow">Model</p>
      <h2>Training configuration</h2>
      <div className="info-grid">
        {items.map(([label, value]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

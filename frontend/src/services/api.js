const API_URL = "http://127.0.0.1:8000/api";

export async function recognizeFace(file) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_URL}/recognize`, {
    method: "POST",
    body: form
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Recognition failed");
  return data;
}

export async function getModelInfo() {
  const response = await fetch(`${API_URL}/model-info`);
  if (!response.ok) throw new Error("Unable to load model information");
  return response.json();
}

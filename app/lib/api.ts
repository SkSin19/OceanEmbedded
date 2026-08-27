// Talks to the FastAPI inference backend, with a static-file fallback so the
// dashboard still works when the backend is not running.

import type {
  Embedding,
  Meta,
  Metrics,
  ModelName,
  Prediction,
  Reconstruction,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8010";

export type Source = "api" | "static";

async function json<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
  return res.json();
}

export async function detectSource(): Promise<Source> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(1500),
    });
    return res.ok ? "api" : "static";
  } catch {
    return "static";
  }
}

export function loadMeta(source: Source): Promise<Meta> {
  return json<Meta>(source === "api" ? `${API_BASE}/meta` : "/data/meta.json");
}

export async function loadModels(source: Source): Promise<ModelName[]> {
  if (source !== "api") return ["baseline"];
  try {
    const r = await json<{ available: ModelName[] }>(`${API_BASE}/models`);
    return r.available;
  } catch {
    return ["baseline"];
  }
}

export type Validation = "holdout" | "argo";

export function loadMetrics(
  source: Source,
  model: ModelName,
  validation: Validation = "holdout"
): Promise<Metrics> {
  if (validation === "argo") {
    return json<Metrics>(
      source === "api" ? `${API_BASE}/metrics?validation=argo` : "/data/metrics_argo.json"
    );
  }
  const staticName = model === "cnn" ? "/data/metrics_cnn.json" : "/data/metrics.json";
  return json<Metrics>(source === "api" ? `${API_BASE}/metrics?model=${model}` : staticName);
}

export function loadPrediction(source: Source, date: string, model: ModelName): Promise<Prediction> {
  return json<Prediction>(
    source === "api"
      ? `${API_BASE}/prediction?date=${date}&model=${model}`
      : "/data/prediction_sample.json"
  );
}

export function loadEmbedding(source: Source, date: string): Promise<Embedding> {
  return json<Embedding>(
    source === "api" ? `${API_BASE}/embedding?date=${date}` : "/data/embedding_sample.json"
  );
}

// The playground needs the live backend to run inference on modified inputs.
export async function reconstructWhatIf(
  date: string,
  model: ModelName,
  sstOffset: number,
  sshOffset: number
): Promise<Reconstruction> {
  const res = await fetch(`${API_BASE}/reconstruct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date, model, sst_offset: sstOffset, ssh_offset: sshOffset }),
  });
  if (!res.ok) throw new Error(`reconstruct failed (${res.status})`);
  return res.json();
}

export async function reconstructImage(
  date: string,
  model: ModelName,
  file: File
): Promise<Reconstruction> {
  const form = new FormData();
  form.append("file", file);
  form.append("date", date);
  form.append("model", model);
  const res = await fetch(`${API_BASE}/reconstruct/image`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`image reconstruct failed (${res.status})`);
  return res.json();
}

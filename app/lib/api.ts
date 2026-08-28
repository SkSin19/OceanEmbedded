// Talks to the FastAPI inference backend, with a static-file fallback so the
// dashboard still works when the backend is not running.

import type {
  Embedding,
  Meta,
  Metrics,
  ModelName,
  Prediction,
  Profile,
  Reconstruction,
  Surface,
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

// ---- single-column time series --------------------------------------------
// The volume explorer needs the same water column on every available date.
// /profile returns one column (~30 numbers) instead of a full 15-level grid,
// so a 31-day series costs less over the wire than a single /prediction call.

export interface ColumnFrame {
  date: string;
  predicted: (number | null)[];
  truth: (number | null)[];
}

const profileCache = new Map<string, Promise<Profile>>();

function profileKey(date: string, lat: number, lon: number, model: ModelName): string {
  return `${model}|${date}|${lat.toFixed(3)},${lon.toFixed(3)}`;
}

/** loadProfile, memoised per (model, date, point) for the life of the tab. */
export function loadProfileCached(
  date: string,
  lat: number,
  lon: number,
  model: ModelName
): Promise<Profile> {
  const key = profileKey(date, lat, lon, model);
  let p = profileCache.get(key);
  if (!p) {
    p = loadProfile(date, lat, lon, model).catch((e) => {
      profileCache.delete(key);
      throw e;
    });
    profileCache.set(key, p);
  }
  return p;
}

/** Seed the cache with a column already sliced out of a loaded Prediction. */
export function seedProfileCache(profile: Profile, date: string, model: ModelName): void {
  profileCache.set(profileKey(date, profile.lat, profile.lon, model), Promise.resolve(profile));
}

/** Evenly thin a date list down to at most `max` entries, keeping both ends. */
export function resampleTimes(times: string[], max: number): string[] {
  if (times.length <= max || max < 2) return times;
  const out: string[] = [];
  for (let i = 0; i < max; i++) {
    out.push(times[Math.round((i / (max - 1)) * (times.length - 1))]);
  }
  return [...new Set(out)];
}

/**
 * Visit order that fills the range by repeated subdivision (ends, midpoint,
 * quarters, ...). A partly-loaded series then spans the whole window instead
 * of only its first few days, so scrubbing is useful long before it finishes.
 */
function spreadOrder(n: number): number[] {
  const out: number[] = [];
  const seen = new Set<number>();
  const push = (i: number) => {
    if (i >= 0 && i < n && !seen.has(i)) {
      seen.add(i);
      out.push(i);
    }
  };
  push(0);
  push(n - 1);
  for (let step = n - 1; step > 1; step = Math.ceil(step / 2)) {
    for (let i = 0; i < n; i += step) push(i);
  }
  for (let i = 0; i < n; i++) push(i);
  return out;
}

export interface ColumnSeriesOptions {
  concurrency?: number;
  signal?: AbortSignal;
  /** Called as each frame lands, so the timeline can fill in progressively. */
  onFrame?: (frame: ColumnFrame, loaded: number, total: number) => void;
}

/**
 * Fetch the column at one point across many dates. Frames resolve out of order
 * but are reported with their date, so callers can slot them into place.
 */
export async function loadColumnSeries(
  dates: string[],
  lat: number,
  lon: number,
  model: ModelName,
  { concurrency = 4, signal, onFrame }: ColumnSeriesOptions = {}
): Promise<ColumnFrame[]> {
  const frames: ColumnFrame[] = [];
  const order = spreadOrder(dates.length);
  let next = 0;
  let loaded = 0;

  const worker = async () => {
    while (next < order.length) {
      if (signal?.aborted) return;
      const date = dates[order[next++]];
      try {
        const p = await loadProfileCached(date, lat, lon, model);
        if (signal?.aborted) return;
        const frame: ColumnFrame = { date, predicted: p.predicted, truth: p.truth };
        frames.push(frame);
        onFrame?.(frame, ++loaded, dates.length);
      } catch {
        loaded++; // a missing day should not stall the rest of the series
      }
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(concurrency, dates.length) }, () => worker())
  );
  return frames.sort((a, b) => a.date.localeCompare(b.date));
}

export function loadEmbedding(source: Source, date: string): Promise<Embedding> {
  return json<Embedding>(
    source === "api" ? `${API_BASE}/embedding?date=${date}` : "/data/embedding_sample.json"
  );
}

// Profile at an arbitrary lat/lon. With the backend this is a real forward pass
// at the nearest grid cell; without it, the caller samples the static grid.
export function loadProfile(
  date: string,
  lat: number,
  lon: number,
  model: ModelName
): Promise<Profile> {
  return json<Profile>(
    `${API_BASE}/profile?lat=${lat}&lon=${lon}&date=${date}&model=${model}`
  );
}

// Real harmonized surface inputs for a day. Static fallback reads the day-0
// sample the harmonize step exports.
export async function loadSurface(source: Source, date: string): Promise<Surface> {
  if (source === "api") return json<Surface>(`${API_BASE}/surface?date=${date}`);
  const day = await json<{
    date: string;
    lat: number[];
    lon: number[];
    surface: Surface["fields"];
  }>("/data/sample_day.json");
  return { date: day.date, lat: day.lat, lon: day.lon, fields: day.surface };
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
  file: File,
  colormap: "grayscale" | "thermal" = "grayscale"
): Promise<Reconstruction> {
  const form = new FormData();
  form.append("file", file);
  form.append("date", date);
  form.append("model", model);
  form.append("colormap", colormap);
  const res = await fetch(`${API_BASE}/reconstruct/image`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`image reconstruct failed (${res.status})`);
  return res.json();
}

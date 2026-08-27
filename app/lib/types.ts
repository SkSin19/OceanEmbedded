// Shapes of the JSON the ML pipeline exports into public/data/.

export type Grid = (number | null)[][]; // [lat][lon], null = land

export interface Meta {
  region: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
    resolution: number;
  };
  depths: number[];
  lat: [number, number];
  lon: [number, number];
  shape: { lat: number; lon: number };
  times: string[];
}

export interface PerDepthMetric {
  depth: number;
  rmse: number;
  corr: number;
  bias: number;
}

export interface Metrics {
  overall: { rmse: number; corr: number; bias: number };
  per_depth: PerDepthMetric[];
}

export interface Prediction {
  date: string;
  depths: number[];
  lat: number[];
  lon: number[];
  predicted: Grid[]; // [depth][lat][lon]
  truth: Grid[]; // [depth][lat][lon]
}

export type ModelName = "cnn" | "baseline";

export type RGB = [number, number, number] | null;

export interface Embedding {
  date: string;
  lat: number[];
  lon: number[];
  rgb: RGB[][]; // [lat][lon]
}

export interface Reconstruction {
  date: string;
  model: ModelName;
  depths: number[];
  lat: number[];
  lon: number[];
  predicted: Grid[]; // modified reconstruction [depth][lat][lon]
  baseline: Grid[]; // unmodified reconstruction
  truth: Grid[];
  sst_input: Grid; // the SST field actually fed to the model [lat][lon]
}

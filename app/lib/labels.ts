import type { ModelName, SurfaceVar } from "./types";

export const MODEL_LABEL: Record<ModelName, string> = {
  cnn: "CNN embedding",
  baseline: "MLP baseline",
};

// The seven harmonized surface channels the model actually consumes.
export const SURFACE_LABEL: Record<SurfaceVar, string> = {
  sst: "SST",
  sss: "SSS",
  ssh: "SSH",
  u: "Current U",
  v: "Current V",
  uwnd: "Wind U",
  vwnd: "Wind V",
};

export const SURFACE_UNIT: Record<SurfaceVar, string> = {
  sst: "degC",
  sss: "PSU",
  ssh: "m",
  u: "m/s",
  v: "m/s",
  uwnd: "m/s",
  vwnd: "m/s",
};

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type * as LeafletNS from "leaflet";
import type {
  Grid,
  Meta,
  ModelName,
  Prediction,
  Profile,
  Surface,
  SurfaceVar,
} from "../lib/types";
import { loadProfile, loadSurface, type Source } from "../lib/api";
import { errorGrid, gridExtent } from "../lib/grid";
import { gridToSmoothCanvas, nearestIndex } from "../lib/raster";
import { SURFACE_LABEL, SURFACE_UNIT } from "../lib/labels";
import ProfileChart from "./ProfileChart";
import Legend from "./Legend";

type Layer =
  | { kind: "temp"; field: "predicted" | "truth" | "error" }
  | { kind: "surface"; field: SurfaceVar };

const TEMP_LAYERS: { id: "predicted" | "truth" | "error"; label: string }[] = [
  { id: "predicted", label: "Reconstruction" },
  { id: "truth", label: "GLORYS truth" },
  { id: "error", label: "Error" },
];

const SURFACE_LAYERS: SurfaceVar[] = ["sst", "sss", "ssh", "u", "v", "uwnd", "vwnd"];

/** Well-known basins in the study region, for quick jumps during a demo. */
const HOTSPOTS: { name: string; lat: number; lon: number }[] = [
  { name: "Bay of Bengal", lat: 15.0, lon: 88.0 },
  { name: "Arabian Sea", lat: 15.5, lon: 65.0 },
  { name: "Somali upwelling", lat: 8.0, lon: 52.0 },
  { name: "Andaman Sea", lat: 11.0, lon: 95.0 },
  { name: "Lakshadweep Sea", lat: 9.0, lon: 73.0 },
];

interface Props {
  source: Source;
  meta: Meta;
  models: ModelName[];
  pred: Prediction | null;
  date: string;
  model: ModelName;
  loading: boolean;
  /** Shared probe cell, so the 3D view cuts through the same column. */
  picked: { row: number; col: number } | null;
  onPick: (p: { row: number; col: number }) => void;
}

export default function OceanMap({
  source,
  meta,
  pred,
  date,
  model,
  loading,
  picked,
  onPick,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletNS.Map | null>(null);
  const overlayRef = useRef<LeafletNS.ImageOverlay | null>(null);
  const markerRef = useRef<LeafletNS.Marker | null>(null);
  const leafletRef = useRef<typeof LeafletNS | null>(null);

  const [ready, setReady] = useState(false);
  const [depthIdx, setDepthIdx] = useState(0);
  const [layer, setLayer] = useState<Layer>({ kind: "temp", field: "predicted" });
  const [opacity, setOpacity] = useState(0.82);
  const [loaded, setLoaded] = useState<{ key: string; profile: Profile | null } | null>(null);
  const [surface, setSurface] = useState<Surface | null>(null);

  // The shared probe cell, expressed as coordinates for the map.
  const pickedLatLon = useMemo(() => {
    if (!picked || !pred) return null;
    return { lat: pred.lat[picked.row], lon: pred.lon[picked.col] };
  }, [picked, pred]);

  // Real surface inputs for the active day; drives the surface layers.
  useEffect(() => {
    let active = true;
    loadSurface(source, date)
      .then((s) => active && setSurface(s))
      .catch(() => active && setSurface(null));
    return () => {
      active = false;
    };
  }, [source, date]);

  // The grid currently painted on the map, plus its colour scale.
  const active = useMemo(() => {
    if (layer.kind === "surface") {
      const g = surface?.fields[layer.field];
      if (!g || !surface) return null;
      const { min, max } = gridExtent(g);
      return {
        grid: g,
        lat: surface.lat,
        lon: surface.lon,
        mode: "thermal" as const,
        min,
        max,
        absMax: 1,
        unit: SURFACE_UNIT[layer.field],
        title: SURFACE_LABEL[layer.field],
      };
    }
    if (!pred) return null;
    const truth = pred.truth[depthIdx];
    const { min, max } = gridExtent(truth);
    if (layer.field === "error") {
      const g = errorGrid(pred.predicted[depthIdx], truth);
      const e = gridExtent(g);
      const absMax = Math.max(Math.abs(e.min), Math.abs(e.max), 0.1);
      return {
        grid: g,
        lat: pred.lat,
        lon: pred.lon,
        mode: "diverging" as const,
        min: -absMax,
        max: absMax,
        absMax,
        unit: "degC",
        title: `Error at ${meta.depths[depthIdx]} m`,
      };
    }
    const g = layer.field === "predicted" ? pred.predicted[depthIdx] : truth;
    return {
      grid: g,
      lat: pred.lat,
      lon: pred.lon,
      mode: "thermal" as const,
      min,
      max,
      absMax: 1,
      unit: "degC",
      title: `${layer.field === "predicted" ? "Reconstruction" : "GLORYS"} at ${meta.depths[depthIdx]} m`,
    };
  }, [layer, pred, depthIdx, surface, meta.depths]);

  // Picking snaps to the nearest model cell and publishes it upward; the
  // profile fetch below reacts to that, so a pick made in any tab lands here.
  const pick = useCallback(
    (lat: number, lon: number) => {
      if (!pred) return;
      onPick({
        row: nearestIndex(pred.lat, lat),
        col: nearestIndex(pred.lon, lon),
      });
    },
    [pred, onPick]
  );

  // The map click handler is installed once, so it reads the latest pick here.
  const pickRef = useRef(pick);
  useEffect(() => {
    pickRef.current = pick;
  }, [pick]);

  // Without a backend the profile is just a column of the volume we already
  // hold, so it is derived rather than fetched.
  const staticProfile = useMemo<Profile | null>(() => {
    if (!pred || !picked) return null;
    const { row, col } = picked;
    return {
      lat: pred.lat[row],
      lon: pred.lon[col],
      depths: pred.depths,
      predicted: pred.predicted.map((d) => d[row][col]),
      truth: pred.truth.map((d) => d[row][col]),
    };
  }, [pred, picked]);

  // Identity of the forward pass the panel currently wants.
  const requestKey = picked ? `${date}|${model}|${picked.row}|${picked.col}` : null;

  // With the backend, ask it for a real forward pass at the probe cell. The
  // result is tagged with its request key so "busy" is derived from a mismatch
  // rather than tracked as separate state.
  useEffect(() => {
    if (source !== "api" || !pred || !picked || !requestKey) return;
    const { row, col } = picked;
    let active = true;
    loadProfile(date, pred.lat[row], pred.lon[col], model)
      .then((p) => active && setLoaded({ key: requestKey, profile: p }))
      .catch(() => active && setLoaded({ key: requestKey, profile: null }));
    return () => {
      active = false;
    };
  }, [source, pred, picked, date, model, requestKey]);

  // Keep the previous profile on screen while the next one is in flight.
  const shownProfile = source === "api" ? loaded?.profile ?? null : staticProfile;
  const profileBusy = source === "api" && requestKey !== null && loaded?.key !== requestKey;

  // Build the map once, after Leaflet is dynamically imported (it touches
  // `window` at module scope, so it cannot be a top-level import).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const L = (await import("leaflet")).default;
      if (cancelled || !containerRef.current || mapRef.current) return;
      leafletRef.current = L;

      const r = meta.region;
      const map = L.map(containerRef.current, {
        center: [(r.min_lat + r.max_lat) / 2, (r.min_lon + r.max_lon) / 2],
        zoom: 4,
        minZoom: 3,
        maxZoom: 9,
        zoomControl: false,
        attributionControl: true,
      });

      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution:
          '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap contributors',
        subdomains: "abcd",
        maxZoom: 19,
      }).addTo(map);

      // Coastlines on top of the data overlay so the basins stay readable.
      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png",
        { subdomains: "abcd", maxZoom: 19, pane: "shadowPane", opacity: 0.85 }
      ).addTo(map);

      L.control.zoom({ position: "topright" }).addTo(map);

      // The study domain, drawn exactly as configured in config.yaml.
      L.rectangle(
        [
          [r.min_lat, r.min_lon],
          [r.max_lat, r.max_lon],
        ],
        { color: "#22d3ee", weight: 1, fillOpacity: 0, dashArray: "4 4" }
      ).addTo(map);

      map.fitBounds([
        [r.min_lat, r.min_lon],
        [r.max_lat, r.max_lon],
      ]);

      map.on("click", (e: LeafletNS.LeafletMouseEvent) => {
        const { lat, lng } = e.latlng;
        if (lat < r.min_lat || lat > r.max_lat || lng < r.min_lon || lng > r.max_lon) return;
        pickRef.current(lat, lng);
      });

      mapRef.current = map;
      setReady(true);
    })();

    return () => {
      cancelled = true;
      mapRef.current?.remove();
      mapRef.current = null;
      overlayRef.current = null;
      markerRef.current = null;
    };
    // Built once; later state is pushed in through the effects below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Repaint the data overlay whenever the active field changes.
  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L || !ready || !active) return;

    const { grid, lat, lon } = active;
    // Grid values are cell centres, so the image spans half a cell further out.
    const dLat = (lat[1] - lat[0]) / 2;
    const dLon = (lon[1] - lon[0]) / 2;
    const bounds: LeafletNS.LatLngBoundsExpression = [
      [lat[0] - dLat, lon[0] - dLon],
      [lat[lat.length - 1] + dLat, lon[lon.length - 1] + dLon],
    ];

    const url = gridToSmoothCanvas(grid as Grid, {
      mode: active.mode,
      min: active.min,
      max: active.max,
      absMax: active.absMax,
      landAlpha: 0,
      scale: 6,
    }).toDataURL();

    if (overlayRef.current) map.removeLayer(overlayRef.current);
    overlayRef.current = L.imageOverlay(url, bounds, {
      opacity,
      interactive: false,
      className: "ocean-overlay",
    }).addTo(map);
  }, [active, ready, opacity]);

  useEffect(() => {
    overlayRef.current?.setOpacity(opacity);
  }, [opacity]);

  // Keep the probe marker in sync, wherever the pick came from.
  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L || !ready || !pickedLatLon) return;

    if (!markerRef.current) {
      const icon = L.divIcon({
        className: "",
        html: `<span class="relative flex h-5 w-5 -translate-x-1/2 -translate-y-1/2 items-center justify-center">
                 <span class="absolute h-5 w-5 animate-ping rounded-full bg-amber-400/40"></span>
                 <span class="relative h-2.5 w-2.5 rounded-full bg-amber-400 ring-2 ring-slate-950"></span>
               </span>`,
        iconSize: [0, 0],
      });
      markerRef.current = L.marker([pickedLatLon.lat, pickedLatLon.lon], {
        icon,
        keyboard: false,
      }).addTo(map);
    } else {
      markerRef.current.setLatLng([pickedLatLon.lat, pickedLatLon.lon]);
    }
  }, [pickedLatLon, ready]);

  function flyTo(lat: number, lon: number) {
    mapRef.current?.flyTo([lat, lon], 6, { duration: 1.1 });
    pick(lat, lon);
  }

  const surfaceAvailable = SURFACE_LAYERS.filter((v) => surface?.fields[v]);
  const isTemp = layer.kind === "temp";

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="animate-fade-up mb-6 max-w-2xl">
        <div className="mb-2 inline-flex items-center gap-1.5 rounded-full bg-sky-500/10 px-2.5 py-1 text-[11px] font-medium tracking-wide text-sky-300 ring-1 ring-sky-500/20">
          Georeferenced map
        </div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50 sm:text-3xl">
          The reconstruction, on the real ocean
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          Every model field projected onto its true coordinates. Click anywhere
          in the domain to run the reconstruction at that point and read the
          vertical profile the network predicts against GLORYS.
        </p>
      </div>

      <div className="animate-fade-up grid grid-cols-1 gap-6 lg:grid-cols-3" style={{ animationDelay: "80ms" }}>
        <section className="glass-panel relative overflow-hidden rounded-2xl p-0 lg:col-span-2">
          <div
            ref={containerRef}
            className="h-[560px] w-full bg-slate-950"
            style={{ cursor: "crosshair" }}
          />

          {/* Layer switcher */}
          <div className="pointer-events-auto absolute left-3 top-3 z-[500] w-44 rounded-xl border border-white/10 bg-slate-950/85 p-2 backdrop-blur-md">
            <div className="px-1 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              Temperature
            </div>
            <div className="space-y-0.5">
              {TEMP_LAYERS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setLayer({ kind: "temp", field: t.id })}
                  className={`flex w-full items-center justify-between rounded-md px-2 py-1 text-xs transition ${
                    isTemp && layer.field === t.id
                      ? "bg-cyan-500/15 font-medium text-cyan-300"
                      : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                  }`}
                >
                  {t.label}
                  {isTemp && layer.field === t.id && <Dot />}
                </button>
              ))}
            </div>

            {surfaceAvailable.length > 0 && (
              <>
                <div className="mt-2 px-1 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Surface inputs
                </div>
                <div className="space-y-0.5">
                  {surfaceAvailable.map((v) => (
                    <button
                      key={v}
                      onClick={() => setLayer({ kind: "surface", field: v })}
                      className={`flex w-full items-center justify-between rounded-md px-2 py-1 text-xs transition ${
                        layer.kind === "surface" && layer.field === v
                          ? "bg-cyan-500/15 font-medium text-cyan-300"
                          : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                      }`}
                    >
                      {SURFACE_LABEL[v]}
                      {layer.kind === "surface" && layer.field === v && <Dot />}
                    </button>
                  ))}
                </div>
              </>
            )}

            <div className="mt-2 border-t border-white/10 pt-2">
              <label className="flex items-center justify-between text-[10px] text-slate-400">
                <span>Opacity</span>
                <span className="font-mono">{Math.round(opacity * 100)}%</span>
              </label>
              <input
                type="range"
                min={0.2}
                max={1}
                step={0.02}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="mt-1 w-full accent-cyan-400"
              />
            </div>
          </div>

          {/* Scale bar */}
          {active && (
            <div className="pointer-events-none absolute bottom-3 left-3 z-[500] w-52 rounded-xl border border-white/10 bg-slate-950/85 p-2.5 backdrop-blur-md">
              <div className="mb-1.5 flex items-baseline justify-between">
                <span className="text-[11px] font-medium text-slate-200">{active.title}</span>
                <span className="text-[10px] text-slate-500">{active.unit}</span>
              </div>
              <Legend
                mode={active.mode}
                min={active.min}
                max={active.max}
                absMax={active.absMax}
              />
            </div>
          )}

          {loading && (
            <div className="pointer-events-none absolute right-3 top-3 z-[500] flex items-center gap-1.5 rounded-full border border-white/10 bg-slate-950/85 px-2.5 py-1 text-[11px] text-amber-300 backdrop-blur-md">
              <span className="h-2.5 w-2.5 animate-spin rounded-full border-[1.5px] border-amber-900 border-t-amber-400" />
              Reconstructing
            </div>
          )}
        </section>

        <section className="flex flex-col gap-4">
          <div className="glass-panel rounded-2xl p-4">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-medium text-slate-200">Profile at point</h2>
              {profileBusy && (
                <span className="h-3 w-3 animate-spin rounded-full border-[1.5px] border-slate-600 border-t-amber-400" />
              )}
            </div>
            {pickedLatLon ? (
              <div className="mb-2 flex flex-wrap gap-1.5">
                <Chip>
                  {pickedLatLon.lat.toFixed(2)}&deg; N, {pickedLatLon.lon.toFixed(2)}&deg; E
                </Chip>
                {shownProfile?.predicted[0] != null && (
                  <Chip accent>Surface {shownProfile.predicted[0].toFixed(2)} degC</Chip>
                )}
              </div>
            ) : (
              <p className="mb-2 text-xs text-slate-500">Click the map to pick a point.</p>
            )}
            <ProfileChart
              depths={shownProfile?.depths ?? meta.depths}
              predicted={shownProfile?.predicted ?? []}
              truth={shownProfile?.truth ?? []}
            />
            <div className="mt-2 flex gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 bg-sky-400" /> Truth (GLORYS)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 bg-amber-500" /> Predicted
              </span>
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-4">
            <label className="flex items-center justify-between text-sm text-slate-300">
              <span>Map depth</span>
              <span className="font-mono text-amber-400">{meta.depths[depthIdx]} m</span>
            </label>
            <input
              type="range"
              min={0}
              max={meta.depths.length - 1}
              value={depthIdx}
              onChange={(e) => setDepthIdx(Number(e.target.value))}
              disabled={layer.kind === "surface"}
              className="mt-2 w-full accent-amber-500 disabled:opacity-40"
            />
            {layer.kind === "surface" && (
              <p className="mt-1.5 text-[11px] text-slate-500">
                Surface layers are 2-D; switch to a temperature layer to sweep depth.
              </p>
            )}

            <div className="mt-4 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              Jump to basin
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {HOTSPOTS.map((h) => (
                <button
                  key={h.name}
                  onClick={() => flyTo(h.lat, h.lon)}
                  className="rounded-md bg-white/5 px-2 py-1 text-[11px] text-slate-300 ring-1 ring-white/10 transition hover:bg-white/10 hover:text-white"
                >
                  {h.name}
                </button>
              ))}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function Dot() {
  return <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />;
}

function Chip({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <span
      className={`rounded-md px-1.5 py-0.5 font-mono text-[11px] ${
        accent ? "bg-amber-500/10 text-amber-300" : "bg-slate-800/80 text-slate-400"
      }`}
    >
      {children}
    </span>
  );
}

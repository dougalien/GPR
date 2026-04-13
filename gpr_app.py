from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


@dataclass
class GPRData:
    file_type: str
    source_file: str
    traces: Optional[np.ndarray] = None
    time_axis: Optional[np.ndarray] = None
    distance_axis: Optional[np.ndarray] = None
    gps: Optional[Dict[str, Any]] = None
    marks: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_bytes: Optional[bytes] = None


@dataclass
class FileInspection:
    source_file: str
    extension: str
    size_bytes: int
    is_binary: bool
    likely_type: str
    notes: str
    preview_text: Optional[str] = None


@dataclass
class RadanProject:
    stem: str
    dzt: Optional[GPRData] = None
    dzg: Optional[GPRData] = None
    dzx: Optional[GPRData] = None
    dza: Optional[GPRData] = None
    other_files: Dict[str, FileInspection] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)


def ollama_available(base_url: str = "http://127.0.0.1:11434") -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        return resp.ok
    except Exception:
        return False


def call_ollama(prompt: str, model: str, base_url: str = "http://127.0.0.1:11434") -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    resp = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return str(resp.json().get("response", "")).strip()


def openai_available() -> bool:
    try:
        return bool(st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""))
    except Exception:
        return bool(os.getenv("OPENAI_API_KEY", ""))


def call_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful GPR teaching assistant. Stay cautious and avoid overclaiming burial detection."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def inspect_uploaded_file(name: str, content: bytes) -> FileInspection:
    ext = Path(name).suffix.lower()
    size = len(content)
    chunk = content[:512]
    nontext = sum(b < 9 or (13 < b < 32) or b > 126 for b in chunk)
    is_binary = nontext > len(chunk) * 0.2 if chunk else False

    preview_text = None
    likely_type = "unknown"
    notes = "Could not confidently identify file type."

    if not is_binary:
        try:
            preview_text = chunk.decode("utf-8", errors="ignore")
        except Exception:
            preview_text = None
        if preview_text:
            if "$GPGGA" in preview_text or "$GNGGA" in preview_text:
                likely_type = "gps_text"
                notes = "Looks like NMEA GPS text."
            elif "," in preview_text:
                likely_type = "csv_or_delimited_text"
                notes = "Looks like delimited text export."
            else:
                likely_type = "plain_text"
                notes = "Looks like plain text."
    else:
        if ext == ".dzt":
            likely_type = "gssi_dzt"
            notes = "Binary file with .dzt extension; likely GSSI radar data."
        elif ext == ".tmf":
            likely_type = "gssi_tmf"
            notes = "Binary file with .tmf extension; likely timing/mark file."
        elif ext == ".dza":
            likely_type = "gssi_dza"
            notes = "Binary file with .dza extension; likely auxiliary line trace data."
        else:
            likely_type = "binary_unknown"
            notes = "Binary file but format not yet supported."
    return FileInspection(name, ext, size, is_binary, likely_type, notes, preview_text)


def parse_dzt(name: str, content: bytes) -> GPRData:
    if len(content) < 1024:
        raise ValueError("File too small to contain a valid 1024-byte DZT header.")
    header = content[:1024]
    data = content[1024:]

    rh_nsamp = struct.unpack_from("<h", header, 4)[0]
    rh_bits = struct.unpack_from("<h", header, 6)[0]
    rh_zero = struct.unpack_from("<h", header, 8)[0]
    rhf_sps = struct.unpack_from("<f", header, 80)[0]
    rhf_spm = struct.unpack_from("<f", header, 84)[0]
    rhf_position = struct.unpack_from("<f", header, 88)[0]
    rhf_range = struct.unpack_from("<f", header, 92)[0]
    try:
        rh_nchan = struct.unpack_from("<h", header, 54)[0]
    except Exception:
        rh_nchan = 1

    if rh_nsamp <= 0:
        raise ValueError(f"Invalid samples/trace value: {rh_nsamp}")
    if rh_bits == 8:
        dtype = np.uint8
        bytes_per_sample = 1
    elif rh_bits == 16:
        dtype = np.uint16
        bytes_per_sample = 2
    else:
        raise ValueError(f"Unsupported DZT bit depth: {rh_bits}")

    trace_size = rh_nsamp * bytes_per_sample
    n_traces = len(data) // trace_size
    if trace_size <= 0 or n_traces == 0:
        raise ValueError("No usable trace data found after header.")
    usable = n_traces * trace_size
    raw = np.frombuffer(data[:usable], dtype=dtype).reshape(n_traces, rh_nsamp)
    centered = raw.astype(np.int32) - rh_zero
    time_axis = np.linspace(0, rhf_range, rh_nsamp) if rhf_range and rhf_range > 0 else None
    distance_axis = np.arange(n_traces) / rhf_spm if rhf_spm and rhf_spm > 0 else None

    metadata = {
        "samples_per_trace": int(rh_nsamp),
        "bits_per_sample": int(rh_bits),
        "zero_offset": int(rh_zero),
        "scans_per_second": float(rhf_sps),
        "scans_per_meter": float(rhf_spm),
        "position_ns": float(rhf_position),
        "range_ns": float(rhf_range),
        "channels": int(rh_nchan),
        "header_size_bytes": 1024,
        "n_traces": int(n_traces),
    }
    return GPRData("dzt", name, centered, time_axis, distance_axis, metadata=metadata)


def parse_csv(name: str, content: bytes) -> GPRData:
    df = pd.read_csv(io.BytesIO(content))
    numeric_df = df.select_dtypes(include=[np.number])
    traces = numeric_df.to_numpy() if not numeric_df.empty else None
    return GPRData("csv", name, traces=traces, metadata={"columns": list(df.columns), "n_rows": int(len(df))})


def parse_text(name: str, content: bytes, forced_type: str = "text") -> GPRData:
    text = content.decode("utf-8", errors="ignore")
    gps = {"note": "NMEA GPS strings detected"} if ("$GPGGA" in text or "$GNGGA" in text) else None
    traces = None
    delimiter_used = None
    for delimiter in [",", "\t", None]:
        try:
            if delimiter is None:
                df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
                delimiter_used = "whitespace"
            else:
                df = pd.read_csv(io.StringIO(text), sep=delimiter, header=None)
                delimiter_used = repr(delimiter)
            if not df.empty:
                numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
                if not numeric_df.empty:
                    traces = numeric_df.to_numpy()
                    break
        except Exception:
            continue
    return GPRData(forced_type, name, traces=traces, gps=gps, metadata={"preview": text[:300], "parsed_delimiter": delimiter_used}, raw_bytes=content)


def parse_dza(name: str, content: bytes) -> GPRData:
    inspection = inspect_uploaded_file(name, content)
    if inspection.is_binary:
        return GPRData("dza", name, metadata={"note": "Binary auxiliary file detected; parsing not implemented yet.", "size_bytes": len(content)}, raw_bytes=content)
    return parse_text(name, content, forced_type="dza")


def load_uploaded_file(uploaded_file) -> Tuple[Optional[GPRData], Optional[FileInspection]]:
    name = uploaded_file.name
    content = uploaded_file.getvalue()
    ext = Path(name).suffix.lower()
    try:
        if ext == ".dzt":
            return parse_dzt(name, content), None
        if ext == ".csv":
            return parse_csv(name, content), None
        if ext in {".txt", ".asc"}:
            return parse_text(name, content), None
        if ext == ".dzg":
            return parse_text(name, content, forced_type="dzg"), inspect_uploaded_file(name, content)
        if ext == ".dzx":
            return parse_text(name, content, forced_type="dzx"), inspect_uploaded_file(name, content)
        if ext == ".dza":
            return parse_dza(name, content), inspect_uploaded_file(name, content)
        return None, inspect_uploaded_file(name, content)
    except Exception as e:
        raise ValueError(f"Could not load file '{name}': {e}") from e


def make_hyperbola(n_traces: int, apex_trace: int, apex_sample: int, scale: float, amplitude: float, width: int = 2) -> np.ndarray:
    out = np.zeros((n_traces, 1), dtype=float)
    # placeholder; injected directly by caller
    return out


def build_synthetic_demo_gpr() -> GPRData:
    n_traces = 180
    n_samples = 320
    t = np.linspace(0, 120, n_samples)
    traces = np.random.normal(0, 12, size=(n_traces, n_samples))
    for i in range(n_traces):
        idx = 18 + int(1.5 * np.sin(i / 15.0))
        if 2 <= idx < n_samples - 2:
            traces[i, idx - 2:idx + 2] += 90
    for i in range(n_traces):
        idx = 90
        traces[i, max(0, idx - 1):min(n_samples, idx + 2)] += 70
    for i in range(n_traces):
        idx = int(130 + 0.35 * i)
        if 1 <= idx < n_samples - 1:
            traces[i, idx - 1:idx + 2] += 95
    for center, apex, scale, amp in [(55, 65, 0.055, 160), (130, 110, 0.035, 135)]:
        for i in range(n_traces):
            idx = int(apex + scale * (i - center) ** 2)
            if 1 <= idx < n_samples - 1:
                traces[i, idx - 1:idx + 2] += amp
    attenuation = np.linspace(1.0, 0.45, n_samples)
    traces = traces * attenuation[np.newaxis, :]
    return GPRData(
        "demo", "demo_line.dzt", traces=traces, time_axis=t, distance_axis=np.linspace(0, 18, n_traces),
        metadata={"demo_mode": True, "demo_type": "generic", "description": "Synthetic GPR demo with reflectors and hyperbolic targets.", "n_traces": n_traces, "samples_per_trace": n_samples},
    )


def build_graveyard_demo_gpr() -> GPRData:
    rng = np.random.default_rng(14)
    n_traces = 260
    n_samples = 380
    distance = np.linspace(0, 26, n_traces)
    time_ns = np.linspace(0, 150, n_samples)

    # Base field-like texture: banded, cluttered, and a little messy.
    traces = rng.normal(0, 4.5, size=(n_traces, n_samples))

    # Direct wave / shallow ringing with slight lateral wobble.
    for i in range(n_traces):
        idx = 18 + int(1.4 * np.sin(i / 17.0))
        amp = 48 + 8 * np.sin(i / 31.0)
        traces[i, max(0, idx - 1):min(n_samples, idx + 2)] += amp
        traces[i, max(0, idx + 6):min(n_samples, idx + 9)] -= 0.35 * amp

    # Add weak horizontal banding and shallow clutter to mimic real student data.
    for base_idx, amp, wobble in [(72, 16, 0.7), (104, 12, 1.0), (176, 14, 1.5), (228, 10, 1.2)]:
        for i in range(n_traces):
            idx = int(base_idx + wobble * np.sin(i / 20.0) + 0.9 * np.sin(i / 7.5))
            if 2 <= idx < n_samples - 2:
                traces[i, idx - 1:idx + 2] += amp
                traces[i, idx + 3:idx + 5] -= 0.45 * amp

    # Slight trace-to-trace gain drift to avoid a too-clean look.
    drift = np.linspace(0.92, 1.08, n_traces)[:, None]
    traces *= drift

    grave_centers = [48, 96, 146, 194]
    grave_notes = []

    # Disturbed grave shafts: weaker coherence, local attenuation, broken reflectors.
    for j, c in enumerate(grave_centers):
        half_width = 9 + (j % 2)
        top = 54 + 3 * j
        base = 148 + 8 * (j % 3)
        for i in range(max(0, c - half_width), min(n_traces, c + half_width + 1)):
            frac = abs(i - c) / max(half_width, 1)
            shaft_top = int(top + 2 * frac)
            shaft_base = int(base + 10 * frac)

            # Disturb and attenuate the shaft, not too strongly.
            traces[i, shaft_top:shaft_base] *= (0.82 - 0.10 * (1 - frac))
            traces[i, shaft_top:shaft_base] += rng.normal(0, 5.5, size=shaft_base - shaft_top)

            # Break the shallow reflector inside the grave area.
            broken_idx = int(72 + 2 * np.sin(i / 8.0) + 1.5 * frac)
            if 2 <= broken_idx < n_samples - 2:
                traces[i, broken_idx - 1:broken_idx + 2] *= 0.35

            # Very subtle downwarp beneath some grave shafts.
            deep_idx = int(176 + 7 * np.exp(-2.6 * frac))
            if 2 <= deep_idx < n_samples - 2:
                traces[i, deep_idx - 1:deep_idx + 2] += 7.0

        grave_notes.append({
            "feature": "grave_shaft_like_zone",
            "trace_center": c,
            "distance_m": round(float(distance[c]), 2),
            "signature": "disturbed zone with broken shallow reflector and mild attenuation",
        })

    # Only two faint coffin-like hyperbolas. Not every grave gets one.
    subtle_hyperbolas = [
        {"center": 49, "apex": 96, "scale": 0.050, "amp": 34},
        {"center": 147, "apex": 102, "scale": 0.043, "amp": 28},
    ]
    for h in subtle_hyperbolas:
        for i in range(max(0, h["center"] - 34), min(n_traces, h["center"] + 35)):
            idx = int(h["apex"] + h["scale"] * (i - h["center"]) ** 2)
            if 2 <= idx < n_samples - 2:
                traces[i, idx - 1:idx + 2] += h["amp"]
                traces[i, idx + 3:idx + 5] -= 0.30 * h["amp"]

    # Ambiguous non-grave disturbances.
    ambiguous_zones = [(118, 133, 84, 142), (214, 228, 92, 160)]
    for left, right, top, base in ambiguous_zones:
        for i in range(left, right):
            traces[i, top:base] *= 0.88
            traces[i, top:base] += rng.normal(0, 6.0, size=base - top)

    # Add a few weak point targets / roots / stones.
    for center, apex, scale, amp in [(30, 88, 0.060, 18), (171, 120, 0.038, 22), (232, 108, 0.055, 16)]:
        for i in range(max(0, center - 20), min(n_traces, center + 21)):
            idx = int(apex + scale * (i - center) ** 2)
            if 2 <= idx < n_samples - 2:
                traces[i, idx - 1:idx + 2] += amp

    # Depth attenuation and mild vertical smoothing for a less cartoon-like appearance.
    attenuation = np.linspace(1.0, 0.42, n_samples)
    traces *= attenuation[np.newaxis, :]
    kernel = np.array([0.22, 0.56, 0.22])
    for i in range(n_traces):
        traces[i] = np.convolve(traces[i], kernel, mode="same")

    metadata = {
        "demo_mode": True,
        "demo_type": "graveyard_training_field_style",
        "description": "Synthetic cemetery training line rebuilt to resemble messier student field data: weak grave-shaft disturbances, broken reflectors, attenuation changes, subtle hyperbolas, and ambiguous clutter.",
        "n_traces": n_traces,
        "samples_per_trace": n_samples,
        "teaching_objective": "Students should look for disturbed zones and reflector disruption first, not textbook coffin hyperbolas.",
        "grave_like_targets": len(grave_centers),
        "hyperbola_like_targets": len(subtle_hyperbolas),
        "ambiguous_zones": len(ambiguous_zones),
        "notes": "Synthetic educational data only; intentionally field-like and ambiguous.",
        "feature_preview": grave_notes,
        "recommended_view": "Try grayscale, modest clipping, and processed radargram view.",
    }
    return GPRData(
        "demo", "graveyard_demo_line.dzt", traces=traces, time_axis=time_ns, distance_axis=distance, metadata=metadata
    )


def build_projects_from_uploads(uploaded_files, demo_mode: str = "none") -> Dict[str, RadanProject]:
    projects: Dict[str, RadanProject] = {}
    if demo_mode in {"generic", "graveyard"}:
        stem = "demo_line" if demo_mode == "generic" else "graveyard_demo"
        dzt = build_synthetic_demo_gpr() if demo_mode == "generic" else build_graveyard_demo_gpr()
        project = RadanProject(stem=stem)
        project.dzt = dzt
        project.dzg = GPRData("dzg", f"{stem}.dzg", gps={"track_preview": [{"trace": 0, "lat": 42.52, "lon": -70.89}, {"trace": int(dzt.metadata['n_traces']) - 1, "lat": 42.5215, "lon": -70.8885}]}, metadata={"demo_mode": True})
        project.dzx = GPRData("dzx", f"{stem}.dzx", metadata={"demo_mode": True, "gain": "auto", "time_zero_shift": 0})
        project.dza = GPRData("dza", f"{stem}.dza", metadata={"demo_mode": True, "description": "Synthetic companion file."})
        projects[stem] = project

    for uploaded_file in uploaded_files:
        stem = Path(uploaded_file.name).stem
        project = projects.setdefault(stem, RadanProject(stem=stem))
        parsed, inspection = load_uploaded_file(uploaded_file)
        ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
        if ext in {"dzt", "dzg", "dzx", "dza"} and parsed is not None:
            setattr(project, ext, parsed)
            project.files[ext] = uploaded_file.name
        else:
            if inspection is None:
                inspection = inspect_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
            project.other_files[uploaded_file.name] = inspection
    return projects


def dewow(traces: np.ndarray, window: int = 21) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    out = np.zeros_like(traces, dtype=float)
    for i in range(traces.shape[0]):
        trend = np.convolve(traces[i], kernel, mode="same")
        out[i] = traces[i] - trend
    return out


def background_remove(traces: np.ndarray) -> np.ndarray:
    background = np.mean(traces, axis=0, keepdims=True)
    return traces - background


def apply_gain(traces: np.ndarray, mode: str = "None", strength: float = 1.5) -> np.ndarray:
    if mode == "None":
        return traces.copy()
    n_samples = traces.shape[1]
    idx = np.linspace(0, 1, n_samples)
    if mode == "Linear":
        gain = 1.0 + strength * idx
    else:
        gain = np.exp(strength * idx)
    return traces * gain[np.newaxis, :]


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    out = traces.astype(float).copy()
    scale = np.max(np.abs(out), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return out / scale


def process_gpr_data(gpr_data: GPRData, time_zero_shift: int = 0, dewow_on: bool = False, dewow_window: int = 21,
                     background_on: bool = False, normalize_on: bool = False, gain_mode: str = "None", gain_strength: float = 1.5,
                     trace_crop: Optional[Tuple[int, int]] = None, sample_crop: Optional[Tuple[int, int]] = None) -> tuple[GPRData, list[str]]:
    traces = np.asarray(gpr_data.traces, dtype=float).copy()
    notes: list[str] = []
    if time_zero_shift != 0:
        traces = np.roll(traces, -int(time_zero_shift), axis=1)
        notes.append(f"Time-zero shift: {time_zero_shift} samples")
    if dewow_on:
        traces = dewow(traces, dewow_window)
        notes.append(f"Dewow applied (window {dewow_window})")
    if background_on:
        traces = background_remove(traces)
        notes.append("Background removal applied")
    if normalize_on:
        traces = normalize_traces(traces)
        notes.append("Per-trace normalization applied")
    if gain_mode != "None":
        traces = apply_gain(traces, gain_mode, gain_strength)
        notes.append(f"{gain_mode} gain applied (strength {gain_strength:.2f})")

    t0, t1 = 0, traces.shape[0] - 1
    s0, s1 = 0, traces.shape[1] - 1
    if trace_crop is not None:
        t0, t1 = max(0, trace_crop[0]), min(traces.shape[0] - 1, trace_crop[1])
    if sample_crop is not None:
        s0, s1 = max(0, sample_crop[0]), min(traces.shape[1] - 1, sample_crop[1])
    traces = traces[t0:t1 + 1, s0:s1 + 1]
    if trace_crop is not None or sample_crop is not None:
        notes.append(f"Crop applied: traces {t0}-{t1}, samples {s0}-{s1}")

    time_axis = gpr_data.time_axis[s0:s1 + 1] if gpr_data.time_axis is not None else None
    dist_axis = gpr_data.distance_axis[t0:t1 + 1] if gpr_data.distance_axis is not None else None
    meta = dict(gpr_data.metadata)
    meta["processed"] = True
    return GPRData(gpr_data.file_type, gpr_data.source_file, traces, time_axis, dist_axis, gpr_data.gps, gpr_data.marks, meta, gpr_data.raw_bytes), notes


def make_radargram_figure(gpr_data: GPRData, use_distance: bool, clip_low: float, clip_high: float, invert_y: bool, cmap: str):
    data = np.asarray(gpr_data.traces)
    n_traces, n_samples = data.shape
    if use_distance and gpr_data.distance_axis is not None and len(gpr_data.distance_axis) == n_traces:
        x = gpr_data.distance_axis
        xlabel = "Distance (m)"
    else:
        x = np.arange(n_traces)
        xlabel = "Trace Number"
    if gpr_data.time_axis is not None and len(gpr_data.time_axis) == n_samples:
        y = gpr_data.time_axis
        ylabel = "Time (ns)"
    else:
        y = np.arange(n_samples)
        ylabel = "Sample"
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    extent = [x[0], x[-1], y[-1], y[0]] if invert_y else [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(data.T, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Radargram: {gpr_data.source_file}")
    fig.tight_layout()
    return fig


def make_trace_figure(raw: GPRData, processed: Optional[GPRData], trace_index: int):
    raw_data = np.asarray(raw.traces)
    n_traces, n_samples = raw_data.shape
    trace_index = max(0, min(trace_index, n_traces - 1))
    raw_trace = raw_data[trace_index, :]
    y = raw.time_axis if raw.time_axis is not None and len(raw.time_axis) == n_samples else np.arange(n_samples)
    ylabel = "Time (ns)" if raw.time_axis is not None else "Sample"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(raw_trace, y, label="Raw", alpha=0.8)
    if processed is not None and processed.traces is not None:
        proc_idx = min(trace_index, processed.traces.shape[0] - 1)
        proc_y = processed.time_axis if processed.time_axis is not None and len(processed.time_axis) == processed.traces.shape[1] else np.arange(processed.traces.shape[1])
        ax.plot(processed.traces[proc_idx, :], proc_y, label="Processed", alpha=0.8)
    ax.set_title(f"Trace {trace_index}")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    return fig


def traces_to_csv_bytes(gpr_data: GPRData) -> bytes:
    return pd.DataFrame(gpr_data.traces).to_csv(index=False).encode("utf-8")


def build_ai_context(project: RadanProject, raw: GPRData, processed: Optional[GPRData], notes: list[str]) -> str:
    lines = [
        f"Project: {project.stem}",
        f"Source file: {raw.source_file}",
        f"Trace matrix shape: {raw.traces.shape if raw.traces is not None else 'none'}",
        f"Metadata: {raw.metadata}",
    ]
    if notes:
        lines.append("Processing notes: " + "; ".join(notes))
    if processed is not None and processed.traces is not None:
        lines.append(f"Processed trace matrix shape: {processed.traces.shape}")
    return "\n".join(lines)


def generate_ai_summary(project: RadanProject, raw: GPRData, processed: Optional[GPRData], notes: list[str], question: str,
                        prefer_local: bool, ollama_model: str, ollama_base_url: str, openai_model: str) -> tuple[str, str]:
    prompt = (
        "Use the following GPR analysis context to answer the user. Be cautious. Do not claim a grave is confirmed. "
        "Frame burial interpretations as first-pass educational possibilities only.\n\n"
        + build_ai_context(project, raw, processed, notes)
        + "\n\nUser request: " + question
    )
    if prefer_local and ollama_available(ollama_base_url):
        try:
            return call_ollama(prompt, ollama_model, ollama_base_url), f"Ollama ({ollama_model})"
        except Exception:
            pass
    if openai_available():
        try:
            return call_openai(prompt, openai_model), f"OpenAI ({openai_model})"
        except Exception:
            pass
    if not prefer_local and ollama_available(ollama_base_url):
        try:
            return call_ollama(prompt, ollama_model, ollama_base_url), f"Ollama ({ollama_model})"
        except Exception:
            pass
    raise RuntimeError("No AI provider is available. Core analysis still works without AI.")


def init_page():
    st.set_page_config(page_title="GPR Control Panel", layout="wide")
    st.title("GPR Control Panel")
    st.caption("Single-file Streamlit app for grouped RADAN project files, now with a cemetery training demo.")


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Display Controls")
    st.sidebar.markdown("**Processing**")
    controls = {
        "clip_low": st.sidebar.slider("Lower contrast clip percentile", 0.0, 20.0, 2.0, 0.5),
        "clip_high": st.sidebar.slider("Upper contrast clip percentile", 80.0, 100.0, 98.0, 0.5),
        "use_distance": st.sidebar.checkbox("Use distance axis if available", value=True),
        "invert_y": st.sidebar.checkbox("Invert vertical axis", value=True),
        "show_trace": st.sidebar.checkbox("Show single trace viewer", value=True),
        "cmap": st.sidebar.selectbox("Colormap", ["gray", "viridis", "plasma", "seismic"], index=0),
        "time_zero_shift": st.sidebar.slider("Time-zero shift (samples)", -40, 40, 0),
        "dewow_on": st.sidebar.checkbox("Apply dewow", value=False),
        "dewow_window": st.sidebar.slider("Dewow window", 3, 61, 21, 2),
        "background_on": st.sidebar.checkbox("Apply background removal", value=False),
        "normalize_on": st.sidebar.checkbox("Normalize traces", value=False),
        "gain_mode": st.sidebar.selectbox("Gain", ["None", "Linear", "Exponential"], index=0),
        "gain_strength": st.sidebar.slider("Gain strength", 0.1, 3.0, 1.4, 0.1),
        "ai_enable": st.sidebar.checkbox("Enable AI help", value=False),
        "ai_prefer_local": st.sidebar.checkbox("Prefer local Ollama", value=True),
        "ollama_model": st.sidebar.text_input("Ollama model", value="llama3.1:8b"),
        "ollama_base_url": st.sidebar.text_input("Ollama URL", value="http://127.0.0.1:11434"),
        "openai_model": st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini"),
    }
    return controls


def render_file_loader():
    st.subheader("1. Load RADAN Project Files")
    demo_mode = st.selectbox("Demo mode", ["None", "Generic synthetic line", "Graveyard field-style training line"], index=0)
    uploaded = st.file_uploader(
        "Upload one or more related files",
        type=["dzt", "csv", "txt", "asc", "dzg", "dzx", "dza"],
        accept_multiple_files=True,
        help="Upload matching files like line01.dzt, line01.dzg, line01.dzx, and line01.dza together.",
    )
    mode_map = {"None": "none", "Generic synthetic line": "generic", "Graveyard field-style training line": "graveyard"}
    return mode_map[demo_mode], uploaded


def render_project_inventory(projects: Dict[str, RadanProject]) -> str:
    rows = []
    stems = sorted(projects.keys())
    for stem in stems:
        p = projects[stem]
        rows.append({"Project": stem, "DZT": "✓" if p.dzt is not None else "✗", "DZG": "✓" if p.dzg is not None else "✗", "DZX": "✓" if p.dzx is not None else "✗", "DZA": "✓" if p.dza is not None else "✗", "Other files": len(p.other_files)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    return st.selectbox("Choose project", stems)


def render_metadata(gpr_data: GPRData):
    st.write(f"**File type:** {gpr_data.file_type}")
    st.write(f"**Source file:** {gpr_data.source_file}")
    if gpr_data.traces is not None:
        st.write(f"**Trace matrix shape:** {gpr_data.traces.shape}")
    if gpr_data.metadata:
        st.dataframe(pd.DataFrame([{"Field": k, "Value": str(v)} for k, v in gpr_data.metadata.items()]), width="stretch", hide_index=True)


def render_companion_panels(project: RadanProject):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**DZG**")
        st.json(project.dzg.gps or project.dzg.metadata) if project.dzg is not None else st.write("No DZG loaded")
    with c2:
        st.write("**DZX**")
        st.json(project.dzx.metadata) if project.dzx is not None else st.write("No DZX loaded")
    with c3:
        st.write("**DZA**")
        st.json(project.dza.metadata) if project.dza is not None else st.write("No DZA loaded")


def main():
    init_page()
    controls = render_sidebar()
    demo_mode, uploaded_files = render_file_loader()
    if demo_mode == "none" and not uploaded_files:
        st.info("Upload one or more related files, or choose a built-in demo project.")
        st.stop()

    projects = build_projects_from_uploads(uploaded_files or [], demo_mode=demo_mode)
    if not projects:
        st.warning("No readable projects were assembled.")
        st.stop()

    st.subheader("2. Project Inventory")
    selected_stem = render_project_inventory(projects)
    project = projects[selected_stem]
    primary = project.dzt or project.dzg or project.dzx or project.dza
    if primary is None:
        st.warning("This project has no primary readable RADAN file yet.")
        render_companion_panels(project)
        st.stop()

    processed = None
    processing_notes: list[str] = []
    if project.dzt is not None:
        n_traces, n_samples = project.dzt.traces.shape
        trace_crop = st.sidebar.slider("Trace crop", 0, n_traces - 1, (0, n_traces - 1))
        sample_crop = st.sidebar.slider("Sample crop", 0, n_samples - 1, (0, n_samples - 1))
        processed, processing_notes = process_gpr_data(
            project.dzt,
            time_zero_shift=controls["time_zero_shift"],
            dewow_on=controls["dewow_on"],
            dewow_window=controls["dewow_window"],
            background_on=controls["background_on"],
            normalize_on=controls["normalize_on"],
            gain_mode=controls["gain_mode"],
            gain_strength=controls["gain_strength"],
            trace_crop=trace_crop,
            sample_crop=sample_crop,
        )

    tabs = st.tabs(["Load / Inventory", "Radargram", "Processed Radargram", "Single Trace", "Companions", "Interpret", "Report / Export"])

    with tabs[0]:
        st.subheader("Project metadata")
        render_metadata(primary)
        if processing_notes:
            st.markdown("**Processing notes**")
            for note in processing_notes:
                st.write(f"- {note}")
        if primary.metadata.get("demo_type") == "graveyard_training":
            st.info("This is a synthetic cemetery teaching line. Students should use it to practice cautious identification of grave-shaft-like disturbances and hyperbola-like targets without overclaiming burial confirmation.")

    with tabs[1]:
        if project.dzt is not None:
            st.subheader("Raw radargram")
            st.pyplot(make_radargram_figure(project.dzt, controls["use_distance"], controls["clip_low"], controls["clip_high"], controls["invert_y"], controls["cmap"]), clear_figure=True, width="stretch")
        else:
            st.info("No DZT loaded.")

    with tabs[2]:
        if processed is not None:
            st.subheader("Processed radargram")
            st.pyplot(make_radargram_figure(processed, controls["use_distance"], controls["clip_low"], controls["clip_high"], controls["invert_y"], controls["cmap"]), clear_figure=True, width="stretch")
        else:
            st.info("No processed radargram available.")

    with tabs[3]:
        if project.dzt is not None:
            st.subheader("Raw vs processed single trace")
            trace_index = st.slider("Trace index", 0, max(0, project.dzt.traces.shape[0] - 1), 0)
            st.pyplot(make_trace_figure(project.dzt, processed, trace_index), clear_figure=True)
        else:
            st.info("No DZT loaded.")

    with tabs[4]:
        st.subheader("Companion files")
        render_companion_panels(project)
        if project.other_files:
            rows = [{"File": n, "Type guess": ins.likely_type, "Binary": ins.is_binary, "Size": ins.size_bytes, "Notes": ins.notes} for n, ins in project.other_files.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tabs[5]:
        st.subheader("Interpret")
        if primary.metadata.get("demo_type") == "graveyard_training":
            st.markdown("**Teaching prompts for students**")
            st.markdown("- Look for grave-shaft-like zones: vertical disturbed columns where reflectors lose continuity.")
            st.markdown("- Look for hyperbola-like targets, but do not assume every hyperbola is a coffin or burial.")
            st.markdown("- Compare anomalies against surrounding background and deeper natural reflectors.")
            st.markdown("- State uncertainty clearly.")
        if controls["ai_enable"]:
            provider_bits = []
            if ollama_available(controls["ollama_base_url"]):
                provider_bits.append(f"Ollama ready at {controls['ollama_base_url']}")
            if openai_available():
                provider_bits.append("OpenAI key detected")
            st.caption(" | ".join(provider_bits) if provider_bits else "No AI backend detected right now. The app still works without AI.")
            ai_question = st.text_area("AI request", value="Give me a concise interpretation of this GPR line for student training, with cautions about uncertainty.")
            if st.button("Generate AI summary", width="stretch"):
                try:
                    ai_text, provider = generate_ai_summary(project, project.dzt or primary, processed, processing_notes, ai_question, controls["ai_prefer_local"], controls["ollama_model"], controls["ollama_base_url"], controls["openai_model"])
                    st.success(f"AI provider used: {provider}")
                    st.markdown(ai_text)
                except Exception as exc:
                    st.warning(str(exc))
        else:
            st.caption("AI is off. Core analysis remains fully available without it.")

    with tabs[6]:
        st.subheader("Report / Export")
        report_lines = [
            f"Project: {project.stem}",
            f"Source file: {primary.source_file}",
            f"File type: {primary.file_type}",
            f"Trace matrix shape: {project.dzt.traces.shape if project.dzt is not None else 'none'}",
        ]
        if processing_notes:
            report_lines.append("Processing notes:")
            report_lines.extend([f"- {n}" for n in processing_notes])
        if primary.metadata.get("demo_type") == "graveyard_training":
            report_lines.append("This is a synthetic cemetery training line. Grave-like targets are educational only.")
        report_text = "\n".join(report_lines)
        st.code(report_text)
        c1, c2, c3 = st.columns(3)
        if project.dzt is not None:
            with c1:
                st.download_button("Download raw trace matrix as CSV", data=traces_to_csv_bytes(project.dzt), file_name=f"{Path(project.dzt.source_file).stem}_raw_traces.csv", mime="text/csv", width="stretch")
        if processed is not None:
            with c2:
                st.download_button("Download processed trace matrix as CSV", data=traces_to_csv_bytes(processed), file_name=f"{Path(project.dzt.source_file).stem}_processed_traces.csv", mime="text/csv", width="stretch")
        with c3:
            st.download_button("Download report as TXT", data=report_text.encode("utf-8"), file_name=f"{project.stem}_report.txt", mime="text/plain", width="stretch")


if __name__ == "__main__":
    main()

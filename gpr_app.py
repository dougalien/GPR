import io
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class GPRData:
    file_type: str
    source_file: str
    traces: Optional[np.ndarray] = None  # shape: (n_traces, n_samples)
    time_axis: Optional[np.ndarray] = None
    distance_axis: Optional[np.ndarray] = None
    gps: Optional[Dict[str, Any]] = None
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


# -----------------------------
# Helpers
# -----------------------------
def safe_get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return str(os.getenv(name, default))


def moving_average_1d(y: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def running_mean_rows(arr: np.ndarray, window: int) -> np.ndarray:
    return np.vstack([moving_average_1d(row, window) for row in arr])


def gaussian_like_smooth_rows(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    out = arr.copy().astype(float)
    kernel = np.array([1, 2, 1], dtype=float) / 4.0
    for _ in range(max(1, passes)):
        out = np.vstack([np.convolve(r, kernel, mode="same") for r in out])
    return out


def normalize_per_trace(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    scale = np.max(np.abs(out), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return out / scale


def linear_gain(n_samples: int, strength: float) -> np.ndarray:
    return np.linspace(1.0, 1.0 + max(0.0, strength), n_samples)


def exponential_gain(n_samples: int, strength: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n_samples)
    return np.exp(max(0.0, strength) * x)


def robust_clip_limits(data: np.ndarray, low_pct: float, high_pct: float) -> Tuple[float, float]:
    return float(np.percentile(data, low_pct)), float(np.percentile(data, high_pct))


def axis_extent(gpr_data: GPRData, invert_y: bool, trace_start: int = 0, trace_end: Optional[int] = None,
                sample_start: int = 0, sample_end: Optional[int] = None, use_distance: bool = True) -> Tuple[List[float], str, str]:
    arr = gpr_data.traces
    n_traces, n_samples = arr.shape
    trace_end = n_traces if trace_end is None else trace_end
    sample_end = n_samples if sample_end is None else sample_end

    if use_distance and gpr_data.distance_axis is not None and len(gpr_data.distance_axis) == n_traces:
        x_vals = gpr_data.distance_axis[trace_start:trace_end]
        xlabel = "Distance"
    else:
        x_vals = np.arange(trace_start, trace_end)
        xlabel = "Trace Number"

    if gpr_data.time_axis is not None and len(gpr_data.time_axis) == n_samples:
        y_vals = gpr_data.time_axis[sample_start:sample_end]
        ylabel = "Time (ns)"
    else:
        y_vals = np.arange(sample_start, sample_end)
        ylabel = "Sample"

    extent = [x_vals[0], x_vals[-1], y_vals[-1], y_vals[0]] if invert_y else [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]
    return extent, xlabel, ylabel


# -----------------------------
# File reading / grouping
# -----------------------------
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
        likely_map = {".dzt": "gssi_dzt", ".tmf": "gssi_tmf", ".dza": "gssi_dza", ".dzg": "gssi_dzg", ".dzx": "gssi_dzx"}
        likely_type = likely_map.get(ext, "binary_unknown")
        notes = f"Binary file with {ext} extension."

    return FileInspection(name, ext, size, is_binary, likely_type, notes, preview_text)


def parse_dzt(name: str, content: bytes) -> GPRData:
    if len(content) < 1024:
        raise ValueError("File too small for a valid 1024-byte DZT header.")

    header = content[:1024]
    data = content[1024:]

    rh_nsamp = struct.unpack_from("<h", header, 4)[0]
    rh_bits = struct.unpack_from("<h", header, 6)[0]
    rh_zero = struct.unpack_from("<h", header, 8)[0]
    rhf_sps = struct.unpack_from("<f", header, 80)[0]
    rhf_spm = struct.unpack_from("<f", header, 84)[0]
    rhf_position = struct.unpack_from("<f", header, 88)[0]
    rhf_range = struct.unpack_from("<f", header, 92)[0]
    rh_nchan = struct.unpack_from("<h", header, 54)[0] if len(header) >= 56 else 1

    if rh_nsamp <= 0:
        raise ValueError(f"Invalid samples per trace: {rh_nsamp}")

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
    if n_traces == 0:
        raise ValueError("No usable trace data found.")

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
        "n_traces": int(n_traces),
        "header_size_bytes": 1024,
    }
    return GPRData("dzt", name, centered, time_axis, distance_axis, metadata=metadata, raw_bytes=content)


def parse_csv_or_text_matrix(name: str, content: bytes, forced_type: str = "text") -> GPRData:
    # Try CSV, tab, whitespace. Expect rows=traces and cols=samples or vice versa.
    text = content.decode("utf-8", errors="ignore")
    df = None
    for sep in [",", "\t", None]:
        try:
            if sep is None:
                tmp = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
            else:
                tmp = pd.read_csv(io.StringIO(text), sep=sep, header=None)
            if not tmp.empty:
                df = tmp.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
                if not df.empty:
                    break
        except Exception:
            continue
    if df is None or df.empty:
        raise ValueError("Could not parse text matrix.")

    arr = df.to_numpy(dtype=float)
    # Heuristic: if fewer rows than cols, rows are likely traces? keep as is if many rows.
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < 80:
        arr = arr.T

    metadata = {"n_traces": int(arr.shape[0]), "samples_per_trace": int(arr.shape[1]), "parsed_from": forced_type}
    return GPRData(forced_type, name, arr, time_axis=np.arange(arr.shape[1]), distance_axis=np.arange(arr.shape[0]), metadata=metadata, raw_bytes=content)


def parse_text_sidecar(name: str, content: bytes, forced_type: str) -> GPRData:
    text = content.decode("utf-8", errors="ignore")
    gps = None
    if "$GPGGA" in text or "$GNGGA" in text:
        gps = {"note": "NMEA GPS strings detected", "preview": text[:300]}
    return GPRData(forced_type, name, gps=gps, metadata={"preview": text[:300], "length_chars": len(text)}, raw_bytes=content)


def load_uploaded_file(uploaded_file) -> Tuple[Optional[GPRData], Optional[FileInspection]]:
    name = uploaded_file.name
    content = uploaded_file.getvalue()
    ext = Path(name).suffix.lower()
    if ext == ".dzt":
        return parse_dzt(name, content), None
    if ext == ".csv":
        return parse_csv_or_text_matrix(name, content, "csv"), None
    if ext in {".txt", ".asc"}:
        return parse_csv_or_text_matrix(name, content, ext.lstrip('.')), None
    if ext in {".dzg", ".dzx", ".dza"}:
        return parse_text_sidecar(name, content, ext.lstrip('.')), inspect_uploaded_file(name, content)
    return None, inspect_uploaded_file(name, content)


def build_projects_from_uploads(uploaded_files) -> Dict[str, RadanProject]:
    projects: Dict[str, RadanProject] = {}
    for uploaded_file in uploaded_files:
        stem = Path(uploaded_file.name).stem
        project = projects.setdefault(stem, RadanProject(stem=stem))
        parsed, inspection = load_uploaded_file(uploaded_file)
        ext = Path(uploaded_file.name).suffix.lower().lstrip('.')
        if ext in {"dzt", "dzg", "dzx", "dza"} and parsed is not None:
            setattr(project, ext, parsed)
            project.files[ext] = uploaded_file.name
        elif parsed is not None and ext in {"csv", "txt", "asc"}:
            # Treat matrix file as primary if no dzt
            project.dzt = parsed
            project.files[ext] = uploaded_file.name
        else:
            if inspection is None:
                inspection = inspect_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
            project.other_files[uploaded_file.name] = inspection
    return projects


# -----------------------------
# Demo raw data
# -----------------------------
def build_plain_training_line() -> GPRData:
    n_traces = 200
    n_samples = 320
    rng = np.random.default_rng(42)
    t = np.linspace(0, 120, n_samples)
    x = np.linspace(0, 20, n_traces)

    data = rng.normal(0, 8, size=(n_traces, n_samples))

    # Horizontal banding / coupling / shallow reflections
    shallow = 18 + (2.0 * np.sin(np.arange(n_traces) / 18.0)).astype(int)
    for i in range(n_traces):
        idx = shallow[i]
        data[i, max(0, idx - 1):min(n_samples, idx + 2)] += 45

    horizon1 = 95 + (3.0 * np.sin(np.arange(n_traces) / 35.0)).astype(int)
    horizon2 = 155 + (5.0 * np.sin(np.arange(n_traces) / 24.0)).astype(int)
    for i in range(n_traces):
        data[i, max(0, horizon1[i] - 1):min(n_samples, horizon1[i] + 2)] += 28
        data[i, max(0, horizon2[i] - 1):min(n_samples, horizon2[i] + 2)] += 20

    # Disturbed zones without explicit perfect targets
    disturbed_cols = [(46, 58), (102, 114), (148, 160)]
    for a, b in disturbed_cols:
        data[a:b, :] *= 0.90
        for i in range(a, b):
            h1 = horizon1[i] + rng.integers(-6, 7)
            h2 = horizon2[i] + rng.integers(-10, 11)
            data[i, max(0, h1 - 1):min(n_samples, h1 + 2)] += rng.normal(8, 6)
            data[i, max(0, h2 - 1):min(n_samples, h2 + 2)] += rng.normal(6, 5)

    # Mild ambiguous hyperbola-like clutter, not textbook clean
    for center, apex, scale, amp in [(74, 118, 0.055, 22), (171, 135, 0.040, 18)]:
        for i in range(n_traces):
            idx = int(apex + scale * (i - center) ** 2)
            if 1 <= idx < n_samples - 1:
                data[i, idx - 1:idx + 2] += amp * (0.7 + 0.3 * rng.random())

    attenuation = np.linspace(1.0, 0.48, n_samples)
    data *= attenuation[np.newaxis, :]

    metadata = {
        "demo_mode": True,
        "description": "Plain raw-style training line with banding, disturbed zones, and ambiguous anomalies.",
        "n_traces": n_traces,
        "samples_per_trace": n_samples,
        "range_ns": float(t[-1]),
    }
    return GPRData("demo", "plain_training_line.csv", data, t, x, metadata=metadata)


# -----------------------------
# Processing
# -----------------------------
def process_radargram(raw: np.ndarray, settings: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    arr = raw.astype(float).copy()
    notes: List[str] = []

    # Crop first for speed and clarity
    t0 = int(settings["trace_start"])
    t1 = int(settings["trace_end"])
    s0 = int(settings["sample_start"])
    s1 = int(settings["sample_end"])
    arr = arr[t0:t1, s0:s1]
    notes.append(f"Cropped to traces {t0}:{t1} and samples {s0}:{s1}.")

    # Time-zero shift
    shift = int(settings["time_zero_shift"])
    if shift != 0:
        arr = np.roll(arr, -shift, axis=1)
        notes.append(f"Applied time-zero shift of {shift} samples.")

    if settings["dewow_on"]:
        w = int(settings["dewow_window"])
        arr = arr - running_mean_rows(arr, w)
        notes.append(f"Applied dewow with window {w}.")

    if settings["background_on"]:
        background = np.mean(arr, axis=0, keepdims=True)
        arr = arr - background
        notes.append("Applied background removal.")

    gain_mode = settings["gain_mode"]
    gain_strength = float(settings["gain_strength"])
    if gain_mode != "None" and gain_strength > 0:
        if gain_mode == "Linear":
            g = linear_gain(arr.shape[1], gain_strength)
        else:
            g = exponential_gain(arr.shape[1], gain_strength)
        arr = arr * g[np.newaxis, :]
        notes.append(f"Applied {gain_mode.lower()} gain with strength {gain_strength:.2f}.")

    if settings["normalize_traces"]:
        arr = normalize_per_trace(arr)
        notes.append("Applied per-trace normalization.")

    if settings["light_smooth_on"]:
        arr = gaussian_like_smooth_rows(arr, passes=int(settings["light_smooth_passes"]))
        notes.append(f"Applied light smoothing ({int(settings['light_smooth_passes'])} passes).")

    return arr, notes


# -----------------------------
# Candidate scoring
# -----------------------------
def local_vertical_variance(arr: np.ndarray) -> np.ndarray:
    centered = arr - np.mean(arr, axis=1, keepdims=True)
    return np.mean(centered**2, axis=1)


def continuity_break_score(arr: np.ndarray) -> np.ndarray:
    # Score traces where lateral change is high relative to neighbors.
    if arr.shape[0] < 3:
        return np.zeros(arr.shape[0])
    diff_prev = np.abs(arr[1:-1] - arr[:-2])
    diff_next = np.abs(arr[1:-1] - arr[2:])
    core = np.mean(diff_prev + diff_next, axis=1)
    out = np.zeros(arr.shape[0])
    out[1:-1] = core
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def hyperbola_like_score(arr: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    n_traces, n_samples = arr.shape
    score_map = np.zeros_like(arr, dtype=float)
    candidates = []

    # Very simple apex finder: local maxima in absolute amplitude where neighboring traces dip slightly deeper.
    abs_arr = np.abs(arr)
    if n_traces < 7 or n_samples < 20:
        return score_map, pd.DataFrame(columns=["Trace", "Sample", "Type", "Score", "Reason"])

    max_amp = np.percentile(abs_arr, 98)
    thresh = max(max_amp * 0.55, np.percentile(abs_arr, 92))

    for i in range(3, n_traces - 3):
        trace = abs_arr[i]
        local_peaks = np.where((trace[1:-1] > trace[:-2]) & (trace[1:-1] >= trace[2:]) & (trace[1:-1] > thresh))[0] + 1
        for j in local_peaks:
            if j < 10 or j > n_samples - 15:
                continue
            center = abs_arr[i, j]
            left_min = min(abs_arr[i - 1, min(j + 1, n_samples - 1)], abs_arr[i - 2, min(j + 2, n_samples - 1)])
            right_min = min(abs_arr[i + 1, min(j + 1, n_samples - 1)], abs_arr[i + 2, min(j + 2, n_samples - 1)])
            neighbor_energy = np.mean([
                abs_arr[i - 1, j:j + 4].mean(),
                abs_arr[i + 1, j:j + 4].mean(),
                abs_arr[i - 2, j:j + 5].mean(),
                abs_arr[i + 2, j:j + 5].mean(),
            ])
            score = max(0.0, center - 0.6 * (left_min + right_min) - 0.2 * neighbor_energy)
            if score > np.percentile(abs_arr, 80):
                score_map[max(0, i - 2):min(n_traces, i + 3), max(0, j - 2):min(n_samples, j + 6)] += score
                candidates.append({
                    "Trace": i,
                    "Sample": j,
                    "Type": "Possible hyperbolic target",
                    "Score": round(float(score), 2),
                    "Reason": "Local apex with deeper neighboring energy.",
                })

    cand_df = pd.DataFrame(candidates)
    if not cand_df.empty:
        cand_df = cand_df.sort_values("Score", ascending=False).drop_duplicates(subset=["Trace", "Sample"]).reset_index(drop=True)
    else:
        cand_df = pd.DataFrame(columns=["Trace", "Sample", "Type", "Score", "Reason"])
    return score_map, cand_df


def disturbance_candidates(arr: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    var_score = local_vertical_variance(arr)
    cont_score = continuity_break_score(arr)
    combo = 0.45 * (var_score / (np.max(var_score) + 1e-9)) + 0.55 * (cont_score / (np.max(cont_score) + 1e-9))

    threshold = np.percentile(combo, 88)
    flagged = combo >= threshold
    rows = []
    score_map = np.zeros_like(arr, dtype=float)

    i = 0
    while i < len(flagged):
        if not flagged[i]:
            i += 1
            continue
        start = i
        while i < len(flagged) and flagged[i]:
            i += 1
        end = i
        width = end - start
        if width >= 3:
            score = float(np.mean(combo[start:end]))
            score_map[start:end, :] += score
            rows.append({
                "Trace": int((start + end) // 2),
                "Sample": int(arr.shape[1] * 0.35),
                "Type": "Possible disturbed zone",
                "Score": round(score, 3),
                "Reason": f"Elevated reflector disruption / variance across traces {start}-{end - 1}.",
                "Trace Start": start,
                "Trace End": end - 1,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Trace", "Sample", "Type", "Score", "Reason", "Trace Start", "Trace End"])
    return score_map, df


def build_candidate_table(processed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dist_map, dist_df = disturbance_candidates(processed)
    hyp_map, hyp_df = hyperbola_like_score(processed)
    frames = []
    if not dist_df.empty:
        frames.append(dist_df)
    if not hyp_df.empty:
        frames.append(hyp_df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Trace", "Sample", "Type", "Score", "Reason"])
    if not combined.empty:
        combined = combined.sort_values("Score", ascending=False).reset_index(drop=True)
    return dist_map, hyp_map, combined


# -----------------------------
# AI guidance
# -----------------------------
def ollama_available(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.2)
        return r.ok
    except Exception:
        return False


def call_ollama(prompt: str, model: str, base_url: str) -> str:
    r = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=90,
    )
    r.raise_for_status()
    return str(r.json().get("response", "")).strip()


def openai_available() -> bool:
    return bool(safe_get_secret("OPENAI_API_KEY", ""))


def call_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    api_key = safe_get_secret("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful GPR teaching assistant. Do not claim graves or definitive object identities. Use cautious language and explain uncertainty."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=90,
    )
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"]).strip()


def generate_guidance(project: RadanProject, notes: List[str], candidates: pd.DataFrame, prefer_local: bool, ollama_model: str, ollama_url: str, openai_model: str) -> Tuple[str, str]:
    primary = project.dzt or project.dzg or project.dzx or project.dza
    meta = primary.metadata if primary else {}
    top_rows = candidates.head(8).to_dict(orient="records") if candidates is not None and not candidates.empty else []
    prompt = (
        "Review this GPR teaching-assistant context and produce a concise interpretation guide for students. "
        "Do not claim graves or certainty. Explain what to review, what alternative explanations exist, and what processing choices might matter.\n\n"
        f"Project: {project.stem}\n"
        f"Metadata: {meta}\n"
        f"Processing notes: {'; '.join(notes)}\n"
        f"Top candidate rows: {top_rows}\n"
    )
    if prefer_local and ollama_available(ollama_url):
        try:
            return call_ollama(prompt, ollama_model, ollama_url), f"Ollama ({ollama_model})"
        except Exception:
            pass
    if openai_available():
        try:
            return call_openai(prompt, openai_model), f"OpenAI ({openai_model})"
        except Exception:
            pass
    raise RuntimeError("No AI provider available. Core interpretation workflow still works without AI.")


# -----------------------------
# Plotting
# -----------------------------
def plot_radargram(gpr_data: GPRData, data: np.ndarray, title: str, use_distance: bool, invert_y: bool,
                   clip_low: float, clip_high: float, cmap: str,
                   trace_start: int, trace_end: int, sample_start: int, sample_end: int,
                   candidate_df: Optional[pd.DataFrame] = None, show_overlay: bool = False):
    extent, xlabel, ylabel = axis_extent(gpr_data, invert_y, trace_start, trace_end, sample_start, sample_end, use_distance)
    vmin, vmax = robust_clip_limits(data, clip_low, clip_high)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(data.T, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_overlay and candidate_df is not None and not candidate_df.empty:
        for _, row in candidate_df.head(20).iterrows():
            tr = int(row.get("Trace", 0))
            smp = int(row.get("Sample", 0))
            typ = str(row.get("Type", "Candidate"))
            if "disturbed" in typ.lower():
                start = int(row.get("Trace Start", max(tr - 2, 0)))
                end = int(row.get("Trace End", min(tr + 2, data.shape[0] - 1)))
                x0 = start if not (use_distance and gpr_data.distance_axis is not None) else gpr_data.distance_axis[min(trace_start + start, len(gpr_data.distance_axis)-1)]
                x1 = end if not (use_distance and gpr_data.distance_axis is not None) else gpr_data.distance_axis[min(trace_start + end, len(gpr_data.distance_axis)-1)]
                y0 = extent[2] if invert_y else extent[2]
                height = abs(extent[3] - extent[2])
                rect = Rectangle((x0, min(extent[2], extent[3])), max(x1 - x0, 0.2), height,
                                 fill=False, edgecolor="deepskyblue", linewidth=1.3, linestyle="--")
                ax.add_patch(rect)
            else:
                x = tr if not (use_distance and gpr_data.distance_axis is not None) else gpr_data.distance_axis[min(trace_start + tr, len(gpr_data.distance_axis)-1)]
                y = smp if gpr_data.time_axis is None else gpr_data.time_axis[min(sample_start + smp, len(gpr_data.time_axis)-1)]
                ax.scatter([x], [y], c="yellow", s=42, edgecolors="black", linewidths=0.5)
    fig.tight_layout()
    return fig


def plot_trace(raw_trace: np.ndarray, proc_trace: np.ndarray, time_axis: np.ndarray, trace_idx: int):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(raw_trace, time_axis, label="Raw", alpha=0.7)
    ax.plot(proc_trace, time_axis, label="Processed", linewidth=1.2)
    ax.set_title(f"Trace {trace_idx}: Raw vs Processed")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Time / Sample")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------
# App
# -----------------------------
def init_page():
    st.set_page_config(page_title="GPR Teaching Assistant", layout="wide")
    st.title("GPR Teaching Assistant")
    st.caption("Raw-first workflow for loading, processing, reviewing, and teaching GPR interpretation.")


def sidebar_controls(n_traces: int, n_samples: int) -> Dict[str, Any]:
    st.sidebar.header("Display")
    clip_low = st.sidebar.slider("Lower contrast clip percentile", 0.0, 20.0, 2.0, 0.5)
    clip_high = st.sidebar.slider("Upper contrast clip percentile", 80.0, 100.0, 98.0, 0.5)
    cmap = st.sidebar.selectbox("Colormap", ["gray", "seismic", "viridis", "plasma"], index=0)
    use_distance = st.sidebar.checkbox("Use distance axis if available", value=True)
    invert_y = st.sidebar.checkbox("Invert vertical axis", value=True)

    st.sidebar.header("Crop")
    trace_range = st.sidebar.slider("Trace range", 0, max(1, n_traces), (0, max(1, n_traces)))
    sample_range = st.sidebar.slider("Sample range", 0, max(1, n_samples), (0, max(1, n_samples)))

    st.sidebar.header("Processing")
    time_zero_shift = st.sidebar.slider("Time-zero shift (samples)", -20, 20, 0)
    dewow_on = st.sidebar.checkbox("Dewow", value=True)
    dewow_window = st.sidebar.slider("Dewow window", 3, 101, 21, step=2)
    background_on = st.sidebar.checkbox("Background removal", value=True)
    gain_mode = st.sidebar.selectbox("Gain mode", ["None", "Linear", "Exponential"], index=1)
    gain_strength = st.sidebar.slider("Gain strength", 0.0, 5.0, 1.2, 0.1)
    normalize_traces = st.sidebar.checkbox("Per-trace normalization", value=False)
    light_smooth_on = st.sidebar.checkbox("Light smoothing", value=False)
    light_smooth_passes = st.sidebar.slider("Smoothing passes", 1, 4, 2)

    st.sidebar.header("AI Guidance")
    ai_enable = st.sidebar.checkbox("Enable optional AI guidance", value=False)
    prefer_local = st.sidebar.checkbox("Prefer local Ollama", value=True)
    ollama_model = st.sidebar.text_input("Ollama model", value="llama3.1:8b")
    ollama_url = st.sidebar.text_input("Ollama URL", value="http://127.0.0.1:11434")
    openai_model = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")

    return {
        "clip_low": clip_low,
        "clip_high": clip_high,
        "cmap": cmap,
        "use_distance": use_distance,
        "invert_y": invert_y,
        "trace_start": trace_range[0],
        "trace_end": trace_range[1],
        "sample_start": sample_range[0],
        "sample_end": sample_range[1],
        "time_zero_shift": time_zero_shift,
        "dewow_on": dewow_on,
        "dewow_window": dewow_window,
        "background_on": background_on,
        "gain_mode": gain_mode,
        "gain_strength": gain_strength,
        "normalize_traces": normalize_traces,
        "light_smooth_on": light_smooth_on,
        "light_smooth_passes": light_smooth_passes,
        "ai_enable": ai_enable,
        "prefer_local": prefer_local,
        "ollama_model": ollama_model,
        "ollama_url": ollama_url,
        "openai_model": openai_model,
    }


def render_companions(project: RadanProject):
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

    if project.other_files:
        rows = [{"File": name, "Type guess": v.likely_type, "Size": v.size_bytes, "Notes": v.notes} for name, v in project.other_files.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def export_report(project: RadanProject, notes: List[str], candidates: pd.DataFrame) -> bytes:
    primary = project.dzt or project.dzg or project.dzx or project.dza
    lines = [
        f"Project: {project.stem}",
        f"Primary source: {primary.source_file if primary else 'None'}",
        "",
        "Metadata:",
    ]
    if primary is not None:
        for k, v in primary.metadata.items():
            lines.append(f"- {k}: {v}")
    lines.extend(["", "Processing notes:"])
    for n in notes:
        lines.append(f"- {n}")
    lines.extend(["", "Candidate review table:"])
    if candidates is not None and not candidates.empty:
        lines.append(candidates.head(20).to_csv(index=False))
    else:
        lines.append("No candidates flagged.")
    return "\n".join(lines).encode("utf-8")


def main():
    init_page()

    st.subheader("1. Load Raw Project Files")
    demo_mode = st.selectbox("Demo mode", ["None", "Plain training line"], index=0)
    uploaded = st.file_uploader(
        "Upload one or more related raw files",
        type=["dzt", "csv", "txt", "asc", "dzg", "dzx", "dza"],
        accept_multiple_files=True,
        help="Upload raw radargram files and any companion files together.",
    )

    projects: Dict[str, RadanProject] = {}
    if demo_mode == "Plain training line":
        rp = RadanProject(stem="plain_training_line")
        rp.dzt = build_plain_training_line()
        projects[rp.stem] = rp
    if uploaded:
        projects.update(build_projects_from_uploads(uploaded))

    if not projects:
        st.info("Upload raw files or choose the plain training line demo.")
        st.stop()

    st.subheader("2. Project Inventory")
    rows = []
    for stem, p in sorted(projects.items()):
        primary = p.dzt or p.dzg or p.dzx or p.dza
        rows.append({
            "Project": stem,
            "Primary": primary.file_type if primary else "None",
            "Source": primary.source_file if primary else "None",
            "DZT/Matrix": "✓" if p.dzt is not None else "✗",
            "DZG": "✓" if p.dzg is not None else "✗",
            "DZX": "✓" if p.dzx is not None else "✗",
            "DZA": "✓" if p.dza is not None else "✗",
            "Other files": len(p.other_files),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    selected_stem = st.selectbox("Choose project", list(sorted(projects.keys())))
    project = projects[selected_stem]
    primary = project.dzt or project.dzg or project.dzx or project.dza
    if primary is None or primary.traces is None:
        st.warning("This project has no readable trace matrix yet.")
        render_companions(project)
        st.stop()

    n_traces, n_samples = primary.traces.shape
    controls = sidebar_controls(n_traces, n_samples)

    processed, notes = process_radargram(primary.traces, controls)
    dist_map, hyp_map, candidate_df = build_candidate_table(processed)
    proc_time = primary.time_axis[controls["sample_start"]:controls["sample_end"]] if primary.time_axis is not None else np.arange(processed.shape[1])

    tabs = st.tabs(["Raw", "Processed", "Overlay / Review", "Single Trace", "Companions", "Interpret", "Report / Export"])

    with tabs[0]:
        st.markdown("### Raw radargram")
        st.caption("This is the baseline image. Interpretation should begin here before relying on overlays.")
        raw_crop = primary.traces[controls["trace_start"]:controls["trace_end"], controls["sample_start"]:controls["sample_end"]]
        fig = plot_radargram(primary, raw_crop, f"Raw Radargram: {primary.source_file}", controls["use_distance"], controls["invert_y"],
                             controls["clip_low"], controls["clip_high"], controls["cmap"],
                             controls["trace_start"], controls["trace_end"], controls["sample_start"], controls["sample_end"])
        st.pyplot(fig, clear_figure=True, width="stretch")
        meta_rows = [{"Field": k, "Value": str(v)} for k, v in primary.metadata.items()]
        st.dataframe(pd.DataFrame(meta_rows), width="stretch", hide_index=True)

    with tabs[1]:
        st.markdown("### Processed radargram")
        st.caption("The processed image is derived from the raw data using the chosen steps in the sidebar.")
        fig = plot_radargram(primary, processed, "Processed Radargram", controls["use_distance"], controls["invert_y"],
                             controls["clip_low"], controls["clip_high"], controls["cmap"],
                             controls["trace_start"], controls["trace_end"], controls["sample_start"], controls["sample_end"])
        st.pyplot(fig, clear_figure=True, width="stretch")
        for n in notes:
            st.write(f"- {n}")

    with tabs[2]:
        st.markdown("### Candidate overlays on raw and processed data")
        left, right = st.columns(2)
        raw_crop = primary.traces[controls["trace_start"]:controls["trace_end"], controls["sample_start"]:controls["sample_end"]]
        with left:
            fig = plot_radargram(primary, raw_crop, "Raw with Candidate Overlays", controls["use_distance"], controls["invert_y"],
                                 controls["clip_low"], controls["clip_high"], controls["cmap"],
                                 controls["trace_start"], controls["trace_end"], controls["sample_start"], controls["sample_end"],
                                 candidate_df=candidate_df, show_overlay=True)
            st.pyplot(fig, clear_figure=True, width="stretch")
        with right:
            fig = plot_radargram(primary, processed, "Processed with Candidate Overlays", controls["use_distance"], controls["invert_y"],
                                 controls["clip_low"], controls["clip_high"], controls["cmap"],
                                 controls["trace_start"], controls["trace_end"], controls["sample_start"], controls["sample_end"],
                                 candidate_df=candidate_df, show_overlay=True)
            st.pyplot(fig, clear_figure=True, width="stretch")

        st.markdown("### Candidate review table")
        if candidate_df.empty:
            st.info("No strong candidate regions were flagged with the current processing settings.")
        else:
            review_df = candidate_df.copy()
            review_df["Student Review"] = ""
            review_df["Instructor Note"] = ""
            st.dataframe(review_df.head(30), width="stretch", hide_index=True)

    with tabs[3]:
        st.markdown("### Single trace: raw vs processed")
        tr = st.slider("Trace index within cropped range", 0, max(0, processed.shape[0] - 1), 0)
        raw_trace = primary.traces[controls["trace_start"] + tr, controls["sample_start"]:controls["sample_end"]]
        proc_trace = processed[tr, :]
        fig = plot_trace(raw_trace, proc_trace, proc_time, controls["trace_start"] + tr)
        st.pyplot(fig, clear_figure=True, width="content")

    with tabs[4]:
        st.markdown("### Companion files")
        render_companions(project)

    with tabs[5]:
        st.markdown("### Guided interpretation")
        st.write("Use the candidate table as a review aid, not as proof of any specific subsurface target.")
        if candidate_df.empty:
            st.write("- No strong candidate regions were flagged. Try different processing settings and compare raw versus processed views.")
        else:
            top = candidate_df.head(8)
            st.write("Common questions to ask:")
            st.write("- Does this feature persist after processing, or is it created by the processing itself?")
            st.write("- Is the anomaly a local apex, a disturbed vertical zone, a broken reflector, or just clutter?")
            st.write("- What alternative explanations exist: roots, stones, surface effects, antenna coupling, uneven gain, banding?")
            st.dataframe(top[[c for c in ["Type", "Trace", "Sample", "Score", "Reason"] if c in top.columns]], width="stretch", hide_index=True)

        if controls["ai_enable"]:
            if st.button("Generate optional AI guidance", width="stretch"):
                try:
                    guidance, provider = generate_guidance(project, notes, candidate_df, controls["prefer_local"], controls["ollama_model"], controls["ollama_url"], controls["openai_model"])
                    st.success(f"Guidance generated with {provider}")
                    st.write(guidance)
                    st.session_state["gpr_ai_guidance"] = guidance
                    st.session_state["gpr_ai_provider"] = provider
                except Exception as e:
                    st.warning(str(e))
        elif st.session_state.get("gpr_ai_guidance"):
            st.write(st.session_state["gpr_ai_guidance"])

    with tabs[6]:
        st.markdown("### Report and exports")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download raw matrix as CSV", pd.DataFrame(primary.traces).to_csv(index=False).encode("utf-8"), f"{project.stem}_raw.csv", "text/csv", width="stretch")
        with c2:
            st.download_button("Download processed matrix as CSV", pd.DataFrame(processed).to_csv(index=False).encode("utf-8"), f"{project.stem}_processed.csv", "text/csv", width="stretch")
        with c3:
            st.download_button("Download report as TXT", export_report(project, notes, candidate_df), f"{project.stem}_report.txt", "text/plain", width="stretch")


if __name__ == "__main__":
    main()


import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.ndimage import gaussian_filter, uniform_filter1d, maximum_filter


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


@dataclass
class CandidateResult:
    disturbance_boxes: pd.DataFrame
    hyperbola_points: pd.DataFrame
    summary_table: pd.DataFrame
    notes: list[str]


# ---------- AI helpers ----------

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
        return bool(st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        return False


def call_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful GPR teaching assistant. Do not claim to identify graves. Speak in terms of possible reflectors, disturbed zones, and review targets only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


# ---------- File loading ----------

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
            elif "," in preview_text or "\t" in preview_text:
                likely_type = "delimited_text"
                notes = "Looks like delimited text."
            else:
                likely_type = "plain_text"
                notes = "Looks like plain text."
    else:
        if ext == ".dzt":
            likely_type = "gssi_dzt"
            notes = "Binary file with .dzt extension; likely GSSI radar data."
        elif ext == ".dzg":
            likely_type = "gssi_dzg"
            notes = "Likely GSSI GPS companion."
        elif ext == ".dzx":
            likely_type = "gssi_dzx"
            notes = "Likely GSSI settings companion."
        elif ext == ".dza":
            likely_type = "gssi_dza"
            notes = "Likely GSSI auxiliary companion."
        else:
            likely_type = "binary_unknown"
            notes = "Binary file but format not yet supported."
    return FileInspection(name, ext, size, is_binary, likely_type, notes, preview_text)


def parse_dzt(name: str, content: bytes) -> GPRData:
    if len(content) < 1024:
        raise ValueError("File too small for valid DZT header.")
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
        raise ValueError(f"Invalid samples/trace value: {rh_nsamp}")

    if rh_bits == 8:
        dtype, bytes_per_sample = np.uint8, 1
    elif rh_bits == 16:
        dtype, bytes_per_sample = np.uint16, 2
    else:
        raise ValueError(f"Unsupported DZT bit depth: {rh_bits}")

    trace_size = rh_nsamp * bytes_per_sample
    n_traces = len(data) // trace_size
    if n_traces == 0:
        raise ValueError("No usable traces found in DZT.")
    usable = n_traces * trace_size
    raw = np.frombuffer(data[:usable], dtype=dtype).reshape(n_traces, rh_nsamp)
    centered = raw.astype(np.float32) - rh_zero

    time_axis = np.linspace(0, rhf_range, rh_nsamp) if rhf_range and rhf_range > 0 else np.arange(rh_nsamp)
    distance_axis = np.arange(n_traces) / rhf_spm if rhf_spm and rhf_spm > 0 else np.arange(n_traces)

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
    }
    return GPRData("dzt", name, centered, time_axis, distance_axis, metadata=metadata, raw_bytes=content)


def _numeric_matrix_from_df(df: pd.DataFrame) -> np.ndarray:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").dropna(axis=0, how="all")
    arr = numeric.to_numpy(dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("No usable numeric matrix found.")
    return arr


def parse_csv(name: str, content: bytes) -> GPRData:
    df = pd.read_csv(io.BytesIO(content), header=None)
    traces = _numeric_matrix_from_df(df)
    return GPRData("csv", name, traces=traces, time_axis=np.arange(traces.shape[1]), distance_axis=np.arange(traces.shape[0]), metadata={"n_traces": int(traces.shape[0]), "samples_per_trace": int(traces.shape[1])})


def parse_text(name: str, content: bytes, forced_type: str = "text") -> GPRData:
    text = content.decode("utf-8", errors="ignore")
    gps = {"note": "NMEA GPS strings detected"} if "$GPGGA" in text or "$GNGGA" in text else None
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
            numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="all")
            if not numeric_df.empty:
                traces = numeric_df.to_numpy(dtype=float)
                break
        except Exception:
            continue
    return GPRData(forced_type, name, traces=traces, time_axis=np.arange(traces.shape[1]) if traces is not None else None, distance_axis=np.arange(traces.shape[0]) if traces is not None else None, gps=gps, metadata={"parsed_delimiter": delimiter_used, "length_chars": len(text), "preview": text[:250]}, raw_bytes=content)


def parse_dza(name: str, content: bytes) -> GPRData:
    inspection = inspect_uploaded_file(name, content)
    if inspection.is_binary:
        return GPRData("dza", name, metadata={"note": "Binary DZA detected; parsing not implemented yet.", "size_bytes": len(content)}, raw_bytes=content)
    return parse_text(name, content, forced_type="dza")


def load_uploaded_file(uploaded_file) -> Tuple[Optional[GPRData], Optional[FileInspection]]:
    name = uploaded_file.name
    content = uploaded_file.getvalue()
    ext = Path(name).suffix.lower()
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


# ---------- Synthetic raw training data ----------

def build_synthetic_demo_gpr(style: str = "generic") -> GPRData:
    rng = np.random.default_rng(24 if style == "generic" else 37)
    n_traces, n_samples = 180, 320
    t = np.linspace(0, 120, n_samples)
    x = np.linspace(0, 18, n_traces)

    traces = rng.normal(0, 8, size=(n_traces, n_samples))

    # direct wave / ringing
    for i in range(n_traces):
        idx = 16 + int(1.5 * np.sin(i / 17))
        traces[i, idx:idx+4] += 65

    # shallow and deeper banding
    for center, amp, width in [(55, 22, 2), (95, 18, 3), (150, 14, 4)]:
        band = np.exp(-0.5 * ((np.arange(n_samples) - center) / width) ** 2)
        mod = 0.7 + 0.3 * np.sin(np.linspace(0, 5 * np.pi, n_traces))
        traces += mod[:, None] * amp * band[None, :]

    # generic objects
    if style == "generic":
        for center, apex, scale, amp in [(55, 70, 0.06, 120), (118, 115, 0.045, 110)]:
            for i in range(n_traces):
                idx = int(apex + scale * (i - center) ** 2)
                if 1 <= idx < n_samples - 1:
                    traces[i, idx-1:idx+2] += amp
        for i in range(85, 112):
            traces[i, 100:165] *= 0.72
            traces[i, 42:90] += rng.normal(3, 2, size=48)
    else:
        # graveyard-like raw field texture: disturbed shafts more than textbook targets
        shaft_ranges = [(34, 49), (78, 92), (121, 135)]
        for left, right in shaft_ranges:
            for i in range(left, right):
                shift = int(rng.integers(-4, 5))
                traces[i] = np.roll(traces[i], shift)
                traces[i, 42:155] *= 0.78
                traces[i, 65:145] += gaussian_filter(rng.normal(0, 6, n_samples), 2)[65:145]
                # break shallow band coherence
                traces[i, 50:64] -= 12 * np.hanning(14)
        # a couple faint hyperbolic objects
        for center, apex, scale, amp in [(43, 88, 0.038, 68), (129, 104, 0.05, 74)]:
            for i in range(n_traces):
                idx = int(apex + scale * (i - center) ** 2)
                if 1 <= idx < n_samples - 1:
                    traces[i, idx-1:idx+2] += amp * (0.8 + 0.2 * rng.random())
        # ambiguous disturbed zone
        for i in range(146, 166):
            traces[i, 70:150] *= 0.84
            traces[i, 86:124] += rng.normal(0, 5, size=38)

    attenuation = np.linspace(1.0, 0.42, n_samples)
    traces = traces * attenuation[None, :]
    traces += gaussian_filter(rng.normal(0, 4, size=traces.shape), sigma=(1.2, 0.7))

    desc = "Synthetic generic line with reflectors and objects." if style == "generic" else "Synthetic cemetery-style raw line with disturbed shafts, subtle attenuation changes, and faint hyperbolic targets."
    return GPRData(
        file_type="demo",
        source_file=f"{style}_demo_line.dzt",
        traces=traces.astype(np.float32),
        time_axis=t,
        distance_axis=x,
        metadata={"samples_per_trace": n_samples, "n_traces": n_traces, "range_ns": float(t[-1]), "demo_mode": True, "description": desc},
    )


def build_projects_from_uploads(uploaded_files, demo_mode: str = "None") -> Dict[str, RadanProject]:
    projects: Dict[str, RadanProject] = {}
    if demo_mode == "Generic synthetic line":
        project = RadanProject(stem="generic_demo")
        project.dzt = build_synthetic_demo_gpr("generic")
        projects[project.stem] = project
    elif demo_mode == "Graveyard training line":
        project = RadanProject(stem="graveyard_demo")
        project.dzt = build_synthetic_demo_gpr("graveyard")
        projects[project.stem] = project

    for uploaded_file in uploaded_files:
        stem = Path(uploaded_file.name).stem
        project = projects.setdefault(stem, RadanProject(stem=stem))
        parsed, inspection = load_uploaded_file(uploaded_file)
        ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
        if ext in {"dzt", "dzg", "dzx", "dza"} and parsed is not None:
            setattr(project, ext, parsed)
            project.files[ext] = uploaded_file.name
        elif parsed is not None and ext in {"csv", "txt", "asc"}:
            project.dzt = parsed
            project.files[ext] = uploaded_file.name
        else:
            if inspection is None:
                inspection = inspect_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
            project.other_files[uploaded_file.name] = inspection
    return projects


# ---------- Processing ----------

def dewow_matrix(data: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    baseline = uniform_filter1d(data, size=window, axis=1, mode="nearest")
    return data - baseline


def background_remove(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=0, keepdims=True)


def apply_gain(data: np.ndarray, mode: str, strength: float) -> np.ndarray:
    n_samples = data.shape[1]
    if mode == "None":
        return data.copy()
    scale = np.linspace(1.0, 1.0 + strength, n_samples)
    if mode == "Exponential":
        scale = np.exp(np.linspace(0.0, strength, n_samples))
    return data * scale[None, :]


def normalize_traces(data: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(data), axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return data / denom


def process_gpr_data(gpr_data: GPRData, cfg: Dict[str, Any]) -> GPRData:
    raw = np.asarray(gpr_data.traces, dtype=float)
    trace_start, trace_end = cfg["trace_crop"]
    sample_start, sample_end = cfg["sample_crop"]
    data = raw[trace_start:trace_end, sample_start:sample_end].copy()

    notes = []
    if cfg["time_zero_shift"] != 0:
        data = np.roll(data, -int(cfg["time_zero_shift"]), axis=1)
        notes.append(f"Time-zero shift {cfg['time_zero_shift']} samples")
    if cfg["dewow_on"]:
        data = dewow_matrix(data, cfg["dewow_window"])
        notes.append(f"Dewow window {cfg['dewow_window']}")
    if cfg["background_on"]:
        data = background_remove(data)
        notes.append("Background removal")
    data = apply_gain(data, cfg["gain_mode"], cfg["gain_strength"])
    if cfg["gain_mode"] != "None":
        notes.append(f"{cfg['gain_mode']} gain {cfg['gain_strength']}")
    if cfg["normalize_on"]:
        data = normalize_traces(data)
        notes.append("Per-trace normalization")

    time_axis = None if gpr_data.time_axis is None else gpr_data.time_axis[sample_start:sample_end]
    distance_axis = None if gpr_data.distance_axis is None else gpr_data.distance_axis[trace_start:trace_end]

    meta = dict(gpr_data.metadata)
    meta["processing_notes"] = notes
    return GPRData("processed", gpr_data.source_file, data.astype(np.float32), time_axis, distance_axis, metadata=meta)


# ---------- Candidate finding ----------

def _local_variance(data: np.ndarray, size_t: int = 9, size_s: int = 13) -> np.ndarray:
    mu = maximum_filter(data, size=(1,1))  # dummy to keep scipy import useful for max_filter
    mu = uniform_filter1d(uniform_filter1d(data, size=size_t, axis=0, mode="nearest"), size=size_s, axis=1, mode="nearest")
    sq = uniform_filter1d(uniform_filter1d(data**2, size=size_t, axis=0, mode="nearest"), size=size_s, axis=1, mode="nearest")
    var = np.maximum(sq - mu**2, 0)
    return var


def find_candidate_features(gpr_data: GPRData, processed: GPRData, top_n: int = 8) -> CandidateResult:
    raw = np.asarray(gpr_data.traces, dtype=float)
    proc = np.asarray(processed.traces, dtype=float)
    n_traces, n_samples = proc.shape

    # disturbance score: variance + column-level discontinuity
    varmap = _local_variance(proc, 9, 17)
    col_score = np.mean(varmap[:, max(8, int(n_samples*0.15)):max(20, int(n_samples*0.65))], axis=1)
    col_score = gaussian_filter(col_score, sigma=2)
    diff_col = np.abs(np.gradient(np.mean(proc, axis=1)))
    shaft_score = col_score / (np.nanmax(col_score) + 1e-9) + 0.7 * diff_col / (np.nanmax(diff_col) + 1e-9)

    # hyperbola-ish apexes: strong local maxima, not too shallow
    amp = np.abs(proc)
    smooth = gaussian_filter(amp, sigma=(1.3, 1.3))
    localmax = smooth == maximum_filter(smooth, size=(9, 15))
    time_mask = np.zeros_like(smooth, dtype=bool)
    start = max(20, int(n_samples * 0.12))
    end = max(start + 10, int(n_samples * 0.72))
    time_mask[:, start:end] = True
    pts = np.argwhere(localmax & time_mask & (smooth > np.percentile(smooth, 98.7)))

    hyper_rows = []
    for tr, smp in pts:
        left = max(0, tr - 6)
        right = min(n_traces, tr + 7)
        window = smooth[left:right, max(0, smp-12):min(n_samples, smp+13)]
        flank_left = smooth[left:tr, smp].mean() if tr > left else 0
        flank_right = smooth[tr+1:right, smp].mean() if tr + 1 < right else 0
        local = smooth[tr, smp]
        symmetry = 1.0 - abs(flank_left - flank_right) / max(local, 1e-9)
        score = float(local * max(symmetry, 0))
        hyper_rows.append({"trace_idx": int(tr), "sample_idx": int(smp), "score": score})

    hyper_df = pd.DataFrame(hyper_rows)
    if not hyper_df.empty:
        hyper_df = hyper_df.sort_values("score", ascending=False)
        kept = []
        for _, row in hyper_df.iterrows():
            if all(abs(row["trace_idx"] - k["trace_idx"]) > 8 or abs(row["sample_idx"] - k["sample_idx"]) > 14 for k in kept):
                kept.append(row.to_dict())
            if len(kept) >= top_n:
                break
        hyper_df = pd.DataFrame(kept)
    else:
        hyper_df = pd.DataFrame(columns=["trace_idx", "sample_idx", "score"])

    # disturbance boxes from shaft_score peaks
    thresh = np.percentile(shaft_score, 88)
    mask = shaft_score >= thresh
    segments = []
    start_idx = None
    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            if i - start_idx >= 4:
                segments.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None and len(mask) - start_idx >= 4:
        segments.append((start_idx, len(mask) - 1))

    box_rows = []
    for left, right in segments[:top_n]:
        sub = np.abs(proc[left:right+1, start:end])
        if sub.size == 0:
            continue
        profile = np.mean(sub, axis=0)
        peak_s = int(np.argmax(profile)) + start
        # estimate vertical extent from local energy
        local = np.mean(sub, axis=0)
        box_rows.append({
            "trace_left": int(left),
            "trace_right": int(right),
            "sample_top": int(max(start, peak_s - 28)),
            "sample_bottom": int(min(n_samples - 1, peak_s + 38)),
            "score": float(np.mean(shaft_score[left:right+1])),
            "type": "Disturbed zone",
        })

    disturb_df = pd.DataFrame(box_rows)
    summary_rows = []
    for _, row in disturb_df.iterrows():
        summary_rows.append({
            "Type": row["type"],
            "Trace Range": f"{row['trace_left']}–{row['trace_right']}",
            "Sample Range": f"{row['sample_top']}–{row['sample_bottom']}",
            "Score": round(float(row["score"]), 3),
            "Review Note": "Possible disturbed or low-coherence column",
        })
    for _, row in hyper_df.iterrows():
        summary_rows.append({
            "Type": "Possible hyperbola apex",
            "Trace Range": str(int(row["trace_idx"])),
            "Sample Range": str(int(row["sample_idx"])),
            "Score": round(float(row["score"]), 3),
            "Review Note": "Possible compact target / hyperbolic apex",
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Score", ascending=False).reset_index(drop=True) if summary_rows else pd.DataFrame(columns=["Type", "Trace Range", "Sample Range", "Score", "Review Note"])

    notes = [
        "Candidate overlays are review aids, not detections.",
        "Disturbed zones reflect locally unusual texture, attenuation, or reflector continuity.",
        "Hyperbola markers reflect local maxima with rough symmetry, not confirmed buried objects.",
    ]
    return CandidateResult(disturb_df, hyper_df, summary_df, notes)


# ---------- Plotting ----------

def _axes_for(gpr_data: GPRData, use_distance: bool):
    data = np.asarray(gpr_data.traces)
    n_traces, n_samples = data.shape
    if use_distance and gpr_data.distance_axis is not None and len(gpr_data.distance_axis) == n_traces:
        x = gpr_data.distance_axis
        xlabel = "Distance"
    else:
        x = np.arange(n_traces)
        xlabel = "Trace Number"
    if gpr_data.time_axis is not None and len(gpr_data.time_axis) == n_samples:
        y = gpr_data.time_axis
        ylabel = "Time (ns)"
    else:
        y = np.arange(n_samples)
        ylabel = "Sample"
    return x, y, xlabel, ylabel


def make_radargram_figure(gpr_data: GPRData, use_distance: bool, clip_low: float, clip_high: float, invert_y: bool, cmap: str, candidates: Optional[CandidateResult] = None):
    data = np.asarray(gpr_data.traces)
    x, y, xlabel, ylabel = _axes_for(gpr_data, use_distance)
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    extent = [x[0], x[-1], y[-1], y[0]] if invert_y else [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.imshow(data.T, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Radargram: {gpr_data.source_file}")

    if candidates is not None:
        for _, row in candidates.disturbance_boxes.iterrows():
            xl = x[int(row["trace_left"])]
            xr = x[min(int(row["trace_right"]), len(x)-1)]
            yt = y[int(row["sample_top"])]
            yb = y[min(int(row["sample_bottom"]), len(y)-1)]
            x0, width = min(xl, xr), abs(xr - xl)
            y0, height = min(yt, yb), abs(yb - yt)
            rect = Rectangle((x0, y0), width if width > 0 else 1, height if height > 0 else 1, linewidth=1.5, edgecolor="deepskyblue", facecolor="none")
            ax.add_patch(rect)
        if not candidates.hyperbola_points.empty:
            hx = [x[int(i)] for i in candidates.hyperbola_points["trace_idx"]]
            hy = [y[int(i)] for i in candidates.hyperbola_points["sample_idx"]]
            ax.scatter(hx, hy, s=35, c="yellow", marker="o", edgecolors="black", linewidths=0.6)
    fig.tight_layout()
    return fig


def make_trace_figure(raw_data: GPRData, processed_data: Optional[GPRData], trace_index: int):
    raw = np.asarray(raw_data.traces)
    trace_index = max(0, min(trace_index, raw.shape[0] - 1))
    if raw_data.time_axis is not None and len(raw_data.time_axis) == raw.shape[1]:
        y = raw_data.time_axis
        ylabel = "Time (ns)"
    else:
        y = np.arange(raw.shape[1])
        ylabel = "Sample"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(raw[trace_index, :], y, label="Raw", lw=1.1)
    if processed_data is not None:
        p = np.asarray(processed_data.traces)
        trace_index = min(trace_index, p.shape[0] - 1)
        py = processed_data.time_axis if processed_data.time_axis is not None and len(processed_data.time_axis) == p.shape[1] else np.arange(p.shape[1])
        ax.plot(p[trace_index, :], py, label="Processed", lw=1.1)
    ax.set_title(f"Trace {trace_index}")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    return fig


# ---------- Export / reporting ----------

def traces_to_csv_bytes(gpr_data: GPRData) -> bytes:
    return pd.DataFrame(gpr_data.traces).to_csv(index=False).encode("utf-8")


def report_text(project: RadanProject, raw: GPRData, processed: Optional[GPRData], candidates: Optional[CandidateResult], ai_text: str) -> str:
    lines = [
        "GPR Review Report",
        f"Project: {project.stem}",
        f"Source: {raw.source_file}",
        f"Raw shape: {None if raw.traces is None else raw.traces.shape}",
        f"Metadata: {raw.metadata}",
        "",
        "Processing Notes:",
    ]
    proc_notes = processed.metadata.get("processing_notes", []) if processed is not None else []
    if proc_notes:
        lines.extend([f"- {n}" for n in proc_notes])
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Candidate Notes:")
    if candidates is not None:
        lines.extend([f"- {n}" for n in candidates.notes])
        if not candidates.summary_table.empty:
            lines.append("")
            lines.append("Top Review Targets:")
            for _, row in candidates.summary_table.head(12).iterrows():
                lines.append(f"- {row['Type']} | traces {row['Trace Range']} | samples {row['Sample Range']} | score {row['Score']}")
    else:
        lines.append("- No candidate analysis run")
    if ai_text:
        lines.append("")
        lines.append("AI Summary:")
        lines.append(ai_text)
    return "\n".join(lines)


# ---------- UI ----------

def init_page():
    st.set_page_config(page_title="GPR Review Console", layout="wide")
    st.title("GPR Review Console")
    st.caption("Raw-first GPR workflow: upload raw data, process it, and review overprinted candidate highlights.")


def render_sidebar(raw_shape: Optional[tuple[int, int]]):
    st.sidebar.header("Display")
    clip_low = st.sidebar.slider("Lower clip percentile", 0.0, 20.0, 2.0, 0.5)
    clip_high = st.sidebar.slider("Upper clip percentile", 80.0, 100.0, 98.0, 0.5)
    use_distance = st.sidebar.checkbox("Use distance axis if available", value=True)
    invert_y = st.sidebar.checkbox("Invert vertical axis", value=True)
    cmap = st.sidebar.selectbox("Colormap", ["gray", "seismic", "viridis", "plasma"], index=0)

    st.sidebar.header("Processing")
    n_traces, n_samples = raw_shape if raw_shape else (100, 200)
    trace_crop = st.sidebar.slider("Trace crop", 0, n_traces, (0, n_traces))
    sample_crop = st.sidebar.slider("Sample crop", 0, n_samples, (0, n_samples))
    time_zero_shift = st.sidebar.slider("Time-zero shift (samples)", -25, 25, 0)
    dewow_on = st.sidebar.checkbox("Dewow", value=True)
    dewow_window = st.sidebar.slider("Dewow window", 3, 81, 21, step=2)
    background_on = st.sidebar.checkbox("Background removal", value=True)
    gain_mode = st.sidebar.selectbox("Gain mode", ["None", "Linear", "Exponential"], index=1)
    gain_strength = st.sidebar.slider("Gain strength", 0.0, 2.5, 0.9, 0.1)
    normalize_on = st.sidebar.checkbox("Per-trace normalization", value=False)

    st.sidebar.header("Optional AI")
    ai_enable = st.sidebar.checkbox("Enable AI summary", value=False)
    ai_prefer_local = st.sidebar.checkbox("Prefer Ollama", value=True)
    ollama_model = st.sidebar.text_input("Ollama model", value="llama3.1:8b")
    ollama_base_url = st.sidebar.text_input("Ollama URL", value="http://127.0.0.1:11434")
    openai_model = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")

    return {
        "clip_low": clip_low, "clip_high": clip_high, "use_distance": use_distance, "invert_y": invert_y, "cmap": cmap,
        "trace_crop": trace_crop, "sample_crop": sample_crop, "time_zero_shift": time_zero_shift,
        "dewow_on": dewow_on, "dewow_window": dewow_window, "background_on": background_on,
        "gain_mode": gain_mode, "gain_strength": gain_strength, "normalize_on": normalize_on,
        "ai_enable": ai_enable, "ai_prefer_local": ai_prefer_local, "ollama_model": ollama_model,
        "ollama_base_url": ollama_base_url, "openai_model": openai_model
    }


def render_loader():
    st.subheader("1. Load raw data")
    demo_mode = st.selectbox("Training / demo mode", ["None", "Generic synthetic line", "Graveyard training line"])
    uploaded = st.file_uploader(
        "Upload raw project files",
        type=["dzt", "csv", "txt", "asc", "dzg", "dzx", "dza"],
        accept_multiple_files=True,
        help="Upload raw GPR data first. Companion files are optional.",
    )
    return demo_mode, uploaded


def render_project_inventory(projects: Dict[str, RadanProject]) -> str:
    st.subheader("2. Project inventory")
    rows = []
    for stem in sorted(projects.keys()):
        p = projects[stem]
        rows.append({
            "Project": stem,
            "Primary data": "✓" if p.dzt is not None else "✗",
            "DZG": "✓" if p.dzg is not None else "✗",
            "DZX": "✓" if p.dzx is not None else "✗",
            "DZA": "✓" if p.dza is not None else "✗",
            "Other files": len(p.other_files),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    return st.selectbox("Choose project", sorted(projects.keys()))


def render_metadata(gpr_data: GPRData):
    st.markdown("### Metadata")
    items = [{"Field": k, "Value": str(v)} for k, v in gpr_data.metadata.items()]
    items.insert(0, {"Field": "Source file", "Value": gpr_data.source_file})
    items.insert(1, {"Field": "File type", "Value": gpr_data.file_type})
    if gpr_data.traces is not None:
        items.insert(2, {"Field": "Trace matrix shape", "Value": str(gpr_data.traces.shape)})
    st.dataframe(pd.DataFrame(items), width="stretch", hide_index=True)


def generate_ai_summary(raw: GPRData, processed: GPRData, candidates: CandidateResult, cfg: Dict[str, Any]) -> tuple[str, str]:
    prompt = (
        "Summarize this GPR run for a student. Do not claim any grave detection. "
        "Describe possible disturbed zones, possible compact targets, and what a human should review next.\n\n"
        f"Source file: {raw.source_file}\n"
        f"Raw shape: {raw.traces.shape if raw.traces is not None else None}\n"
        f"Processing notes: {processed.metadata.get('processing_notes', [])}\n"
        f"Top review targets:\n{candidates.summary_table.head(8).to_string(index=False) if not candidates.summary_table.empty else 'None'}\n"
        f"Notes: {'; '.join(candidates.notes)}"
    )
    if cfg["ai_prefer_local"] and ollama_available(cfg["ollama_base_url"]):
        return call_ollama(prompt, cfg["ollama_model"], cfg["ollama_base_url"]), f"Ollama ({cfg['ollama_model']})"
    if openai_available():
        return call_openai(prompt, cfg["openai_model"]), f"OpenAI ({cfg['openai_model']})"
    if ollama_available(cfg["ollama_base_url"]):
        return call_ollama(prompt, cfg["ollama_model"], cfg["ollama_base_url"]), f"Ollama ({cfg['ollama_model']})"
    raise RuntimeError("No AI provider available.")


def main():
    init_page()
    demo_mode, uploaded_files = render_loader()

    if demo_mode == "None" and not uploaded_files:
        st.info("Upload raw data files, or choose a training/demo mode.")
        st.stop()

    projects = build_projects_from_uploads(uploaded_files or [], demo_mode=demo_mode)
    if not projects:
        st.warning("No readable projects were assembled.")
        st.stop()

    selected_stem = render_project_inventory(projects)
    project = projects[selected_stem]
    primary = project.dzt or project.dzg or project.dzx or project.dza
    if primary is None or primary.traces is None:
        st.warning("This project has no readable primary trace matrix yet.")
        st.stop()

    cfg = render_sidebar(primary.traces.shape)
    processed = process_gpr_data(primary, cfg)
    candidates = find_candidate_features(primary, processed, top_n=10)

    tabs = st.tabs(["Load", "Raw", "Processed", "Overlay / Review", "Single Trace", "Companions", "Report"])

    with tabs[0]:
        render_metadata(primary)
        st.markdown("### Workflow")
        st.write("1. Upload raw data")
        st.write("2. Inspect raw radargram")
        st.write("3. Process raw data")
        st.write("4. Review overprinted candidate highlights")
        st.write("5. Export notes/report")
        if uploaded_files:
            st.caption("Uploaded files: " + ", ".join([f.name for f in uploaded_files]))

    with tabs[1]:
        st.markdown("### Raw radargram")
        fig = make_radargram_figure(primary, cfg["use_distance"], cfg["clip_low"], cfg["clip_high"], cfg["invert_y"], cfg["cmap"], candidates=None)
        st.pyplot(fig, clear_figure=True, width="stretch")

    with tabs[2]:
        st.markdown("### Processed radargram")
        fig = make_radargram_figure(processed, cfg["use_distance"], cfg["clip_low"], cfg["clip_high"], cfg["invert_y"], cfg["cmap"], candidates=None)
        st.pyplot(fig, clear_figure=True, width="stretch")
        st.caption("Processing notes: " + "; ".join(processed.metadata.get("processing_notes", [])))

    with tabs[3]:
        st.markdown("### Overprinted review targets")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raw + overlays**")
            fig = make_radargram_figure(primary, cfg["use_distance"], cfg["clip_low"], cfg["clip_high"], cfg["invert_y"], cfg["cmap"], candidates=candidates)
            st.pyplot(fig, clear_figure=True, width="stretch")
        with c2:
            st.markdown("**Processed + overlays**")
            fig = make_radargram_figure(processed, cfg["use_distance"], cfg["clip_low"], cfg["clip_high"], cfg["invert_y"], cfg["cmap"], candidates=candidates)
            st.pyplot(fig, clear_figure=True, width="stretch")
        st.markdown("**Candidate review table**")
        st.dataframe(candidates.summary_table, width="stretch", hide_index=True)
        for n in candidates.notes:
            st.caption(n)

    with tabs[4]:
        st.markdown("### Single trace comparison")
        idx = st.slider("Trace index", 0, max(0, primary.traces.shape[0] - 1), 0)
        fig = make_trace_figure(primary, processed, idx)
        st.pyplot(fig, clear_figure=True)

    with tabs[5]:
        st.markdown("### Companion files")
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
            rows = [{"File": name, "Type guess": ins.likely_type, "Binary": ins.is_binary, "Size": ins.size_bytes, "Notes": ins.notes} for name, ins in project.other_files.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tabs[6]:
        st.markdown("### Report / export")
        ai_text = ""
        provider = ""
        if cfg["ai_enable"]:
            try:
                ai_text, provider = generate_ai_summary(primary, processed, candidates, cfg)
                st.markdown(f"**AI summary source:** {provider}")
                st.write(ai_text)
            except Exception as e:
                st.warning(f"AI summary unavailable: {e}")
        text = report_text(project, primary, processed, candidates, ai_text)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download raw trace matrix CSV", data=traces_to_csv_bytes(primary), file_name=f"{Path(primary.source_file).stem}_raw.csv", mime="text/csv", width="stretch")
        with c2:
            st.download_button("Download processed trace matrix CSV", data=traces_to_csv_bytes(processed), file_name=f"{Path(primary.source_file).stem}_processed.csv", mime="text/csv", width="stretch")
        with c3:
            st.download_button("Download report TXT", data=text.encode("utf-8"), file_name=f"{project.stem}_report.txt", mime="text/plain", width="stretch")
        st.text_area("Report preview", text, height=300)


if __name__ == "__main__":
    main()

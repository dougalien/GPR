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
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter


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
        "usable_data_bytes": int(usable),
        "leftover_bytes": int(len(data) - usable),
    }
    return GPRData("dzt", name, centered, time_axis, distance_axis, metadata=metadata)


def parse_csv(name: str, content: bytes) -> GPRData:
    df = pd.read_csv(io.BytesIO(content))
    numeric_df = df.select_dtypes(include=[np.number])
    traces = numeric_df.to_numpy() if not numeric_df.empty else None
    return GPRData("csv", name, traces=traces, metadata={"columns": list(df.columns), "n_rows": int(len(df)), "n_numeric_columns": int(len(numeric_df.columns))})


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
            if not df.empty:
                numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
                if not numeric_df.empty:
                    traces = numeric_df.to_numpy()
                    break
        except Exception:
            continue
    return GPRData(forced_type, name, traces=traces, gps=gps, metadata={"length_chars": len(text), "preview": text[:300], "parsed_delimiter": delimiter_used}, raw_bytes=content)


def parse_dza(name: str, content: bytes) -> GPRData:
    inspection = inspect_uploaded_file(name, content)
    if inspection.is_binary:
        return GPRData("dza", name, metadata={"note": "DZA detected. Binary auxiliary line trace parsing not implemented yet.", "size_bytes": len(content), "likely_type": inspection.likely_type}, raw_bytes=content)
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
    return GPRData("demo", "demo_line.dzt", traces, t, np.linspace(0, 18, n_traces), metadata={"demo_mode": True, "description": "Synthetic generic training line.", "n_traces": n_traces, "samples_per_trace": n_samples, "range_ns": float(t[-1])})


def build_graveyard_demo_gpr() -> GPRData:
    n_traces = 220
    n_samples = 340
    t = np.linspace(0, 140, n_samples)
    x = np.linspace(0, 22, n_traces)
    traces = np.random.normal(0, 8, size=(n_traces, n_samples))
    shallow = 20 + 3 * np.sin(np.linspace(0, 5, n_traces))
    for i in range(n_traces):
        idx = int(shallow[i])
        traces[i, max(0, idx - 1):min(n_samples, idx + 2)] += 55
    for center in [55, 105, 155]:
        width = 10
        for i in range(max(0, center - width), min(n_traces, center + width + 1)):
            depth = int(36 + 2 * np.sin((i - center) / 2.0))
            traces[i, depth:depth + 80] *= 0.72
            if 0 <= depth < n_samples:
                traces[i, max(0, depth - 1):min(n_samples, depth + 2)] -= 35
    for center, apex, scale, amp in [(58, 68, 0.09, 65), (158, 78, 0.08, 55)]:
        for i in range(max(0, center - 18), min(n_traces, center + 19)):
            idx = int(apex + scale * (i - center) ** 2)
            if 1 <= idx < n_samples - 1:
                traces[i, idx - 1:idx + 2] += amp
    for center in [30, 88, 190]:
        for i in range(max(0, center - 7), min(n_traces, center + 8)):
            depth = int(52 + 4 * np.cos((i - center) / 3))
            traces[i, depth:depth + 45] *= 0.83
    traces += np.linspace(16, -6, n_samples)[None, :]
    traces += gaussian_filter(np.random.normal(0, 4, size=traces.shape), sigma=(6, 2))
    traces *= np.linspace(1.0, 0.42, n_samples)[None, :]
    return GPRData("demo", "graveyard_training_line.dzt", traces, t, x, metadata={"demo_mode": True, "description": "Synthetic graveyard training line with subtle disturbed zones and faint hyperbolic targets.", "n_traces": n_traces, "samples_per_trace": n_samples, "range_ns": float(t[-1])})


def build_projects_from_uploads(uploaded_files, demo_mode: str = "None") -> Dict[str, RadanProject]:
    projects: Dict[str, RadanProject] = {}
    if demo_mode != "None":
        project = RadanProject(stem="demo_project")
        project.dzt = build_graveyard_demo_gpr() if demo_mode == "Graveyard training line" else build_synthetic_demo_gpr()
        project.dzg = GPRData("dzg", "demo_line.dzg", gps={"track_preview": [{"trace": 0, "lat": 42.52, "lon": -70.89}, {"trace": 80, "lat": 42.5208, "lon": -70.8892}, {"trace": 180, "lat": 42.5215, "lon": -70.8885}]}, metadata={"demo_mode": True})
        project.dzx = GPRData("dzx", "demo_line.dzx", metadata={"demo_mode": True, "gain": "auto", "time_zero_shift": 0})
        project.dza = GPRData("dza", "demo_line.dza", metadata={"demo_mode": True, "description": "Synthetic auxiliary companion."})
        projects[project.stem] = project
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


def dewow(traces: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    return traces - uniform_filter(traces, size=(1, window), mode="nearest")


def background_remove(traces: np.ndarray) -> np.ndarray:
    return traces - np.mean(traces, axis=0, keepdims=True)


def apply_gain(traces: np.ndarray, mode: str, strength: float) -> np.ndarray:
    ns = traces.shape[1]
    factor = np.ones(ns)
    s = max(float(strength), 0.0)
    if mode == "Linear":
        factor = 1.0 + s * np.linspace(0, 1, ns)
    elif mode == "Exponential":
        factor = np.exp(s * np.linspace(0, 1, ns))
    return traces * factor[np.newaxis, :]


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(traces), axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    return traces / peak


def crop_traces(traces: np.ndarray, trace_slice: tuple[int, int], sample_slice: tuple[int, int]) -> np.ndarray:
    return traces[trace_slice[0]:trace_slice[1], sample_slice[0]:sample_slice[1]]


def process_gpr(gpr_data: GPRData, controls: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any], list[str]]:
    data = np.asarray(gpr_data.traces, dtype=float).copy()
    notes = []
    if controls["trace_crop"] != (0, data.shape[0]):
        notes.append(f"Trace crop: {controls['trace_crop'][0]} to {controls['trace_crop'][1]}")
    if controls["sample_crop"] != (0, data.shape[1]):
        notes.append(f"Sample crop: {controls['sample_crop'][0]} to {controls['sample_crop'][1]}")
    data = crop_traces(data, controls["trace_crop"], controls["sample_crop"])
    if controls["time_zero_shift"] != 0:
        data = np.roll(data, -int(controls["time_zero_shift"]), axis=1)
        notes.append(f"Time-zero shift: {controls['time_zero_shift']} samples")
    if controls["dewow_on"]:
        data = dewow(data, controls["dewow_window"])
        notes.append(f"Dewow applied, window {controls['dewow_window']}")
    if controls["background_remove_on"]:
        data = background_remove(data)
        notes.append("Background removal applied")
    if controls["gain_mode"] != "None":
        data = apply_gain(data, controls["gain_mode"], controls["gain_strength"])
        notes.append(f"{controls['gain_mode']} gain, strength {controls['gain_strength']:.2f}")
    if controls["normalize_traces_on"]:
        data = normalize_traces(data)
        notes.append("Per-trace normalization applied")
    meta = {
        "n_traces": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
    }
    return data, meta, notes


def compute_candidate_layers(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    abs_data = np.abs(traces)
    local_var = gaussian_filter((traces - gaussian_filter(traces, sigma=2)) ** 2, sigma=3)
    lateral_mean = uniform_filter(abs_data, size=(21, 1), mode="nearest")
    disturbance = gaussian_filter(np.abs(abs_data - lateral_mean), sigma=2)
    apex = maximum_filter(abs_data, size=(11, 7))
    below = np.roll(abs_data, -8, axis=1)
    sides = (np.roll(abs_data, -5, axis=0) + np.roll(abs_data, 5, axis=0)) / 2.0
    hyperbola = gaussian_filter(np.clip((below + sides) - 1.35 * abs_data, 0, None), sigma=2)
    disturbance_score = disturbance * (0.6 + 0.4 * (local_var / (np.nanmax(local_var) + 1e-9)))
    hyperbola_score = hyperbola * (abs_data >= 0.6 * apex)
    d_thresh = np.percentile(disturbance_score, 99.2)
    h_thresh = np.percentile(hyperbola_score, 99.6)
    d_idx = np.argwhere(disturbance_score >= d_thresh)
    h_idx = np.argwhere(hyperbola_score >= h_thresh)
    rows = []
    for label, arr, score_grid in [("disturbance", d_idx, disturbance_score), ("hyperbola", h_idx, hyperbola_score)]:
        for trace_idx, sample_idx in arr[:20]:
            rows.append({"Type": label, "Trace": int(trace_idx), "Sample": int(sample_idx), "Score": float(score_grid[trace_idx, sample_idx])})
    table = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["Type", "Trace", "Sample", "Score"])
    return disturbance_score, hyperbola_score, table


def make_radargram_figure(data: np.ndarray, gpr_data: GPRData, use_distance: bool, clip_low: float, clip_high: float, invert_y: bool, cmap: str, title: str, overlay_table: pd.DataFrame | None = None):
    n_traces, n_samples = data.shape
    if use_distance and gpr_data.distance_axis is not None and len(gpr_data.distance_axis) >= n_traces:
        x = gpr_data.distance_axis[:n_traces]
        xlabel = "Distance"
    else:
        x = np.arange(n_traces)
        xlabel = "Trace Number"
    if gpr_data.time_axis is not None and len(gpr_data.time_axis) >= n_samples:
        y = gpr_data.time_axis[:n_samples]
        ylabel = "Time (ns)"
    else:
        y = np.arange(n_samples)
        ylabel = "Sample"
    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    extent = [x[0], x[-1], y[-1], y[0]] if invert_y else [x[0], x[-1], y[0], y[-1]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(data.T, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if overlay_table is not None and not overlay_table.empty:
        colors = {"disturbance": "cyan", "hyperbola": "red"}
        for kind, group in overlay_table.groupby("Type"):
            xs = [x[min(int(t), len(x)-1)] for t in group["Trace"]]
            ys = [y[min(int(s), len(y)-1)] for s in group["Sample"]]
            ax.scatter(xs, ys, s=28, facecolors='none', edgecolors=colors.get(kind, 'yellow'), linewidths=1.1, label=kind)
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def make_trace_figure(raw_trace: np.ndarray, proc_trace: np.ndarray, gpr_data: GPRData, trace_index: int):
    n_samples = len(raw_trace)
    y = gpr_data.time_axis[:n_samples] if gpr_data.time_axis is not None and len(gpr_data.time_axis) >= n_samples else np.arange(n_samples)
    ylabel = "Time (ns)" if gpr_data.time_axis is not None else "Sample"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(raw_trace, y, label="Raw")
    ax.plot(proc_trace, y, label="Processed")
    ax.set_title(f"Trace {trace_index}")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    return fig


def traces_to_csv_bytes(traces: np.ndarray) -> bytes:
    return pd.DataFrame(traces).to_csv(index=False).encode("utf-8")


def metadata_text(project: RadanProject, notes: list[str], candidates: pd.DataFrame) -> str:
    lines = [f"Project: {project.stem}"]
    if project.dzt is not None:
        for k, v in project.dzt.metadata.items():
            lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Processing notes:")
    lines.extend([f"- {n}" for n in notes] if notes else ["- none"])
    lines.append("")
    lines.append("Top review candidates:")
    if candidates.empty:
        lines.append("- none")
    else:
        for _, row in candidates.head(12).iterrows():
            lines.append(f"- {row['Type']} at trace {int(row['Trace'])}, sample {int(row['Sample'])}, score {row['Score']:.3f}")
    return "\n".join(lines)


def ollama_available(base_url: str = "http://127.0.0.1:11434") -> bool:
    try:
        return requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5).ok
    except Exception:
        return False


def call_ollama(prompt: str, model: str, base_url: str = "http://127.0.0.1:11434") -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    resp = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return str(resp.json().get("response", "")).strip()


def openai_available() -> bool:
    try:
        return bool(os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        return bool(os.getenv("OPENAI_API_KEY"))


def call_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": "You are a careful GPR interpretation assistant. Be cautious and do not claim graves are confirmed."}, {"role": "user", "content": prompt}], "temperature": 0.2}
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


def generate_ai_summary(project: RadanProject, notes: list[str], candidates: pd.DataFrame, prefer_local: bool, ollama_model: str, ollama_url: str, openai_model: str) -> tuple[str, str]:
    prompt = (
        "Summarize this GPR run for students. Be concise. Explain that the app is finding candidate anomalies only, not confirming graves. "
        + metadata_text(project, notes, candidates)
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
    if not prefer_local and ollama_available(ollama_url):
        return call_ollama(prompt, ollama_model, ollama_url), f"Ollama ({ollama_model})"
    raise RuntimeError("No AI provider available. The app still works without AI.")


def main():
    st.set_page_config(page_title="GPR Control Panel", layout="wide")
    st.title("GPR Control Panel")
    st.caption("Upload raw GPR files, inspect radargrams, process them, and review candidate anomalies.")

    st.sidebar.header("Display")
    clip_low = st.sidebar.slider("Lower contrast clip percentile", 0.0, 20.0, 2.0, 0.5)
    clip_high = st.sidebar.slider("Upper contrast clip percentile", 80.0, 100.0, 98.0, 0.5)
    use_distance = st.sidebar.checkbox("Use distance axis if available", value=True)
    invert_y = st.sidebar.checkbox("Invert vertical axis", value=True)
    cmap = st.sidebar.selectbox("Colormap", ["gray", "seismic", "viridis", "plasma"], index=0)
    show_trace = st.sidebar.checkbox("Show single trace viewer", value=True)

    st.subheader("1. Load RADAN Project Files")
    demo_mode = st.selectbox("Demo mode", ["None", "Generic synthetic line", "Graveyard training line"], index=0)
    uploaded_files = st.file_uploader(
        "Upload one or more related files",
        type=["dzt", "csv", "txt", "asc", "dzg", "dzx", "dza"],
        accept_multiple_files=True,
        help="Upload matching files like line01.dzt, line01.dzg, line01.dzx, and line01.dza together.",
    )
    if demo_mode == "None" and not uploaded_files:
        st.info("Upload one or more raw files, or choose a demo mode.")
        st.stop()

    projects = build_projects_from_uploads(uploaded_files or [], demo_mode=demo_mode)
    if not projects:
        st.warning("No readable projects were assembled.")
        st.stop()

    st.subheader("2. Project Inventory")
    rows = []
    for stem in sorted(projects.keys()):
        p = projects[stem]
        rows.append({"Project": stem, "DZT": "✓" if p.dzt is not None else "✗", "DZG": "✓" if p.dzg is not None else "✗", "DZX": "✓" if p.dzx is not None else "✗", "DZA": "✓" if p.dza is not None else "✗", "Other files": len(p.other_files)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    selected_stem = st.selectbox("Choose project", sorted(projects.keys()))
    project = projects[selected_stem]
    if project.dzt is None:
        st.warning("This project has no DZT/raw radargram file yet.")
        st.stop()

    raw = np.asarray(project.dzt.traces, dtype=float)
    n_traces, n_samples = raw.shape

    st.sidebar.header("Processing")
    trace_crop = st.sidebar.slider("Trace crop", 0, n_traces, (0, n_traces))
    sample_crop = st.sidebar.slider("Sample crop", 0, n_samples, (0, n_samples))
    time_zero_shift = st.sidebar.slider("Time-zero shift (samples)", -40, 40, 0)
    dewow_on = st.sidebar.checkbox("Dewow", value=True)
    dewow_window = st.sidebar.slider("Dewow window", 3, 81, 21, 2)
    background_remove_on = st.sidebar.checkbox("Background removal", value=True)
    gain_mode = st.sidebar.selectbox("Gain", ["None", "Linear", "Exponential"], index=1)
    gain_strength = st.sidebar.slider("Gain strength", 0.0, 3.0, 1.2, 0.1)
    normalize_traces_on = st.sidebar.checkbox("Per-trace normalization", value=False)

    controls = {
        "trace_crop": trace_crop,
        "sample_crop": sample_crop,
        "time_zero_shift": time_zero_shift,
        "dewow_on": dewow_on,
        "dewow_window": dewow_window,
        "background_remove_on": background_remove_on,
        "gain_mode": gain_mode,
        "gain_strength": gain_strength,
        "normalize_traces_on": normalize_traces_on,
    }

    processed, proc_meta, notes = process_gpr(project.dzt, controls)
    disturbance, hyperbola, candidate_table = compute_candidate_layers(processed)

    tabs = st.tabs(["Load / Inventory", "Raw Radargram", "Processed Radargram", "Single Trace", "Companions", "Interpret", "Report / Export"])

    with tabs[0]:
        st.subheader("Project metadata")
        meta_rows = [{"Field": k, "Value": str(v)} for k, v in project.dzt.metadata.items()]
        st.dataframe(pd.DataFrame(meta_rows), width="stretch", hide_index=True)
        st.markdown("**Processing notes**")
        for note in notes or ["No processing applied."]:
            st.write(f"- {note}")

    with tabs[1]:
        st.pyplot(make_radargram_figure(raw, project.dzt, use_distance, clip_low, clip_high, invert_y, cmap, f"Raw Radargram: {project.dzt.source_file}", candidate_table), clear_figure=True, width="stretch")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(make_radargram_figure(processed, project.dzt, use_distance, clip_low, clip_high, invert_y, cmap, "Processed Radargram", candidate_table), clear_figure=True, width="stretch")
        with c2:
            st.pyplot(make_radargram_figure(disturbance, project.dzt, use_distance, 2.0, 99.0, invert_y, "viridis", "Disturbance Score"), clear_figure=True, width="stretch")
            st.pyplot(make_radargram_figure(hyperbola, project.dzt, use_distance, 2.0, 99.0, invert_y, "plasma", "Hyperbola Score"), clear_figure=True, width="stretch")
        st.markdown("**Top candidate review targets**")
        st.dataframe(candidate_table.head(25), width="stretch", hide_index=True)

    with tabs[3]:
        if show_trace:
            trace_idx = st.slider("Trace index", 0, max(0, processed.shape[0] - 1), min(10, processed.shape[0] - 1))
            raw_start, raw_end = trace_crop
            raw_idx = min(raw_start + trace_idx, raw.shape[0] - 1)
            raw_trace = raw[raw_idx, sample_crop[0]:sample_crop[1]]
            proc_trace = processed[trace_idx, :]
            st.pyplot(make_trace_figure(raw_trace, proc_trace, project.dzt, trace_idx), clear_figure=True)

    with tabs[4]:
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
            extra_rows = [{"File": name, "Type guess": ins.likely_type, "Binary": ins.is_binary, "Size": ins.size_bytes, "Notes": ins.notes} for name, ins in project.other_files.items()]
            st.dataframe(pd.DataFrame(extra_rows), width="stretch", hide_index=True)

    with tabs[5]:
        st.info("This tab highlights candidate anomalies for review. It does not confirm graves or subsurface identity.")
        ai_enable = st.checkbox("Enable optional AI summary", value=False)
        prefer_local = st.checkbox("Prefer Ollama first", value=True)
        ollama_model = st.text_input("Ollama model", value="llama3.1:8b")
        ollama_url = st.text_input("Ollama URL", value="http://127.0.0.1:11434")
        openai_model = st.text_input("OpenAI model", value="gpt-4.1-mini")
        if ai_enable and st.button("Generate AI summary", width="stretch"):
            try:
                text, provider = generate_ai_summary(project, notes, candidate_table, prefer_local, ollama_model, ollama_url, openai_model)
                st.success(f"Used {provider}")
                st.write(text)
                st.session_state["gpr_ai_text"] = text
            except Exception as exc:
                st.warning(str(exc))
        if candidate_table.empty:
            st.write("No strong candidates rose above the current thresholds.")
        else:
            st.write("Use the candidate table alongside the processed radargram. Disturbance targets suggest broken or unusual local texture. Hyperbola targets suggest point-like reflectors worth review.")

    with tabs[6]:
        report_text = metadata_text(project, notes, candidate_table)
        st.text_area("Run report", value=report_text, height=360)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download raw trace matrix CSV", data=traces_to_csv_bytes(raw), file_name=f"{project.stem}_raw_traces.csv", mime="text/csv", width="stretch")
        with c2:
            st.download_button("Download processed trace matrix CSV", data=traces_to_csv_bytes(processed), file_name=f"{project.stem}_processed_traces.csv", mime="text/csv", width="stretch")
        with c3:
            st.download_button("Download report TXT", data=report_text.encode("utf-8"), file_name=f"{project.stem}_report.txt", mime="text/plain", width="stretch")
        if st.session_state.get("gpr_ai_text"):
            st.markdown("**Last AI summary**")
            st.write(st.session_state["gpr_ai_text"])

if __name__ == "__main__":
    main()

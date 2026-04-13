import io
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    return GPRData("dzt", name, centered, time_axis, distance_axis, None, None, metadata)


def parse_csv(name: str, content: bytes) -> GPRData:
    df = pd.read_csv(io.BytesIO(content))
    numeric_df = df.select_dtypes(include=[np.number])
    traces = numeric_df.to_numpy() if not numeric_df.empty else None
    return GPRData(
        file_type="csv",
        source_file=name,
        traces=traces,
        metadata={
            "columns": list(df.columns),
            "n_rows": int(len(df)),
            "n_numeric_columns": int(len(numeric_df.columns)),
        },
    )


def parse_text(name: str, content: bytes, forced_type: str = "text") -> GPRData:
    text = content.decode("utf-8", errors="ignore")
    gps = None
    if "$GPGGA" in text or "$GNGGA" in text:
        gps = {"note": "NMEA GPS strings detected"}

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

    return GPRData(forced_type, name, traces, gps=gps, metadata={"length_chars": len(text), "preview": text[:300], "parsed_delimiter": delimiter_used}, raw_bytes=content)


def parse_dza(name: str, content: bytes) -> GPRData:
    inspection = inspect_uploaded_file(name, content)
    if inspection.is_binary:
        return GPRData(
            file_type="dza",
            source_file=name,
            metadata={"note": "DZA detected. Binary auxiliary line trace parsing not implemented yet.", "size_bytes": len(content), "likely_type": inspection.likely_type},
            raw_bytes=content,
        )
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


def build_synthetic_demo_gpr() -> GPRData:
    rng = np.random.default_rng(7)
    n_traces = 180
    n_samples = 320
    t = np.linspace(0, 120, n_samples)
    traces = rng.normal(0, 12, size=(n_traces, n_samples))
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
        file_type="demo",
        source_file="demo_line.dzt",
        traces=traces,
        time_axis=t,
        distance_axis=np.linspace(0, 18, n_traces),
        metadata={
            "samples_per_trace": n_samples,
            "bits_per_sample": "synthetic",
            "zero_offset": 0,
            "scans_per_meter": n_traces / 18,
            "range_ns": float(t[-1]),
            "channels": 1,
            "n_traces": n_traces,
            "demo_mode": True,
            "description": "Synthetic GPR-style demo with reflectors and hyperbolic targets.",
        },
    )


def build_projects_from_uploads(uploaded_files, use_demo: bool = False) -> Dict[str, RadanProject]:
    projects: Dict[str, RadanProject] = {}
    if use_demo:
        project = RadanProject(stem="demo_line")
        project.dzt = build_synthetic_demo_gpr()
        project.dzg = GPRData(file_type="dzg", source_file="demo_line.dzg", gps={"track_preview": [{"trace": 0, "lat": 42.5200, "lon": -70.8900}, {"trace": 60, "lat": 42.5205, "lon": -70.8895}, {"trace": 120, "lat": 42.5210, "lon": -70.8890}, {"trace": 179, "lat": 42.5215, "lon": -70.8885}]}, metadata={"demo_mode": True, "description": "Synthetic GPS companion preview."})
        project.dzx = GPRData(file_type="dzx", source_file="demo_line.dzx", metadata={"demo_mode": True, "gain": "auto", "time_zero_shift": 0})
        project.dza = GPRData(file_type="dza", source_file="demo_line.dza", metadata={"demo_mode": True, "description": "Synthetic auxiliary companion."})
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


def moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def dewow_traces(traces: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(traces, dtype=float)
    for i in range(traces.shape[0]):
        baseline = moving_average_1d(traces[i], window)
        out[i] = traces[i] - baseline
    return out


def background_remove(traces: np.ndarray) -> np.ndarray:
    column_mean = np.mean(traces, axis=0, keepdims=True)
    return traces - column_mean


def apply_gain(traces: np.ndarray, mode: str, strength: float) -> np.ndarray:
    if mode == "None" or strength <= 0:
        return traces.copy()
    n_samples = traces.shape[1]
    scale = np.linspace(1.0, 1.0 + strength, n_samples)
    if mode == "Exponential":
        scale = np.exp(np.linspace(0.0, strength, n_samples))
    return traces * scale[np.newaxis, :]


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(traces), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    return traces / max_abs


def crop_gpr(gpr_data: GPRData, traces: np.ndarray, trace_range: tuple[int, int], sample_range: tuple[int, int]) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    t0, t1 = trace_range
    s0, s1 = sample_range
    cropped = traces[t0:t1 + 1, s0:s1 + 1]
    time_axis = gpr_data.time_axis[s0:s1 + 1] if gpr_data.time_axis is not None else None
    distance_axis = gpr_data.distance_axis[t0:t1 + 1] if gpr_data.distance_axis is not None else None
    return cropped, time_axis, distance_axis


def process_gpr_data(gpr_data: GPRData, settings: Dict[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    traces = np.asarray(gpr_data.traces, dtype=float).copy()
    notes = []
    if settings["time_zero_shift"] != 0:
        traces = np.roll(traces, -int(settings["time_zero_shift"]), axis=1)
        notes.append(f"Time-zero shift applied: {int(settings['time_zero_shift'])} samples.")
    if settings["dewow_on"]:
        traces = dewow_traces(traces, int(settings["dewow_window"]))
        notes.append(f"Dewow applied with window {int(settings['dewow_window'])}.")
    if settings["background_on"]:
        traces = background_remove(traces)
        notes.append("Background removal applied.")
    if settings["trace_normalize_on"]:
        traces = normalize_traces(traces)
        notes.append("Per-trace normalization applied.")
    traces = apply_gain(traces, settings["gain_mode"], float(settings["gain_strength"]))
    if settings["gain_mode"] != "None" and float(settings["gain_strength"]) > 0:
        notes.append(f"{settings['gain_mode']} gain applied with strength {float(settings['gain_strength']):.2f}.")

    cropped, time_axis, distance_axis = crop_gpr(gpr_data, traces, settings["trace_range"], settings["sample_range"])
    if settings["trace_range"] != (0, traces.shape[0] - 1):
        notes.append(f"Trace crop: {settings['trace_range'][0]} to {settings['trace_range'][1]}.")
    if settings["sample_range"] != (0, traces.shape[1] - 1):
        notes.append(f"Sample crop: {settings['sample_range'][0]} to {settings['sample_range'][1]}.")
    return cropped, time_axis, distance_axis, notes


def heatmap_figure(traces: np.ndarray, time_axis: Optional[np.ndarray], distance_axis: Optional[np.ndarray], use_distance: bool, clip_low: float, clip_high: float, invert_y: bool, colorscale: str, title: str):
    n_traces, n_samples = traces.shape
    x = distance_axis if use_distance and distance_axis is not None and len(distance_axis) == n_traces else np.arange(n_traces)
    y = time_axis if time_axis is not None and len(time_axis) == n_samples else np.arange(n_samples)
    z = traces.T
    zmin = np.percentile(traces, clip_low)
    zmax = np.percentile(traces, clip_high)
    fig = go.Figure(go.Heatmap(x=x, y=y, z=z, zmin=zmin, zmax=zmax, colorscale=colorscale, colorbar=dict(title="Amp")))
    fig.update_layout(title=title, xaxis_title="Distance" if use_distance and distance_axis is not None else "Trace Number", yaxis_title="Time (ns)" if time_axis is not None else "Sample", height=580, margin=dict(l=20, r=20, t=55, b=20))
    fig.update_yaxes(autorange="reversed" if invert_y else True)
    return fig


def trace_figure(traces: np.ndarray, trace_index: int, time_axis: Optional[np.ndarray], title: str):
    n_traces, n_samples = traces.shape
    idx = max(0, min(trace_index, n_traces - 1))
    y = time_axis if time_axis is not None and len(time_axis) == n_samples else np.arange(n_samples)
    fig = go.Figure(go.Scatter(x=traces[idx, :], y=y, mode="lines", name=f"Trace {idx}"))
    fig.update_layout(title=title, xaxis_title="Amplitude", yaxis_title="Time (ns)" if time_axis is not None else "Sample", height=520, margin=dict(l=20, r=20, t=55, b=20))
    fig.update_yaxes(autorange="reversed")
    return fig


def traces_to_csv_bytes(gpr_data: GPRData) -> bytes:
    return pd.DataFrame(gpr_data.traces).to_csv(index=False).encode("utf-8")


def ollama_available(base_url: str = "http://127.0.0.1:11434") -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        return resp.ok
    except Exception:
        return False


def call_ollama(prompt: str, model: str, base_url: str = "http://127.0.0.1:11434") -> str:
    resp = requests.post(f"{base_url.rstrip('/')}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}, timeout=120)
    resp.raise_for_status()
    return str(resp.json().get("response", "")).strip()


def openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or getattr(st, "secrets", {}).get("OPENAI_API_KEY", ""))


def call_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": "You are a careful GPR teaching assistant. Use only the supplied context. Be concise and cautious about uncertainty."}, {"role": "user", "content": prompt}], "temperature": 0.2}
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


def ai_summary_context(project: RadanProject, raw_traces: np.ndarray, processed_traces: np.ndarray, notes: list[str]) -> str:
    raw_std = float(np.std(raw_traces))
    proc_std = float(np.std(processed_traces))
    top_energy = np.mean(np.abs(processed_traces), axis=0)
    top_idx = int(np.argmax(top_energy))
    return "\n".join([
        f"Project: {project.stem}",
        f"Source: {project.dzt.source_file if project.dzt else 'No DZT'}",
        f"Raw shape: {raw_traces.shape}",
        f"Processed shape: {processed_traces.shape}",
        f"Raw amplitude std: {raw_std:.3f}",
        f"Processed amplitude std: {proc_std:.3f}",
        f"Highest mean-energy sample index: {top_idx}",
        f"Companion files present: DZG={project.dzg is not None}, DZX={project.dzx is not None}, DZA={project.dza is not None}",
        "Processing notes: " + ("; ".join(notes) if notes else "None"),
    ])


def generate_ai_summary(project: RadanProject, raw_traces: np.ndarray, processed_traces: np.ndarray, notes: list[str], question: str, prefer_local: bool, ollama_model: str, ollama_base_url: str, openai_model: str) -> tuple[str, str]:
    prompt = (
        "Use the following GPR processing context to answer the user's request. "
        "Stay cautious, describe likely patterns rather than claiming certainty, and be concise.\n\n"
        + ai_summary_context(project, raw_traces, processed_traces, notes)
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
        return call_ollama(prompt, ollama_model, ollama_base_url), f"Ollama ({ollama_model})"
    raise RuntimeError("No AI provider is available. Core GPR analysis still works without AI.")


def init_page():
    st.set_page_config(page_title="GPR Control Panel", layout="wide")
    st.title("GPR Control Panel")
    st.caption("Grouped RADAN project viewer with processing, comparison, report, and optional local-first interpretation.")


def main():
    init_page()

    with st.sidebar:
        st.header("Display")
        clip_low = st.slider("Lower contrast clip percentile", 0.0, 20.0, 2.0, 0.5)
        clip_high = st.slider("Upper contrast clip percentile", 80.0, 100.0, 98.0, 0.5)
        use_distance = st.checkbox("Use distance axis if available", value=True)
        invert_y = st.checkbox("Invert vertical axis", value=True)
        colorscale = st.selectbox("Colorscale", ["Gray", "Viridis", "Plasma", "RdBu"], index=0)

        st.header("Processing")
        time_zero_shift = st.slider("Time-zero shift (samples)", -50, 50, 0)
        dewow_on = st.checkbox("Dewow", value=False)
        dewow_window = st.slider("Dewow window", 3, 101, 21, step=2)
        background_on = st.checkbox("Background removal", value=False)
        trace_normalize_on = st.checkbox("Normalize each trace", value=False)
        gain_mode = st.selectbox("Gain mode", ["None", "Linear", "Exponential"], index=0)
        gain_strength = st.slider("Gain strength", 0.0, 4.0, 0.8, 0.1)

        st.header("Optional AI")
        ai_enable = st.checkbox("Enable AI interpretation", value=False)
        ai_prefer_local = st.checkbox("Prefer Ollama first", value=True)
        ollama_base_url = st.text_input("Ollama URL", value="http://127.0.0.1:11434")
        ollama_model = st.text_input("Ollama model", value="llama3.1:8b")
        openai_model = st.text_input("OpenAI model", value="gpt-4.1-mini")

    st.subheader("1. Load RADAN Project Files")
    demo_mode = st.checkbox("Use built-in demo project", value=False)
    uploaded_files = st.file_uploader(
        "Upload one or more related files",
        type=["dzt", "csv", "txt", "asc", "dzg", "dzx", "dza"],
        accept_multiple_files=True,
        help="Upload matching files like line01.dzt, line01.dzg, line01.dzx, and line01.dza together.",
    )

    if not demo_mode and not uploaded_files:
        st.info("Upload one or more related files, or turn on the built-in demo project.")
        st.stop()

    projects = build_projects_from_uploads(uploaded_files or [], use_demo=demo_mode)
    if not projects:
        st.warning("No readable projects were assembled.")
        st.stop()

    st.subheader("2. Project Inventory")
    rows = []
    stems = sorted(projects.keys())
    for stem in stems:
        p = projects[stem]
        rows.append({"Project": stem, "DZT": "✓" if p.dzt is not None else "✗", "DZG": "✓" if p.dzg is not None else "✗", "DZX": "✓" if p.dzx is not None else "✗", "DZA": "✓" if p.dza is not None else "✗", "Other files": len(p.other_files)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    selected_stem = st.selectbox("Choose project", stems)
    project = projects[selected_stem]
    primary = project.dzt or project.dzg or project.dzx or project.dza
    if primary is None:
        st.warning("This project has no primary readable RADAN file yet.")
        st.stop()

    if project.dzt is None or project.dzt.traces is None:
        st.info("No DZT file loaded for this project, so there is no radargram to process yet.")
        companion_tab = st.tabs(["Companions"])[0]
        with companion_tab:
            render_companions(project)
        st.stop()

    raw_traces = np.asarray(project.dzt.traces, dtype=float)
    n_traces, n_samples = raw_traces.shape
    trace_range = st.slider("Trace crop", 0, n_traces - 1, (0, n_traces - 1))
    sample_range = st.slider("Sample crop", 0, n_samples - 1, (0, n_samples - 1))

    settings = {
        "time_zero_shift": time_zero_shift,
        "dewow_on": dewow_on,
        "dewow_window": dewow_window,
        "background_on": background_on,
        "trace_normalize_on": trace_normalize_on,
        "gain_mode": gain_mode,
        "gain_strength": gain_strength,
        "trace_range": trace_range,
        "sample_range": sample_range,
    }

    processed_traces, processed_time_axis, processed_distance_axis, notes = process_gpr_data(project.dzt, settings)
    raw_cropped, raw_time_axis, raw_distance_axis = crop_gpr(project.dzt, raw_traces, trace_range, sample_range)

    tabs = st.tabs(["Load / Inventory", "Radargram", "Processed Radargram", "Single Trace", "Companions", "Interpret", "Report / Export"])

    with tabs[0]:
        st.markdown("### Project metadata")
        st.write(f"**File type:** {primary.file_type}")
        st.write(f"**Source file:** {primary.source_file}")
        st.write(f"**Trace matrix shape:** {raw_traces.shape}")
        st.dataframe(pd.DataFrame([{"Field": k, "Value": str(v)} for k, v in primary.metadata.items()]), width="stretch", hide_index=True)
        st.markdown("### Processing plan")
        if notes:
            for item in notes:
                st.write(f"- {item}")
        else:
            st.write("- No processing beyond crop/display yet.")

    with tabs[1]:
        fig = heatmap_figure(raw_cropped, raw_time_axis, raw_distance_axis, use_distance, clip_low, clip_high, invert_y, colorscale, f"Raw Radargram: {project.dzt.source_file}")
        st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        fig = heatmap_figure(processed_traces, processed_time_axis, processed_distance_axis, use_distance, clip_low, clip_high, invert_y, colorscale, f"Processed Radargram: {project.dzt.source_file}")
        st.plotly_chart(fig, width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Traces", processed_traces.shape[0])
        c2.metric("Samples", processed_traces.shape[1])
        c3.metric("Std Dev", f"{float(np.std(processed_traces)):.2f}")
        c4.metric("Mean |Amp|", f"{float(np.mean(np.abs(processed_traces))):.2f}")

        st.markdown("### Raw vs processed comparison")
        compare_cols = st.columns(2)
        with compare_cols[0]:
            st.plotly_chart(heatmap_figure(raw_cropped, raw_time_axis, raw_distance_axis, use_distance, clip_low, clip_high, invert_y, colorscale, "Raw (cropped)"), width="stretch")
        with compare_cols[1]:
            st.plotly_chart(heatmap_figure(processed_traces, processed_time_axis, processed_distance_axis, use_distance, clip_low, clip_high, invert_y, colorscale, "Processed"), width="stretch")

    with tabs[3]:
        trace_index = st.slider("Trace index", 0, processed_traces.shape[0] - 1, 0)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(trace_figure(raw_cropped, trace_index, raw_time_axis, "Raw trace"), width="stretch")
        with right:
            st.plotly_chart(trace_figure(processed_traces, trace_index, processed_time_axis, "Processed trace"), width="stretch")

    with tabs[4]:
        render_companions(project)

    with tabs[5]:
        st.markdown("### First-pass interpretation")
        st.info("Use this as a cautious teaching aid. It should describe likely reflector patterns and processing effects, not claim final subsurface truth.")
        energy_profile = np.mean(np.abs(processed_traces), axis=0)
        strongest_sample = int(np.argmax(energy_profile))
        strongest_time = float(processed_time_axis[strongest_sample]) if processed_time_axis is not None else strongest_sample
        st.write(f"Strongest mean-energy sample/time: **{strongest_time:.2f}**")
        st.write(f"Companion files present: DZG={project.dzg is not None}, DZX={project.dzx is not None}, DZA={project.dza is not None}")
        ai_question = st.text_area("Optional AI request", value="Give me a concise interpretation of likely reflector patterns, processing effects, and what a student should inspect next.")
        if ai_enable:
            local_status = "available" if ollama_available(ollama_base_url) else "not available"
            openai_status = "available" if openai_available() else "not available"
            st.caption(f"Ollama is {local_status}. OpenAI fallback is {openai_status}.")
            if st.button("Generate AI summary", width="stretch"):
                try:
                    response, provider = generate_ai_summary(project, raw_cropped, processed_traces, notes, ai_question, ai_prefer_local, ollama_model, ollama_base_url, openai_model)
                    st.session_state["gpr_ai_response"] = response
                    st.session_state["gpr_ai_provider"] = provider
                except Exception as exc:
                    st.error(str(exc))
        if st.session_state.get("gpr_ai_response"):
            st.markdown(f"**AI provider:** {st.session_state.get('gpr_ai_provider','')} ")
            st.write(st.session_state.get("gpr_ai_response", ""))

    with tabs[6]:
        st.markdown("### Report")
        report_lines = [
            f"Project: {project.stem}",
            f"Source file: {project.dzt.source_file}",
            f"Raw trace matrix shape: {raw_traces.shape}",
            f"Processed trace matrix shape: {processed_traces.shape}",
            f"Distance axis available: {project.dzt.distance_axis is not None}",
            f"Time axis available: {project.dzt.time_axis is not None}",
            "Processing notes:",
        ] + [f"- {item}" for item in notes]
        if st.session_state.get("gpr_ai_response"):
            report_lines += ["", f"AI summary ({st.session_state.get('gpr_ai_provider','')}):", st.session_state.get("gpr_ai_response", "")]
        report_text = "\n".join(report_lines)
        st.text_area("Report preview", value=report_text, height=320)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download processed traces as CSV", data=pd.DataFrame(processed_traces).to_csv(index=False).encode("utf-8"), file_name=f"{project.stem}_processed_traces.csv", mime="text/csv", width="stretch")
        with c2:
            st.download_button("Download raw traces as CSV", data=traces_to_csv_bytes(project.dzt), file_name=f"{project.stem}_raw_traces.csv", mime="text/csv", width="stretch")
        with c3:
            st.download_button("Download report as TXT", data=report_text.encode("utf-8"), file_name=f"{project.stem}_report.txt", mime="text/plain", width="stretch")


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
        rows = []
        for name, inspection in project.other_files.items():
            rows.append({"File": name, "Type guess": inspection.likely_type, "Binary": inspection.is_binary, "Size": inspection.size_bytes, "Notes": inspection.notes})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

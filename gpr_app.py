
import io
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter


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
class ProcessedGPR:
    raw: np.ndarray
    processed: np.ndarray
    time_axis: np.ndarray
    distance_axis: np.ndarray
    notes: List[str]
    disturbance_score: np.ndarray
    hyperbola_score: np.ndarray
    reflector_score: np.ndarray
    candidate_table: pd.DataFrame


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

    return FileInspection(
        source_file=name,
        extension=ext,
        size_bytes=size,
        is_binary=is_binary,
        likely_type=likely_type,
        notes=notes,
        preview_text=preview_text,
    )


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
    centered = raw.astype(np.float32) - float(rh_zero)

    time_axis = np.linspace(0, rhf_range, rh_nsamp) if rhf_range and rhf_range > 0 else np.arange(rh_nsamp, dtype=float)
    distance_axis = np.arange(n_traces) / rhf_spm if rhf_spm and rhf_spm > 0 else np.arange(n_traces, dtype=float)

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

    return GPRData(
        file_type="dzt",
        source_file=name,
        traces=centered,
        time_axis=time_axis,
        distance_axis=distance_axis,
        metadata=metadata,
    )


def parse_csv(name: str, content: bytes) -> GPRData:
    df = pd.read_csv(io.BytesIO(content))
    numeric_df = df.select_dtypes(include=[np.number])
    traces = numeric_df.to_numpy(dtype=float) if not numeric_df.empty else None
    return GPRData(
        file_type="csv",
        source_file=name,
        traces=traces,
        metadata={"columns": list(df.columns), "n_rows": int(len(df)), "n_numeric_columns": int(len(numeric_df.columns))},
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
                    traces = numeric_df.to_numpy(dtype=float)
                    break
        except Exception:
            continue

    return GPRData(
        file_type=forced_type,
        source_file=name,
        traces=traces,
        gps=gps,
        metadata={"length_chars": len(text), "preview": text[:300], "parsed_delimiter": delimiter_used},
        raw_bytes=content,
    )


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


def build_projects_from_uploads(uploaded_files) -> Dict[str, RadanProject]:
    projects: Dict[str, RadanProject] = {}
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
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def dewow(traces: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(traces, dtype=float)
    for i in range(traces.shape[0]):
        out[i] = traces[i] - moving_average_1d(traces[i], window)
    return out


def apply_time_zero_shift(traces: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples == 0:
        return traces.copy()
    shifted = np.zeros_like(traces)
    if shift_samples > 0:
        shifted[:, :-shift_samples] = traces[:, shift_samples:]
    else:
        shifted[:, -shift_samples:] = traces[:, :shift_samples]
    return shifted


def apply_gain(traces: np.ndarray, mode: str, strength: float) -> np.ndarray:
    if mode == "None" or strength <= 0:
        return traces.copy()
    n_samples = traces.shape[1]
    t = np.linspace(0.0, 1.0, n_samples)
    if mode == "Linear":
        gain = 1.0 + strength * t
    else:
        gain = np.exp(strength * t)
    return traces * gain[np.newaxis, :]


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(traces), axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return traces / denom


def background_remove(traces: np.ndarray) -> np.ndarray:
    return traces - np.mean(traces, axis=0, keepdims=True)


def _normalize_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    score -= np.nanmin(score)
    mx = np.nanmax(score)
    if mx > 0:
        score = score / mx
    return score


def reflector_score_map(data: np.ndarray) -> np.ndarray:
    absd = np.abs(data)
    coherent = gaussian_filter(absd, sigma=(4.0, 0.9))
    local = gaussian_filter(absd, sigma=(1.2, 1.2))
    broad = gaussian_filter(absd, sigma=(10.0, 4.0))
    contrast = np.maximum(local - broad, 0.0)
    score = 0.7 * coherent + 0.3 * contrast
    return _normalize_score(score)


def disturbance_map(data: np.ndarray) -> np.ndarray:
    absd = np.abs(data)
    local_energy = uniform_filter(absd, size=(9, 17), mode="nearest")
    broad_energy = uniform_filter(absd, size=(31, 41), mode="nearest")
    texture = np.maximum(local_energy - broad_energy, 0.0)

    horiz = gaussian_filter(absd, sigma=(5.5, 1.1))
    lateral_break = np.abs(np.gradient(horiz, axis=0))
    vertical_break = np.abs(np.gradient(gaussian_filter(absd, sigma=(1.2, 3.5)), axis=1))

    score = 0.45 * texture + 0.35 * gaussian_filter(lateral_break, sigma=(2.0, 1.2)) + 0.20 * gaussian_filter(vertical_break, sigma=(1.5, 1.5))
    return _normalize_score(score)


def hyperbola_score_map(data: np.ndarray, apex_half_width: int = 6) -> np.ndarray:
    data_abs = np.abs(data)
    smooth = gaussian_filter(data_abs, sigma=(1.0, 1.0))
    local_max = maximum_filter(smooth, size=(3, apex_half_width * 2 + 1))
    apex_mask = smooth >= local_max * 0.965
    curved_support = np.zeros_like(smooth)
    for w in [3, 5, 7, 9]:
        up = np.roll(smooth, shift=w, axis=1)
        left = np.roll(smooth, shift=-w, axis=0)
        right = np.roll(smooth, shift=w, axis=0)
        curved_support += np.minimum(up, (left + right) / 2.0)
    score = (0.45 * smooth + 0.55 * curved_support) * apex_mask.astype(float)
    score = gaussian_filter(score, sigma=(2.2, 1.2))
    return _normalize_score(score)


def candidate_table_from_scores(dist_score: np.ndarray, hyp_score: np.ndarray, refl_score: np.ndarray, time_axis: np.ndarray, distance_axis: np.ndarray, top_n: int = 12) -> pd.DataFrame:
    maps = {
        "possible hyperbolic target": hyp_score,
        "possible disturbed zone": dist_score,
        "possible strong reflector / target": refl_score,
    }
    rows = []
    seen = []

    def far_enough(tr, smp):
        for tr0, smp0 in seen:
            if abs(tr - tr0) <= 12 and abs(smp - smp0) <= 18:
                return False
        return True

    for label, arr in maps.items():
        local = maximum_filter(arr, size=(11, 21))
        thresh = np.quantile(arr, 0.975 if label != "possible disturbed zone" else 0.982)
        peaks = np.argwhere((arr >= local) & (arr >= thresh))
        peak_rows = []
        for tr, smp in peaks:
            combined = 0.35 * refl_score[tr, smp] + 0.35 * hyp_score[tr, smp] + 0.30 * dist_score[tr, smp]
            peak_rows.append((combined, tr, smp))
        peak_rows.sort(reverse=True, key=lambda x: x[0])
        for combined, tr, smp in peak_rows[: max(top_n, 8)]:
            if not far_enough(tr, smp):
                continue
            seen.append((tr, smp))
            rows.append({
                "Trace Index": int(tr),
                "Sample Index": int(smp),
                "Distance": float(distance_axis[tr]),
                "Time": float(time_axis[smp]),
                "Disturbance Score": float(dist_score[tr, smp]),
                "Hyperbola Score": float(hyp_score[tr, smp]),
                "Reflector Score": float(refl_score[tr, smp]),
                "Combined Score": float(combined),
                "Likely Pattern": label,
            })

    if not rows:
        return pd.DataFrame(columns=["Trace Index", "Sample Index", "Distance", "Time", "Disturbance Score", "Hyperbola Score", "Reflector Score", "Combined Score", "Likely Pattern"])

    out = pd.DataFrame(rows).sort_values(["Combined Score", "Reflector Score", "Hyperbola Score"], ascending=False).head(top_n).reset_index(drop=True)
    return out


def process_gpr(gpr_data: GPRData, controls: Dict[str, Any]) -> ProcessedGPR:
    raw = np.asarray(gpr_data.traces, dtype=float)
    notes: List[str] = []
    distance_axis = gpr_data.distance_axis if gpr_data.distance_axis is not None and len(gpr_data.distance_axis) == raw.shape[0] else np.arange(raw.shape[0], dtype=float)
    time_axis = gpr_data.time_axis if gpr_data.time_axis is not None and len(gpr_data.time_axis) == raw.shape[1] else np.arange(raw.shape[1], dtype=float)

    tr0 = int(controls["trace_crop_min"])
    tr1 = int(controls["trace_crop_max"])
    sm0 = int(controls["sample_crop_min"])
    sm1 = int(controls["sample_crop_max"])
    raw = raw[tr0:tr1 + 1, sm0:sm1 + 1]
    distance_axis = distance_axis[tr0:tr1 + 1]
    time_axis = time_axis[sm0:sm1 + 1]
    notes.append(f"Cropped traces to {tr0}-{tr1} and samples to {sm0}-{sm1}.")

    proc = raw.copy()
    tz = int(controls["time_zero_shift"])
    if tz != 0:
        proc = apply_time_zero_shift(proc, tz)
        notes.append(f"Applied time-zero shift of {tz} samples.")
    if controls["dewow_on"]:
        proc = dewow(proc, int(controls["dewow_window"]))
        notes.append(f"Applied dewow with window {int(controls['dewow_window'])}.")
    if controls["background_remove"]:
        proc = background_remove(proc)
        notes.append("Applied background removal.")
    proc = apply_gain(proc, controls["gain_mode"], float(controls["gain_strength"]))
    if controls["gain_mode"] != "None" and float(controls["gain_strength"]) > 0:
        notes.append(f"Applied {controls['gain_mode'].lower()} gain with strength {float(controls['gain_strength']):.2f}.")
    if controls["trace_normalize"]:
        proc = normalize_traces(proc)
        notes.append("Applied per-trace normalization.")

    refl_score = reflector_score_map(proc)
    dist_score = disturbance_map(proc)
    hyp_score = hyperbola_score_map(proc)
    candidates = candidate_table_from_scores(dist_score, hyp_score, refl_score, time_axis, distance_axis, top_n=int(controls["top_candidates"]))
    if not candidates.empty:
        notes.append(f"Flagged {len(candidates)} review targets from processed radargram scoring.")
    else:
        notes.append("No strong automated review targets exceeded the current candidate threshold.")

    return ProcessedGPR(raw=raw, processed=proc, time_axis=time_axis, distance_axis=distance_axis, notes=notes, disturbance_score=dist_score, hyperbola_score=hyp_score, reflector_score=refl_score, candidate_table=candidates)


def make_radargram_figure(data: np.ndarray, time_axis: np.ndarray, distance_axis: np.ndarray, use_distance: bool, clip_low: float, clip_high: float, invert_y: bool, cmap: str, title: str, overlay_df: Optional[pd.DataFrame] = None):
    x = distance_axis if use_distance and len(distance_axis) == data.shape[0] else np.arange(data.shape[0])
    xlabel = "Distance" if use_distance and len(distance_axis) == data.shape[0] else "Trace Number"
    y = time_axis if len(time_axis) == data.shape[0 if False else 1] else np.arange(data.shape[1])
    ylabel = "Time (ns)" if len(time_axis) == data.shape[1] else "Sample"

    vmin = np.percentile(data, clip_low)
    vmax = np.percentile(data, clip_high)
    extent = [x[0], x[-1], y[-1], y[0]] if invert_y else [x[0], x[-1], y[0], y[-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(data.T, aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if overlay_df is not None and not overlay_df.empty:
        ox = overlay_df["Distance"].to_numpy() if xlabel == "Distance" else overlay_df["Trace Index"].to_numpy()
        oy = overlay_df["Time"].to_numpy()
        ax.scatter(ox, oy, s=40, facecolors="none", edgecolors="red", linewidths=1.5)
    fig.tight_layout()
    return fig


def make_trace_figure(raw: np.ndarray, processed: np.ndarray, time_axis: np.ndarray, trace_index: int):
    trace_index = max(0, min(trace_index, raw.shape[0] - 1))
    y = time_axis if len(time_axis) == raw.shape[1] else np.arange(raw.shape[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(raw[trace_index], y, label="Raw", alpha=0.8)
    ax.plot(processed[trace_index], y, label="Processed", alpha=0.8)
    ax.set_title(f"Trace {trace_index}")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Time (ns)" if len(time_axis) == raw.shape[1] else "Sample")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    return fig


def traces_to_csv_bytes(traces: np.ndarray) -> bytes:
    return pd.DataFrame(traces).to_csv(index=False).encode("utf-8")


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
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    return bool(os.getenv("OPENAI_API_KEY") or key)


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
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful geophysical assistant. Use only the supplied radargram context. Be concise and do not claim graves or definitive objects."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


def build_ai_context(project: RadanProject, processed: ProcessedGPR) -> str:
    lines = [
        f"Source file: {project.dzt.source_file if project.dzt else project.stem}",
        f"Trace count: {processed.processed.shape[0]}",
        f"Samples per trace: {processed.processed.shape[1]}",
        f"Top candidate count: {len(processed.candidate_table)}",
    ]
    if not processed.candidate_table.empty:
        sample = processed.candidate_table.head(8)
        lines.append("Top review targets:")
        for _, row in sample.iterrows():
            lines.append(
                f"- trace {int(row['Trace Index'])}, time {row['Time']:.2f}, combined {row['Combined Score']:.2f}, pattern {row['Likely Pattern']}"
            )
    lines.append("Processing notes: " + "; ".join(processed.notes))
    return "\n".join(lines)


def generate_ai_summary(project: RadanProject, processed: ProcessedGPR, question: str, prefer_local: bool, ollama_model: str, ollama_base_url: str, openai_model: str) -> Tuple[str, str]:
    prompt = (
        "Use the following GPR processing context to answer the user. "
        "Treat all outputs as review targets, not confirmed graves or confirmed buried objects. "
        "Be concise and mention uncertainty.\n\n"
        + build_ai_context(project, processed)
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
    if (not prefer_local) and ollama_available(ollama_base_url):
        try:
            return call_ollama(prompt, ollama_model, ollama_base_url), f"Ollama ({ollama_model})"
        except Exception:
            pass
    raise RuntimeError("No AI provider available. Core GPR analysis still works without AI.")


def init_page():
    st.set_page_config(page_title="GPR Control Panel", layout="wide")
    st.title("GPR Control Panel")
    st.caption("Raw-file-first GPR review tool for grouped RADAN project files. Candidate finding is assistive only, not definitive interpretation.")


def render_sidebar(project: Optional[RadanProject] = None) -> Dict[str, Any]:
    st.sidebar.header("Display Controls")
    n_traces = int(project.dzt.traces.shape[0]) if project and project.dzt and project.dzt.traces is not None else 200
    n_samples = int(project.dzt.traces.shape[1]) if project and project.dzt and project.dzt.traces is not None else 256

    controls = {
        "clip_low": st.sidebar.slider("Lower contrast clip percentile", 0.0, 20.0, 2.0, 0.5),
        "clip_high": st.sidebar.slider("Upper contrast clip percentile", 80.0, 100.0, 98.0, 0.5),
        "use_distance": st.sidebar.checkbox("Use distance axis if available", value=True),
        "invert_y": st.sidebar.checkbox("Invert vertical axis", value=True),
        "cmap": st.sidebar.selectbox("Colormap", ["gray", "seismic", "viridis", "plasma"], index=0),
        "time_zero_shift": st.sidebar.slider("Time-zero shift (samples)", -40, 40, 0, 1),
        "dewow_on": st.sidebar.checkbox("Dewow", value=True),
        "dewow_window": st.sidebar.slider("Dewow window", 3, 81, 21, 2),
        "background_remove": st.sidebar.checkbox("Background removal", value=True),
        "trace_normalize": st.sidebar.checkbox("Per-trace normalization", value=False),
        "gain_mode": st.sidebar.selectbox("Gain mode", ["None", "Linear", "Exponential"], index=1),
        "gain_strength": st.sidebar.slider("Gain strength", 0.0, 8.0, 2.0, 0.1),
        "trace_crop_min": st.sidebar.slider("Trace crop start", 0, max(0, n_traces - 1), 0, 1),
        "trace_crop_max": st.sidebar.slider("Trace crop end", 0, max(0, n_traces - 1), max(0, n_traces - 1), 1),
        "sample_crop_min": st.sidebar.slider("Sample crop start", 0, max(0, n_samples - 1), 0, 1),
        "sample_crop_max": st.sidebar.slider("Sample crop end", 0, max(0, n_samples - 1), max(0, n_samples - 1), 1),
        "top_candidates": st.sidebar.slider("Top review targets", 3, 25, 10, 1),
        "ai_enable": st.sidebar.checkbox("Enable optional AI interpretation", value=False),
        "ai_prefer_local": st.sidebar.checkbox("Prefer Ollama", value=True),
        "ollama_model": st.sidebar.text_input("Ollama model", value="llama3.1:8b"),
        "ollama_base_url": st.sidebar.text_input("Ollama URL", value="http://127.0.0.1:11434"),
        "openai_model": st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini"),
    }
    if controls["trace_crop_max"] < controls["trace_crop_min"]:
        controls["trace_crop_max"] = controls["trace_crop_min"]
    if controls["sample_crop_max"] < controls["sample_crop_min"]:
        controls["sample_crop_max"] = controls["sample_crop_min"]
    return controls


def render_file_loader():
    st.subheader("1. Load RADAN Project Files")
    uploaded = st.file_uploader(
        "Upload one or more related files",
        type=["dzt", "csv", "txt", "asc", "dzg", "dzx", "dza"],
        accept_multiple_files=True,
        help="Upload matching files like line01.dzt, line01.dzg, line01.dzx, and line01.dza together.",
    )
    return uploaded


def render_project_inventory(projects: Dict[str, RadanProject]) -> str:
    st.subheader("2. Project Inventory")
    rows = []
    for stem in sorted(projects.keys()):
        p = projects[stem]
        rows.append({"Project": stem, "DZT": "✓" if p.dzt else "✗", "DZG": "✓" if p.dzg else "✗", "DZX": "✓" if p.dzx else "✗", "DZA": "✓" if p.dza else "✗", "Other files": len(p.other_files)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    return st.selectbox("Choose project", sorted(projects.keys()))


def render_metadata(gpr_data: GPRData):
    st.subheader("Metadata")
    st.write(f"**File type:** {gpr_data.file_type}")
    st.write(f"**Source file:** {gpr_data.source_file}")
    if gpr_data.traces is not None:
        st.write(f"**Trace matrix shape:** {gpr_data.traces.shape}")
    if gpr_data.gps is not None:
        st.json(gpr_data.gps)
    if gpr_data.metadata:
        st.dataframe(pd.DataFrame([{"Field": k, "Value": str(v)} for k, v in gpr_data.metadata.items()]), width="stretch", hide_index=True)


def render_companion_panels(project: RadanProject):
    st.subheader("Companion Files")
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


def main():
    init_page()
    uploaded_files = render_file_loader()
    if not uploaded_files:
        st.info("Upload one or more related files to begin.")
        st.stop()

    projects = build_projects_from_uploads(uploaded_files)
    if not projects:
        st.warning("No readable projects were assembled.")
        st.stop()

    selected_stem = render_project_inventory(projects)
    project = projects[selected_stem]
    primary = project.dzt or project.dzg or project.dzx or project.dza
    if primary is None:
        st.warning("This project has no primary readable RADAN file yet.")
        render_companion_panels(project)
        st.stop()

    controls = render_sidebar(project)
    render_metadata(primary)

    if project.dzt is None:
        st.info("No DZT file loaded for this project, so there is no radargram to process yet.")
        render_companion_panels(project)
        st.stop()

    processed = process_gpr(project.dzt, controls)

    tabs = st.tabs(["Load / Inventory", "Radargram", "Processed Radargram", "Single Trace", "Companions", "Interpret", "Report / Export"])

    with tabs[0]:
        st.markdown("### Current project")
        render_metadata(project.dzt)
        st.markdown("### Project files")
        inv = {
            "DZT": project.files.get("dzt", "—"),
            "DZG": project.files.get("dzg", "—"),
            "DZX": project.files.get("dzx", "—"),
            "DZA": project.files.get("dza", "—"),
        }
        st.dataframe(pd.DataFrame([inv]), width="stretch", hide_index=True)
        st.markdown("### Processing preview")
        for msg in processed.notes:
            st.write(f"- {msg}")

    with tabs[1]:
        st.markdown("### Raw radargram")
        fig = make_radargram_figure(processed.raw, processed.time_axis, processed.distance_axis, controls["use_distance"], controls["clip_low"], controls["clip_high"], controls["invert_y"], controls["cmap"], f"Raw radargram: {project.dzt.source_file}", processed.candidate_table)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        if not processed.candidate_table.empty:
            st.caption("Red circles mark automated review targets derived from the processed data, projected back onto the raw view.")

    with tabs[2]:
        st.markdown("### Processed radargram")
        fig = make_radargram_figure(processed.processed, processed.time_axis, processed.distance_axis, controls["use_distance"], controls["clip_low"], controls["clip_high"], controls["invert_y"], controls["cmap"], "Processed radargram with review targets", processed.candidate_table)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Disturbance score map**")
            fig2 = make_radargram_figure(processed.disturbance_score, processed.time_axis, processed.distance_axis, controls["use_distance"], 0.0, 100.0, controls["invert_y"], "viridis", "Possible disturbed zones", processed.candidate_table)
            st.pyplot(fig2, clear_figure=True, use_container_width=True)
        with c2:
            st.markdown("**Hyperbola score map**")
            fig3 = make_radargram_figure(processed.hyperbola_score, processed.time_axis, processed.distance_axis, controls["use_distance"], 0.0, 100.0, controls["invert_y"], "plasma", "Possible hyperbolic targets", processed.candidate_table)
            st.pyplot(fig3, clear_figure=True, use_container_width=True)

    with tabs[3]:
        st.markdown("### Raw vs processed single trace")
        trace_index = st.slider("Trace index", 0, max(0, processed.raw.shape[0] - 1), 0)
        fig = make_trace_figure(processed.raw, processed.processed, processed.time_axis, trace_index)
        st.pyplot(fig, clear_figure=True)
        if not processed.candidate_table.empty:
            nearby = processed.candidate_table.iloc[(processed.candidate_table["Trace Index"] - trace_index).abs().argsort()].head(5)
            st.markdown("**Nearby review targets**")
            st.dataframe(nearby, width="stretch", hide_index=True)

    with tabs[4]:
        render_companion_panels(project)

    with tabs[5]:
        st.markdown("### Candidate review")
        st.info("These outputs are review targets from the processed radargram. They are not confirmed graves or confirmed buried objects.")
        if processed.candidate_table.empty:
            st.warning("No strong review targets exceeded the current scoring threshold.")
        else:
            st.dataframe(processed.candidate_table, width="stretch", hide_index=True)
        st.markdown("### Interpretation notes")
        st.write("- Disturbance score emphasizes local texture change, reflector disruption, and unusual energy patterns.")
        st.write("- Hyperbola score emphasizes local apex-like bright zones with curved support beneath them.")
        st.write("- A strong score means 'review here', not 'grave detected'.")

        st.markdown("---")
        st.markdown("### Optional AI assistant")
        if not controls["ai_enable"]:
            st.caption("AI is off. Core analysis remains fully available without it.")
        else:
            provider_bits = []
            if ollama_available(controls["ollama_base_url"]):
                provider_bits.append(f"Ollama ready at {controls['ollama_base_url']}")
            if openai_available():
                provider_bits.append("OpenAI key detected")
            st.caption(" | ".join(provider_bits) if provider_bits else "No AI backend detected right now. The app still works without AI.")
            question = st.text_area("AI request", value="Give me a concise interpretation of the strongest review targets and what a student should inspect next.")
            if st.button("Generate AI summary", width="stretch"):
                try:
                    text, provider = generate_ai_summary(project, processed, question, controls["ai_prefer_local"], controls["ollama_model"], controls["ollama_base_url"], controls["openai_model"])
                    st.success(f"Provider used: {provider}")
                    st.write(text)
                    st.session_state["gpr_ai_text"] = text
                    st.session_state["gpr_ai_provider"] = provider
                except Exception as e:
                    st.warning(str(e))

    with tabs[6]:
        st.markdown("### Report")
        lines = [
            "GPR Control Panel Report",
            f"Project: {project.stem}",
            f"Source file: {project.dzt.source_file}",
            f"Trace count: {processed.processed.shape[0]}",
            f"Samples per trace: {processed.processed.shape[1]}",
            "",
            "Processing notes:",
            *[f"- {n}" for n in processed.notes],
            "",
            f"Candidate count: {len(processed.candidate_table)}",
        ]
        if not processed.candidate_table.empty:
            lines.append("")
            lines.append("Top review targets:")
            for _, row in processed.candidate_table.head(10).iterrows():
                lines.append(f"- trace {int(row['Trace Index'])}, time {row['Time']:.2f}, combined {row['Combined Score']:.2f}, {row['Likely Pattern']}")
        ai_text = st.session_state.get("gpr_ai_text", "")
        if ai_text:
            lines.extend(["", "Optional AI summary:", ai_text])
        report_text = "\n".join(lines)
        st.text_area("Report preview", report_text, height=320)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download raw trace matrix as CSV", data=traces_to_csv_bytes(processed.raw), file_name=f"{Path(project.dzt.source_file).stem}_raw_traces.csv", mime="text/csv", width="stretch")
        with c2:
            st.download_button("Download processed trace matrix as CSV", data=traces_to_csv_bytes(processed.processed), file_name=f"{Path(project.dzt.source_file).stem}_processed_traces.csv", mime="text/csv", width="stretch")
        with c3:
            st.download_button("Download report as TXT", data=report_text.encode("utf-8"), file_name=f"{Path(project.dzt.source_file).stem}_report.txt", mime="text/plain", width="stretch")


if __name__ == "__main__":
    main()

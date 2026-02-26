"""
Speech Signal Analyzer  v3.0
=============================
Part 1 : Signal Feature Extraction  (STE, ZCR, Mean, Var, SD, RMS, SNR)
Part 2 : Phoneme Recording & Time-Domain Analysis
Part 3 : ZCR Window Inspector  (per-frame ZCR, polarity, sample table)

Install:
    pip install PyQt5 matplotlib numpy scipy sounddevice soundfile
Run:
    python speech_analyzer.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert

import sounddevice as sd
import soundfile as sf

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QFrame, QSizePolicy, QScrollArea, QSplitter,
    QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# ============================================================
# PALETTE
# ============================================================
C = {
    "bg":         "#0A0E17",
    "surface":    "#111827",
    "surface2":   "#1C2333",
    "border":     "#2A3347",
    "border_hi":  "#3B4F6E",
    "accent":     "#4F8EF7",
    "accent2":    "#7C5CFC",
    "green":      "#34D399",
    "amber":      "#FBBF24",
    "red":        "#F87171",
    "cyan":       "#38BDF8",
    "orange":     "#FB923C",
    "text":       "#E2E8F0",
    "text_muted": "#64748B",
    "text_dim":   "#94A3B8",
    "plot_bg":    "#080C14",
    # plot series
    "waveform":   "#4F8EF7",
    "ste_line":   "#34D399",
    "zcr_line":   "#FBBF24",
    "noisy":      "#F87171",
    "spectrum":   "#7C5CFC",
    "envelope":   "#FB923C",
    "acf":        "#38BDF8",
}

# ============================================================
# STYLESHEET
# ============================================================
APP_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}}
/* ── Tabs ── */
QTabWidget::pane  {{ border: none; background-color: {C['bg']}; }}
QTabBar           {{ background: {C['surface']}; }}
QTabBar::tab {{
    background: transparent;
    color: {C['text_muted']};
    padding: 12px 26px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.7px;
    border: none;
    border-bottom: 2px solid transparent;
    text-transform: uppercase;
}}
QTabBar::tab:selected      {{ color: {C['accent']};  border-bottom: 2px solid {C['accent']}; }}
QTabBar::tab:hover:!selected {{ color: {C['text']};  border-bottom: 2px solid {C['border_hi']}; }}

/* ── Buttons ── */
QPushButton {{
    background-color: {C['surface2']};
    color: {C['text']};
    border: 1px solid {C['border']};
    padding: 8px 16px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
    min-height: 32px;
}}
QPushButton:hover   {{ background-color:{C['border_hi']}; border-color:{C['accent']}; color:{C['accent']}; }}
QPushButton:pressed {{ background-color:{C['accent']}; color:white; }}
QPushButton:disabled {{ background-color:{C['surface']}; color:{C['text_muted']}; border-color:{C['border']}; }}

QPushButton#btn_primary {{
    background-color:{C['accent']}; color:white; border:none;
    font-size:13px; padding:10px 22px; border-radius:8px;
}}
QPushButton#btn_primary:hover   {{ background-color:#6BA3FF; }}
QPushButton#btn_primary:pressed {{ background-color:#3B7CE8; }}
QPushButton#btn_primary:disabled {{ background-color:{C['border']}; color:{C['text_muted']}; }}

QPushButton#btn_record {{
    background-color:transparent; color:{C['red']};
    border:2px solid {C['red']}; font-size:12px;
    padding:9px 20px; border-radius:8px; font-weight:700;
}}
QPushButton#btn_record:hover {{ background-color:{C['red']}; color:white; }}

QPushButton#btn_record_active {{
    background-color:{C['red']}; color:white;
    border:2px solid {C['red']}; font-size:12px;
    padding:9px 20px; border-radius:8px; font-weight:700;
}}
QPushButton#btn_play {{
    background-color:transparent; color:{C['green']};
    border:1px solid {C['green']}; border-radius:7px;
    padding:7px 14px; font-weight:600; min-height:28px;
}}
QPushButton#btn_play:hover   {{ background-color:{C['green']}; color:{C['bg']}; }}
QPushButton#btn_play:disabled {{ border-color:{C['border']}; color:{C['text_muted']}; }}

QPushButton#btn_danger {{
    background-color:transparent; color:{C['red']};
    border:1px solid {C['border']}; border-radius:7px;
    padding:7px 14px; font-weight:600;
}}
QPushButton#btn_danger:hover {{ border-color:{C['red']}; color:{C['red']}; }}

/* ── Inputs ── */
QComboBox {{
    background-color:{C['surface2']}; border:1px solid {C['border']};
    border-radius:7px; padding:7px 12px; color:{C['text']};
    font-size:12px; min-height:32px;
}}
QComboBox:hover {{ border-color:{C['accent']}; }}
QComboBox::drop-down {{ border:none; width:20px; }}
QComboBox QAbstractItemView {{
    background-color:{C['surface2']}; border:1px solid {C['border_hi']};
    color:{C['text']}; selection-background-color:{C['accent']}; outline:none;
}}

QSpinBox, QDoubleSpinBox {{
    background-color:{C['surface2']}; border:1px solid {C['border']};
    border-radius:7px; padding:6px 10px; color:{C['text']};
    font-size:12px; min-height:32px;
}}
QSpinBox:hover, QDoubleSpinBox:hover {{ border-color:{C['accent']}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background:transparent; border:none; width:16px;
}}

/* ── Progress bars ── */
QProgressBar {{
    border:none; border-radius:3px; background-color:{C['surface2']};
    color:transparent; max-height:6px;
}}
QProgressBar::chunk {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C['accent']}, stop:1 {C['accent2']});
    border-radius:3px;
}}
QProgressBar#vu_bar {{ max-height:8px; }}
QProgressBar#vu_bar::chunk {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C['green']}, stop:0.65 {C['amber']}, stop:1 {C['red']});
}}

/* ── Table ── */
QTableWidget {{
    background-color:transparent; border:none;
    color:{C['text']}; gridline-color:{C['border']};
    font-size:12px; outline:none;
    alternate-background-color:{C['surface2']};
}}
QTableWidget::item {{ padding:6px 10px; }}
QTableWidget::item:selected {{ background-color:{C['border_hi']}; color:{C['accent']}; }}
QHeaderView::section {{
    background-color:{C['surface']}; color:{C['text_muted']};
    border:none; border-bottom:1px solid {C['border']};
    border-right:1px solid {C['border']};
    padding:7px 10px; font-size:10px; font-weight:700;
    letter-spacing:0.7px; text-transform:uppercase;
}}

/* ── Frames / Cards ── */
QFrame#card {{
    background-color:{C['surface']}; border:1px solid {C['border']}; border-radius:10px;
}}
QFrame#divider {{
    background-color:{C['border']}; max-height:1px; min-height:1px;
}}
QFrame#stat_card {{
    background-color:{C['surface']}; border:1px solid {C['border']}; border-radius:10px;
}}

/* ── Scrollbar ── */
QScrollBar:vertical {{ background:transparent; width:5px; }}
QScrollBar::handle:vertical {{ background:{C['border_hi']}; border-radius:3px; min-height:20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{ background:transparent; height:5px; }}
QScrollBar::handle:horizontal {{ background:{C['border_hi']}; border-radius:3px; min-width:20px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}

QLabel#stat_value {{
    color:{C['accent']}; font-size:19px; font-weight:700;
}}
"""

# ============================================================
# MATPLOTLIB HELPERS
# ============================================================
def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C["plot_bg"])
    for sp in ax.spines.values():
        sp.set_color(C["border"])
        sp.set_linewidth(0.7)
    ax.tick_params(colors=C["text_muted"], labelsize=8, length=3, width=0.7)
    ax.set_title(title, color=C["text_dim"], fontsize=9, fontweight="600", pad=6, loc="left")
    ax.set_xlabel(xlabel, color=C["text_muted"], fontsize=8)
    ax.set_ylabel(ylabel, color=C["text_muted"], fontsize=8)
    ax.grid(True, color=C["border"], linewidth=0.35, alpha=0.7, linestyle="--")
    ax.set_axisbelow(True)
    return ax


class PlotCanvas(FigureCanvas):
    def __init__(self, rows=1, cols=1, height=5, width=10):
        self.fig = Figure(figsize=(width, height))
        self.fig.set_facecolor(C["plot_bg"])
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = [self.fig.add_subplot(rows, cols, i + 1) for i in range(rows * cols)]
        for ax in self.axes:
            style_ax(ax)

    def reset(self):
        for ax in self.axes:
            ax.cla()
            style_ax(ax)

    def flush(self, pad=2.2, h_pad=2.8, w_pad=2.0):
        self.fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        self.draw()


# ============================================================
# DSP CORE  (all fixed + extended)
# ============================================================
def compute_ste(audio, frame_size=512, hop_size=256):
    """Short-Time Energy per frame."""
    if len(audio) < frame_size:
        return np.array([np.sum(audio ** 2)])
    n = (len(audio) - frame_size) // hop_size
    return np.array([
        float(np.sum(audio[i * hop_size: i * hop_size + frame_size] ** 2))
        for i in range(n)
    ])


def compute_zcr(audio, frame_size=512, hop_size=256):
    """Zero Crossing Rate per frame (crossings / sample)."""
    if len(audio) < frame_size:
        signs = np.sign(audio)
        signs[signs == 0] = 1
        return np.array([float(np.sum(np.abs(np.diff(signs)))) / (2 * len(audio))])
    n = (len(audio) - frame_size) // hop_size
    out = []
    for i in range(n):
        frame = audio[i * hop_size: i * hop_size + frame_size]
        signs = np.sign(frame)
        signs[signs == 0] = 1          # treat 0 as positive to avoid double-count
        zcr = float(np.sum(np.abs(np.diff(signs)))) / (2 * frame_size)
        out.append(zcr)
    return np.array(out)


def compute_zcr_detailed(audio, frame_size=512, hop_size=256, sr=22050):
    """
    Returns a list of dicts per frame:
      frame_idx, start_sample, end_sample, start_time_s, end_time_s,
      frame_length, zcr, n_crossings, polarity (dominant sign),
      mean_amplitude, sample_values (first 8)
    """
    if len(audio) < frame_size:
        frame_size = len(audio)
        hop_size   = len(audio)

    n = max(1, (len(audio) - frame_size) // hop_size)
    rows = []
    for i in range(n):
        start = i * hop_size
        end   = start + frame_size
        frame = audio[start:end]

        signs = np.sign(frame)
        signs[signs == 0] = 1
        n_cross = int(np.sum(np.abs(np.diff(signs)))) // 2
        zcr     = n_cross / frame_size

        # dominant polarity
        pos = int(np.sum(frame > 0))
        neg = int(np.sum(frame < 0))
        if pos > neg:
            polarity = "Positive"
        elif neg > pos:
            polarity = "Negative"
        else:
            polarity = "Mixed"

        rows.append({
            "frame_idx":    i,
            "start_sample": start,
            "end_sample":   end,
            "start_time_s": start / sr,
            "end_time_s":   end   / sr,
            "frame_length": frame_size,
            "zcr":          zcr,
            "n_crossings":  n_cross,
            "polarity":     polarity,
            "mean_amp":     float(np.mean(np.abs(frame))),
            "samples_preview": frame[:8].tolist(),
        })
    return rows


def compute_rms(audio):
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def compute_snr(original, noisy):
    """SNR in dB. Returns np.inf when noise is zero."""
    if original is None or noisy is None:
        return None
    noise     = (noisy - original).astype(np.float64)
    noise_pwr = float(np.mean(noise ** 2))
    sig_pwr   = float(np.mean(original.astype(np.float64) ** 2))
    if noise_pwr < 1e-15:
        return np.inf
    if sig_pwr < 1e-15:
        return -np.inf
    return 10.0 * np.log10(sig_pwr / noise_pwr)


def periodicity_score(audio, sr, frame=4096):
    """Returns (peak_acf_value, full_normalized_acf_array)."""
    x = audio[:min(frame, len(audio))].astype(np.float64)
    if len(x) < 4:
        return 0.0, np.zeros(4)
    acf      = np.correlate(x, x, mode="full")[len(x) - 1:]
    denom    = acf[0] if abs(acf[0]) > 1e-15 else 1.0
    acf_norm = acf / denom
    # search from 20 samples up to half the frame
    half = max(21, len(acf_norm) // 2)
    search   = acf_norm[20:half]
    peak_val = float(np.max(search)) if len(search) > 0 else 0.0
    return peak_val, acf_norm


# ============================================================
# NOISE GENERATORS
# ============================================================
def _scale_noise(noise, audio, snr_db):
    sig_pwr = float(np.mean(audio.astype(np.float64) ** 2))
    if sig_pwr < 1e-15:
        return noise
    target  = sig_pwr / (10.0 ** (snr_db / 10.0))
    cur     = float(np.mean(noise.astype(np.float64) ** 2))
    if cur < 1e-15:
        return noise
    return noise * np.sqrt(target / cur)


def noise_gaussian(audio, snr_db):
    n = np.random.randn(len(audio)).astype(np.float32)
    return np.clip(audio + _scale_noise(n, audio, snr_db).astype(np.float32), -1.0, 1.0)


def noise_pink(audio, snr_db):
    white = np.random.randn(len(audio))
    b = np.array([ 0.049922035, -0.095993537,  0.050612699, -0.004408786])
    a = np.array([ 1.0,         -2.494956002,  2.017265875, -0.522189400])
    pink = sp_signal.lfilter(b, a, white)
    std  = float(np.std(pink))
    if std > 1e-12:
        pink /= std
    return np.clip(audio + _scale_noise(pink.astype(np.float32), audio, snr_db).astype(np.float32), -1.0, 1.0)


def noise_impulse(audio, snr_db, density=0.008):
    noisy   = audio.copy()
    n_imp   = max(1, int(len(audio) * density))
    idx     = np.random.choice(len(audio), n_imp, replace=False)
    sig_pwr = float(np.mean(audio.astype(np.float64) ** 2))
    amp     = np.sqrt(max(sig_pwr, 1e-15) / (10.0 ** (snr_db / 10.0)) / (density + 1e-12))
    noisy[idx] += (np.random.choice([-1.0, 1.0], n_imp) * amp).astype(np.float32)
    return np.clip(noisy, -1.0, 1.0)


def noise_bandlimited(audio, snr_db, sr=22050, lo=300, hi=3400):
    nyq  = sr / 2.0
    lo_n = max(0.001, lo / nyq)
    hi_n = min(0.999, hi / nyq)
    b, a = sp_signal.butter(4, [lo_n, hi_n], btype="band")
    raw  = np.random.randn(len(audio))
    filt = sp_signal.lfilter(b, a, raw)
    std  = float(np.std(filt))
    if std > 1e-12:
        filt /= std
    return np.clip(audio + _scale_noise(filt.astype(np.float32), audio, snr_db).astype(np.float32), -1.0, 1.0)


def noise_harmonic(audio, snr_db):
    level = max(0.02, 10.0 ** (-abs(snr_db) / 40.0))
    mx    = float(np.max(np.abs(audio))) + 1e-12
    dist  = audio + (level * np.sin(2.0 * np.pi * audio / mx) * mx).astype(np.float32)
    return np.clip(dist, -1.0, 1.0)


NOISE_FUNCTIONS = {
    "Gaussian (White)":           noise_gaussian,
    "Pink (1/f)":                 noise_pink,
    "Impulse (Salt & Pepper)":    noise_impulse,
    "Band-Limited (300-3400 Hz)": noise_bandlimited,
    "Harmonic Distortion":        noise_harmonic,
}


# ============================================================
# RECORDING THREAD
# ============================================================
class RecordThread(QThread):
    finished  = pyqtSignal(np.ndarray, int)
    vu_update = pyqtSignal(float)

    def __init__(self, duration=3, sr=22050):
        super().__init__()
        self.duration = duration
        self.sr       = sr
        self._stop    = False

    def stop(self):
        self._stop = True

    def run(self):
        chunks     = []
        chunk_size = int(self.sr * 0.04)      # 40 ms chunks
        n_chunks   = int(np.ceil(self.duration / 0.04))
        try:
            with sd.InputStream(samplerate=self.sr, channels=1,
                                dtype="float32", blocksize=chunk_size) as stream:
                for _ in range(n_chunks):
                    if self._stop:
                        break
                    data, _ = stream.read(chunk_size)
                    mono    = data[:, 0].copy()
                    chunks.append(mono)
                    self.vu_update.emit(float(np.sqrt(np.mean(mono ** 2))))
        except Exception as exc:
            print(f"[RecordThread] {exc}")
        if chunks:
            self.finished.emit(np.concatenate(chunks).astype(np.float32), self.sr)


# ============================================================
# SHARED UI COMPONENTS
# ============================================================
class StatCard(QFrame):
    def __init__(self, label, unit=""):
        super().__init__()
        self.setObjectName("stat_card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(3)

        self.lbl_name = QLabel(label.upper())
        self.lbl_name.setStyleSheet(
            f"color:{C['text_muted']}; font-size:10px; font-weight:700; letter-spacing:1px;"
        )
        self.lbl_val = QLabel("—")
        self.lbl_val.setObjectName("stat_value")
        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet(f"color:{C['text_muted']}; font-size:10px;")

        lay.addWidget(self.lbl_name)
        lay.addWidget(self.lbl_val)
        lay.addWidget(self.lbl_unit)

    def set_value(self, val, fmt="{:.5f}", color=None):
        col = color or C["accent"]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            self.lbl_val.setText("—")
        elif isinstance(val, float) and np.isinf(val):
            self.lbl_val.setText("inf" if val > 0 else "-inf")
        else:
            self.lbl_val.setText(fmt.format(val))
        self.lbl_val.setStyleSheet(f"color:{col}; font-size:18px; font-weight:700;")

    def set_text(self, text, color=None):
        col = color or C["accent"]
        self.lbl_val.setText(str(text))
        self.lbl_val.setStyleSheet(f"color:{col}; font-size:17px; font-weight:700;")


def section_header(text):
    w   = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 4, 0, 4)
    row.setSpacing(8)
    bar = QFrame()
    bar.setFixedSize(3, 16)
    bar.setStyleSheet(
        f"background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        f"stop:0 {C['accent']},stop:1 {C['accent2']});"
        f"border-radius:2px;"
    )
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color:{C['text_dim']}; font-size:10px; font-weight:700; letter-spacing:1.2px;"
    )
    row.addWidget(bar)
    row.addWidget(lbl)
    row.addStretch()
    return w


def hdivider():
    d = QFrame()
    d.setObjectName("divider")
    return d


def make_sidebar(width=268):
    w = QWidget()
    w.setFixedWidth(width)
    w.setStyleSheet(
        f"background-color:{C['surface']};"
        f"border-right:1px solid {C['border']};"
    )
    return w


def lbl_dim(text):
    l = QLabel(text)
    l.setStyleSheet(f"color:{C['text_dim']}; font-size:12px;")
    return l


# ============================================================
# PART 1 — SIGNAL FEATURE EXTRACTION
# ============================================================
class FeatureTab(QWidget):
    def __init__(self):
        super().__init__()
        self.audio       = None
        self.audio_noisy = None
        self.sr          = 22050
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sb = make_sidebar()
        sv = QVBoxLayout(sb)
        sv.setContentsMargins(16, 20, 16, 16)
        sv.setSpacing(12)

        self._sb_title(sv, "Signal Analyzer", "Part 1 — Feature Extraction")
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Signal Source"))
        self.btn_load  = QPushButton("Load Audio File")
        self.btn_synth = QPushButton("Generate Synthetic")
        self.lbl_file  = QLabel("No signal loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_synth.clicked.connect(self.gen_synth)
        sv.addWidget(self.btn_load)
        sv.addWidget(self.btn_synth)
        sv.addWidget(self.lbl_file)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Noise Injection"))
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(list(NOISE_FUNCTIONS.keys()))
        sv.addWidget(self.noise_combo)

        snr_row = QHBoxLayout()
        snr_row.addWidget(lbl_dim("SNR"))
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-5, 60)
        self.snr_spin.setValue(20)
        self.snr_spin.setSuffix(" dB")
        self.snr_spin.setSingleStep(5)
        snr_row.addWidget(self.snr_spin)
        sv.addLayout(snr_row)

        self.btn_apply_noise = QPushButton("Apply Noise")
        self.btn_apply_noise.setEnabled(False)
        self.btn_apply_noise.clicked.connect(self.apply_noise)
        self.btn_clear_noise = QPushButton("Clear Noise")
        self.btn_clear_noise.setObjectName("btn_danger")
        self.btn_clear_noise.setEnabled(False)
        self.btn_clear_noise.clicked.connect(self.clear_noise)
        sv.addWidget(self.btn_apply_noise)
        sv.addWidget(self.btn_clear_noise)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Frame Parameters"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.addWidget(lbl_dim("Frame Size"), 0, 0)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(128, 4096)
        self.frame_spin.setValue(512)
        self.frame_spin.setSingleStep(128)
        grid.addWidget(self.frame_spin, 0, 1)
        grid.addWidget(lbl_dim("Hop Size"), 1, 0)
        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(64, 2048)
        self.hop_spin.setValue(256)
        self.hop_spin.setSingleStep(64)
        grid.addWidget(self.hop_spin, 1, 1)
        sv.addLayout(grid)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Playback"))
        pb = QHBoxLayout()
        pb.setSpacing(6)
        self.btn_play_orig  = QPushButton("Original")
        self.btn_play_orig.setObjectName("btn_play")
        self.btn_play_orig.setEnabled(False)
        self.btn_play_noisy = QPushButton("Noisy")
        self.btn_play_noisy.setObjectName("btn_play")
        self.btn_play_noisy.setEnabled(False)
        self.btn_play_orig.clicked.connect(lambda: sd.play(self.audio, self.sr))
        self.btn_play_noisy.clicked.connect(self._play_noisy)
        pb.addWidget(self.btn_play_orig)
        pb.addWidget(self.btn_play_noisy)
        sv.addLayout(pb)
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(sd.stop)
        sv.addWidget(btn_stop)
        sv.addStretch()

        self.btn_analyze = QPushButton("Analyze Signal")
        self.btn_analyze.setObjectName("btn_primary")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.analyze)
        sv.addWidget(self.btn_analyze)

        root.addWidget(sb)

        # ── Content ───────────────────────────────────────────────────────────
        content = QWidget()
        cv = QVBoxLayout(content)
        cv.setContentsMargins(20, 20, 20, 16)
        cv.setSpacing(14)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.c_mean = StatCard("Mean",     "amplitude")
        self.c_var  = StatCard("Variance", "")
        self.c_std  = StatCard("Std Dev",  "")
        self.c_rms  = StatCard("RMS",      "")
        self.c_ste  = StatCard("STE mean", "energy/frame")
        self.c_zcr  = StatCard("ZCR mean", "crossings/sample")
        self.c_snr  = StatCard("SNR",      "dB")
        for c in (self.c_mean, self.c_var, self.c_std, self.c_rms,
                  self.c_ste,  self.c_zcr,  self.c_snr):
            cards.addWidget(c)
        cv.addLayout(cards)

        self.canvas = PlotCanvas(rows=3, cols=2, height=9)
        cv.addWidget(self.canvas)
        root.addWidget(content)

    @staticmethod
    def _sb_title(lay, title, sub):
        t = QLabel(title)
        t.setStyleSheet(f"color:{C['text']}; font-size:16px; font-weight:700;")
        s = QLabel(sub)
        s.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        lay.addWidget(t)
        lay.addWidget(s)

    # ── Signal sources ───────────────────────────────────────────────────────
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.flac *.ogg *.aiff *.mp3)"
        )
        if not path:
            return
        try:
            audio, sr = sf.read(path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self.audio       = audio.astype(np.float32)
            self.sr          = int(sr)
            self.audio_noisy = None
            dur              = len(audio) / sr
            self.lbl_file.setText(f"{os.path.basename(path)}\n{dur:.2f}s  |  {sr} Hz")
            self._signal_ready()
        except Exception as exc:
            self.lbl_file.setText(f"Error: {exc}")

    def gen_synth(self):
        sr = 22050
        t  = np.linspace(0, 2.5, int(2.5 * sr), endpoint=False)
        audio = (
            0.55 * np.sin(2 * np.pi * 200 * t)
            + 0.30 * np.sin(2 * np.pi * 400 * t)
            + 0.15 * np.sin(2 * np.pi * 600 * t + 0.4)
            + 0.10 * np.sin(2 * np.pi * 800 * t)
            + 0.04 * np.sin(2 * np.pi * 1200 * t)
            + 0.025 * np.random.randn(len(t))
        )
        peak = np.max(np.abs(audio))
        if peak > 1e-12:
            audio /= peak
        self.audio       = audio.astype(np.float32)
        self.sr          = sr
        self.audio_noisy = None
        self.lbl_file.setText("Synthetic: 200 Hz harmonic series\n2.50s  |  22050 Hz")
        self._signal_ready()

    def _signal_ready(self):
        self.btn_analyze.setEnabled(True)
        self.btn_play_orig.setEnabled(True)
        self.btn_apply_noise.setEnabled(True)
        self.btn_play_noisy.setEnabled(False)
        self.btn_clear_noise.setEnabled(False)

    # ── Noise ────────────────────────────────────────────────────────────────
    def apply_noise(self):
        if self.audio is None:
            return
        try:
            fn = NOISE_FUNCTIONS[self.noise_combo.currentText()]
            self.audio_noisy = fn(self.audio, self.snr_spin.value())
            self.btn_play_noisy.setEnabled(True)
            self.btn_clear_noise.setEnabled(True)
            self.analyze()
        except Exception as exc:
            print(f"[apply_noise] {exc}")

    def clear_noise(self):
        self.audio_noisy = None
        self.btn_play_noisy.setEnabled(False)
        self.btn_clear_noise.setEnabled(False)
        self.analyze()

    def _play_noisy(self):
        if self.audio_noisy is not None:
            sd.play(self.audio_noisy, self.sr)

    # ── Analysis ─────────────────────────────────────────────────────────────
    def analyze(self):
        if self.audio is None:
            return
        audio = self.audio
        noisy = self.audio_noisy
        sr    = self.sr
        fs    = self.frame_spin.value()
        hs    = self.hop_spin.value()
        t     = np.arange(len(audio)) / sr

        ste   = compute_ste(audio, fs, hs)
        zcr   = compute_zcr(audio, fs, hs)
        rms_v = compute_rms(audio)
        snr_v = compute_snr(audio, noisy)

        self.c_mean.set_value(float(np.mean(audio)),  "{:.6f}")
        self.c_var.set_value( float(np.var(audio)),   "{:.6f}")
        self.c_std.set_value( float(np.std(audio)),   "{:.6f}")
        self.c_rms.set_value( rms_v,                  "{:.6f}")
        self.c_ste.set_value( float(np.mean(ste)),    "{:.5f}")
        self.c_zcr.set_value( float(np.mean(zcr)),    "{:.4f}")
        if snr_v is not None:
            self.c_snr.set_value(snr_v, "{:.2f}",
                                 color=C["green"] if snr_v > 20 else C["amber"] if snr_v > 5 else C["red"])
        else:
            self.c_snr.set_value(None)

        t_f  = np.arange(len(ste)) * hs / sr
        N    = len(audio)
        freq = fftfreq(N, 1 / sr)[:N // 2]
        mag  = np.abs(fft(audio.astype(np.float64)))[:N // 2]

        self.canvas.reset()
        axs = self.canvas.axes

        # 0 — Waveform
        ax = axs[0]
        ax.plot(t, audio, color=C["waveform"], linewidth=0.65, alpha=0.9)
        if noisy is not None:
            ax.plot(t, noisy, color=C["noisy"], linewidth=0.45, alpha=0.5, label="Noisy")
            ax.legend(fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                      framealpha=0.9, edgecolor=C["border"])
        style_ax(ax, "Waveform", "Time (s)", "Amplitude")

        # 1 — Log Spectrum
        ax = axs[1]
        ax.fill_between(freq, mag, color=C["spectrum"], alpha=0.22)
        ax.plot(freq, mag, color=C["spectrum"], linewidth=0.8)
        ax.set_xlim(0, min(sr // 2, 8000))
        ax.set_yscale("log")
        style_ax(ax, "Frequency Spectrum — log magnitude", "Frequency (Hz)", "Magnitude")

        # 2 — STE
        ax = axs[2]
        ste_m = float(np.mean(ste))
        ax.fill_between(t_f, ste, color=C["ste_line"], alpha=0.28)
        ax.plot(t_f, ste, color=C["ste_line"], linewidth=1.0)
        ax.axhline(ste_m, color=C["amber"], linewidth=1.0, linestyle="--", alpha=0.85,
                   label=f"Mean  {ste_m:.5f}")
        ax.legend(fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                  framealpha=0.9, edgecolor=C["border"])
        style_ax(ax, "Short-Time Energy (STE)", "Time (s)", "Energy")

        # 3 — ZCR
        ax = axs[3]
        zcr_m = float(np.mean(zcr))
        ax.fill_between(t_f, zcr, color=C["zcr_line"], alpha=0.28)
        ax.plot(t_f, zcr, color=C["zcr_line"], linewidth=1.0)
        ax.axhline(zcr_m, color=C["accent"], linewidth=1.0, linestyle="--", alpha=0.85,
                   label=f"Mean  {zcr_m:.4f}")
        ax.legend(fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                  framealpha=0.9, edgecolor=C["border"])
        style_ax(ax, "Zero Crossing Rate (ZCR)", "Time (s)", "ZCR")

        # 4 — Spectrogram
        ax = axs[4]
        f_sg, t_sg, Sxx = sp_signal.spectrogram(audio, sr, nperseg=min(512, len(audio) // 4 or 512),
                                                  noverlap=384)
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-15))
        im = ax.pcolormesh(t_sg, f_sg, Sxx_db, cmap="inferno", shading="gouraud",
                           vmin=-90, vmax=0)
        ax.set_ylim(0, min(8000, sr // 2))
        style_ax(ax, "Spectrogram", "Time (s)", "Frequency (Hz)")
        cb = self.canvas.fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cb.ax.tick_params(labelsize=7, colors=C["text_muted"])
        cb.set_label("dB", color=C["text_muted"], fontsize=8)

        # 5 — Autocorrelation  (or noise spectrogram when noise active)
        ax = axs[5]
        if noisy is not None and snr_v is not None:
            noise_only   = (noisy - audio)
            f_n, t_n, Sn = sp_signal.spectrogram(noise_only, sr,
                                                  nperseg=min(512, len(noise_only) // 4 or 512),
                                                  noverlap=384)
            Sn_db = 10 * np.log10(np.maximum(Sn, 1e-15))
            ax.pcolormesh(t_n, f_n, Sn_db, cmap="plasma", shading="gouraud")
            ax.set_ylim(0, min(8000, sr // 2))
            snr_str = f"{snr_v:.1f}" if not np.isinf(snr_v) else "inf"
            style_ax(ax, f"Noise Spectrogram   SNR = {snr_str} dB", "Time (s)", "Frequency (Hz)")
        else:
            per_val, acf_norm = periodicity_score(audio, sr)
            lag_ms = np.arange(len(acf_norm)) / sr * 1000
            disp   = min(400, len(acf_norm))
            ax.fill_between(lag_ms[:disp], acf_norm[:disp], color=C["acf"], alpha=0.22)
            ax.plot(lag_ms[:disp], acf_norm[:disp], color=C["acf"], linewidth=0.9)
            ax.axhline(0, color=C["border_hi"], linewidth=0.5)
            style_ax(ax, "Autocorrelation — Periodicity", "Lag (ms)", "Normalized ACF")

        self.canvas.flush()


# ============================================================
# PART 2 — PHONEME ANALYSIS
# ============================================================
PHONEME_GUIDE = [
    ("/ae/",  "cat",   "Low ZCR  |  High STE  |  Open front vowel"),
    ("/i/",   "bit",   "Low ZCR  |  High STE  |  Close front vowel"),
    ("/u/",   "book",  "Low ZCR  |  High STE  |  Close back vowel"),
    ("/s/",   "hiss",  "High ZCR  |  Low STE  |  Unvoiced fricative"),
    ("/f/",   "fee",   "High ZCR  |  Low STE  |  Labiodental fricative"),
    ("/m/",   "hum",   "Low ZCR  |  Mid STE  |  Nasal sonorant"),
    ("/n/",   "nano",  "Low ZCR  |  Mid STE  |  Alveolar nasal"),
    ("/sh/",  "shoe",  "Mid ZCR  |  Low STE  |  Palatal fricative"),
]


class PhonemeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.audio = None
        self.sr    = 22050
        self._rec  = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sb = make_sidebar()
        sv = QVBoxLayout(sb)
        sv.setContentsMargins(16, 20, 16, 16)
        sv.setSpacing(12)

        t = QLabel("Phoneme Analyzer")
        t.setStyleSheet(f"color:{C['text']}; font-size:16px; font-weight:700;")
        s = QLabel("Part 2 — Time-Domain Characteristics")
        s.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        sv.addWidget(t); sv.addWidget(s)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Recording"))
        dur_row = QHBoxLayout()
        dur_row.addWidget(lbl_dim("Duration"))
        self.dur_spin = QSpinBox()
        self.dur_spin.setRange(1, 15)
        self.dur_spin.setValue(3)
        self.dur_spin.setSuffix(" s")
        dur_row.addWidget(self.dur_spin)
        sv.addLayout(dur_row)

        sv.addWidget(lbl_dim("Mic Level"))
        self.vu_bar = QProgressBar()
        self.vu_bar.setObjectName("vu_bar")
        self.vu_bar.setRange(0, 100)
        self.vu_bar.setTextVisible(False)
        sv.addWidget(self.vu_bar)

        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setObjectName("btn_record")
        self.btn_record.clicked.connect(self.toggle_record)
        sv.addWidget(self.btn_record)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        sv.addWidget(self.lbl_status)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Or Load File"))
        self.btn_load = QPushButton("Load Audio File")
        self.btn_load.clicked.connect(self.load_file)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        sv.addWidget(self.btn_load)
        sv.addWidget(self.lbl_file)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Playback"))
        pb = QHBoxLayout()
        pb.setSpacing(6)
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("btn_play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(lambda: self.audio is not None and sd.play(self.audio, self.sr))
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(sd.stop)
        pb.addWidget(self.btn_play)
        pb.addWidget(btn_stop)
        sv.addLayout(pb)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Phoneme Reference"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:transparent; border:none;")
        rw = QWidget()
        rw.setStyleSheet("background:transparent;")
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(0, 4, 0, 4)
        rl.setSpacing(5)
        for ph, word, desc in PHONEME_GUIDE:
            row = QHBoxLayout()
            row.setSpacing(8)
            lp = QLabel(ph); lp.setFixedWidth(38)
            lp.setStyleSheet(f"color:{C['accent']}; font-weight:700; font-size:12px;")
            lw = QLabel(word); lw.setFixedWidth(40)
            lw.setStyleSheet(f"color:{C['text']}; font-size:12px; font-weight:600;")
            ld = QLabel(desc)
            ld.setStyleSheet(f"color:{C['text_muted']}; font-size:10px;")
            ld.setWordWrap(True)
            row.addWidget(lp); row.addWidget(lw); row.addWidget(ld)
            rw2 = QWidget(); rw2.setLayout(row)
            rl.addWidget(rw2)
        scroll.setWidget(rw)
        sv.addWidget(scroll)
        sv.addStretch()

        self.btn_analyze = QPushButton("Analyze Phoneme")
        self.btn_analyze.setObjectName("btn_primary")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.analyze)
        sv.addWidget(self.btn_analyze)

        root.addWidget(sb)

        # ── Content ───────────────────────────────────────────────────────────
        content = QWidget()
        cv = QVBoxLayout(content)
        cv.setContentsMargins(20, 20, 20, 16)
        cv.setSpacing(14)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.c_dur  = StatCard("Duration",       "seconds")
        self.c_amp  = StatCard("Mean Amplitude", "")
        self.c_var  = StatCard("Amplitude Var",  "")
        self.c_zcr  = StatCard("Mean ZCR",       "crossings/sample")
        self.c_ste  = StatCard("Mean STE",       "energy/frame")
        self.c_per  = StatCard("Periodicity",    "")
        for c in (self.c_dur, self.c_amp, self.c_var, self.c_zcr, self.c_ste, self.c_per):
            cards.addWidget(c)
        cv.addLayout(cards)

        banner = QFrame()
        banner.setObjectName("card")
        banner.setFixedHeight(52)
        bl = QHBoxLayout(banner)
        bl.setContentsMargins(16, 0, 16, 0)
        bl.setSpacing(14)
        cls_lbl = QLabel("CLASSIFICATION")
        cls_lbl.setStyleSheet(
            f"color:{C['text_muted']}; font-size:10px; font-weight:700; letter-spacing:1px;"
        )
        self.lbl_detected = QLabel("Record or load audio to begin analysis")
        self.lbl_detected.setStyleSheet(
            f"color:{C['text_dim']}; font-size:14px; font-weight:600;"
        )
        bl.addWidget(cls_lbl)
        bl.addWidget(self.lbl_detected)
        bl.addStretch()
        cv.addWidget(banner)

        self.canvas = PlotCanvas(rows=2, cols=3, height=7)
        cv.addWidget(self.canvas)
        root.addWidget(content)

    # ── Record ────────────────────────────────────────────────────────────────
    def toggle_record(self):
        if self._rec and self._rec.isRunning():
            self._rec.stop()
            self._set_record_idle()
            self.lbl_status.setText("Stopped early")
            return
        self.btn_record.setObjectName("btn_record_active")
        self.btn_record.setText("Stop Recording")
        self.btn_record.setStyle(self.btn_record.style())
        self.lbl_status.setText(f"Recording {self.dur_spin.value()}s ...")
        self.vu_bar.setValue(0)
        self._rec = RecordThread(self.dur_spin.value(), self.sr)
        self._rec.finished.connect(self.on_recorded)
        self._rec.vu_update.connect(lambda v: self.vu_bar.setValue(int(min(100, v * 600))))
        self._rec.start()

    def _set_record_idle(self):
        self.btn_record.setObjectName("btn_record")
        self.btn_record.setText("Start Recording")
        self.btn_record.setStyle(self.btn_record.style())

    def on_recorded(self, audio, sr):
        self.audio = audio
        self.sr    = int(sr)
        self._set_record_idle()
        self.lbl_status.setText(f"Recorded  {len(audio) / sr:.2f}s")
        self.vu_bar.setValue(0)
        self.btn_play.setEnabled(True)
        self.btn_analyze.setEnabled(True)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.flac *.ogg *.aiff *.mp3)"
        )
        if not path:
            return
        try:
            audio, sr = sf.read(path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self.audio = audio.astype(np.float32)
            self.sr    = int(sr)
            self.lbl_file.setText(f"{os.path.basename(path)}\n{len(audio) / sr:.2f}s  |  {sr} Hz")
            self.btn_play.setEnabled(True)
            self.btn_analyze.setEnabled(True)
        except Exception as exc:
            self.lbl_file.setText(f"Error: {exc}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    def analyze(self):
        if self.audio is None:
            return
        audio = self.audio
        sr    = self.sr
        t     = np.arange(len(audio)) / sr
        fs, hs = 512, 256

        ste = compute_ste(audio, fs, hs)
        zcr = compute_zcr(audio, fs, hs)
        t_f = np.arange(len(ste)) * hs / sr
        env = np.abs(hilbert(audio.astype(np.float64))).astype(np.float32)

        per_val, acf_norm = periodicity_score(audio, sr)
        per_lbl = "High" if per_val > 0.5 else ("Medium" if per_val > 0.25 else "Low")
        per_col = {
            "High": C["green"], "Medium": C["amber"], "Low": C["red"]
        }[per_lbl]

        self.c_dur.set_value(len(audio) / sr,       "{:.3f}")
        self.c_amp.set_value(float(np.mean(np.abs(audio))), "{:.5f}")
        self.c_var.set_value(float(np.var(audio)),  "{:.6f}")
        self.c_zcr.set_value(float(np.mean(zcr)),   "{:.4f}")
        self.c_ste.set_value(float(np.mean(ste)),   "{:.5f}")
        self.c_per.set_text(per_lbl, per_col)

        # Classification
        zcr_m = float(np.mean(zcr))
        if zcr_m < 0.08 and per_val > 0.4:
            ptype, pcol = "Voiced / Vowel-like",          C["green"]
        elif zcr_m > 0.20:
            ptype, pcol = "Unvoiced Fricative",            C["amber"]
        elif per_val > 0.3:
            ptype, pcol = "Voiced Consonant / Sonorant",   C["accent"]
        else:
            ptype, pcol = "Plosive / Mixed",               C["red"]
        self.lbl_detected.setText(ptype)
        self.lbl_detected.setStyleSheet(f"color:{pcol}; font-size:15px; font-weight:700;")

        N    = len(audio)
        freq = fftfreq(N, 1 / sr)[:N // 2]
        mag  = np.abs(fft(audio.astype(np.float64)))[:N // 2]

        self.canvas.reset()
        axs = self.canvas.axes

        # 0 — Waveform + envelope
        ax = axs[0]
        ax.plot(t, audio, color=C["waveform"], linewidth=0.65, alpha=0.85)
        ax.plot(t,  env,  color=C["envelope"], linewidth=1.2, alpha=0.9, label="Envelope")
        ax.plot(t, -env,  color=C["envelope"], linewidth=1.2, alpha=0.9)
        ax.axhline(0, color=C["border_hi"], linewidth=0.5)
        ax.legend(fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                  framealpha=0.9, edgecolor=C["border"])
        style_ax(ax, "Waveform + Amplitude Envelope", "Time (s)", "Amplitude")

        # 1 — STE
        ax = axs[1]
        ste_m = float(np.mean(ste))
        ax.fill_between(t_f, ste, color=C["ste_line"], alpha=0.28)
        ax.plot(t_f, ste, color=C["ste_line"], linewidth=1.0)
        ax.axhline(ste_m, color=C["amber"], linewidth=1.0, linestyle="--", alpha=0.85,
                   label=f"Mean  {ste_m:.5f}")
        ax.legend(fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                  framealpha=0.9, edgecolor=C["border"])
        style_ax(ax, "Short-Time Energy", "Time (s)", "Energy")

        # 2 — ZCR
        ax = axs[2]
        ax.fill_between(t_f, zcr, color=C["zcr_line"], alpha=0.28)
        ax.plot(t_f, zcr, color=C["zcr_line"], linewidth=1.0)
        ax.axhline(zcr_m, color=C["accent"], linewidth=1.0, linestyle="--", alpha=0.85,
                   label=f"Mean  {zcr_m:.4f}")
        ax.legend(fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                  framealpha=0.9, edgecolor=C["border"])
        style_ax(ax, "Zero Crossing Rate (ZCR)", "Time (s)", "ZCR")

        # 3 — Spectrogram
        ax = axs[3]
        nperseg = min(512, max(16, len(audio) // 8))
        noverlap = min(nperseg - 1, 384)
        f_sg, t_sg, Sxx = sp_signal.spectrogram(audio, sr, nperseg=nperseg, noverlap=noverlap)
        im = ax.pcolormesh(t_sg, f_sg, 10 * np.log10(np.maximum(Sxx, 1e-15)),
                           cmap="magma", shading="gouraud", vmin=-90, vmax=0)
        ax.set_ylim(0, min(8000, sr // 2))
        style_ax(ax, "Spectrogram", "Time (s)", "Frequency (Hz)")
        cb = self.canvas.fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cb.ax.tick_params(labelsize=7, colors=C["text_muted"])
        cb.set_label("dB", color=C["text_muted"], fontsize=8)

        # 4 — Autocorrelation
        ax = axs[4]
        lag_ms = np.arange(len(acf_norm)) / sr * 1000
        disp   = min(350, len(acf_norm))
        ax.fill_between(lag_ms[:disp], acf_norm[:disp], color=C["acf"], alpha=0.22)
        ax.plot(lag_ms[:disp], acf_norm[:disp], color=C["acf"], linewidth=0.9)
        ax.axhline(0, color=C["border_hi"], linewidth=0.5)
        ax.set_ylim(-0.65, 1.05)
        style_ax(ax, f"Autocorrelation — Periodicity: {per_lbl} ({per_val:.3f})",
                 "Lag (ms)", "Normalized ACF")

        # 5 — Frequency content
        ax = axs[5]
        ax.fill_between(freq, mag, color=C["spectrum"], alpha=0.22)
        ax.plot(freq, mag, color=C["spectrum"], linewidth=0.8)
        ax.set_xlim(0, min(8000, sr // 2))
        ax.set_yscale("log")
        style_ax(ax, "Frequency Content — log magnitude", "Frequency (Hz)", "Magnitude")

        self.canvas.flush()


# ============================================================
# PART 3 — ZCR WINDOW INSPECTOR
# ============================================================
class ZCRInspectorTab(QWidget):
    """
    Detailed per-frame ZCR analysis:
      - Configurable frame length, hop size, selected window number
      - Top: full signal with highlighted selected window + all ZC markers
      - Middle: zoomed selected window with polarity-colored samples,
                sign function overlay, and ZC crossing arrows
      - Bottom-left: bar chart — ZCR value per frame
      - Bottom-right: data table — Frame #, Start, End, Length, ZCR, Crossings, Polarity
    """

    def __init__(self):
        super().__init__()
        self.audio   = None
        self.sr      = 22050
        self._rec    = None
        self._frames = []          # list of dicts from compute_zcr_detailed
        self._build()

    # ──────────────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sb = make_sidebar(width=272)
        sv = QVBoxLayout(sb)
        sv.setContentsMargins(16, 20, 16, 16)
        sv.setSpacing(12)

        t = QLabel("ZCR Window Inspector")
        t.setStyleSheet(f"color:{C['text']}; font-size:16px; font-weight:700;")
        s = QLabel("Part 3 — Per-Frame ZCR Detail")
        s.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        sv.addWidget(t); sv.addWidget(s)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Signal Source"))
        self.btn_load  = QPushButton("Load Audio File")
        self.btn_synth = QPushButton("Generate Synthetic")
        self.lbl_file  = QLabel("No signal loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_synth.clicked.connect(self.gen_synth)
        sv.addWidget(self.btn_load)
        sv.addWidget(self.btn_synth)
        sv.addWidget(self.lbl_file)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Record"))
        dr = QHBoxLayout()
        dr.addWidget(lbl_dim("Duration"))
        self.dur_spin3 = QSpinBox()
        self.dur_spin3.setRange(1, 15)
        self.dur_spin3.setValue(2)
        self.dur_spin3.setSuffix(" s")
        dr.addWidget(self.dur_spin3)
        sv.addLayout(dr)
        sv.addWidget(lbl_dim("Mic Level"))
        self.vu_bar3 = QProgressBar()
        self.vu_bar3.setObjectName("vu_bar")
        self.vu_bar3.setRange(0, 100)
        self.vu_bar3.setTextVisible(False)
        sv.addWidget(self.vu_bar3)
        self.btn_record3 = QPushButton("Start Recording")
        self.btn_record3.setObjectName("btn_record")
        self.btn_record3.clicked.connect(self.toggle_record)
        sv.addWidget(self.btn_record3)
        self.lbl_rec3 = QLabel("Ready")
        self.lbl_rec3.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        sv.addWidget(self.lbl_rec3)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Frame Parameters"))
        pg = QGridLayout()
        pg.setHorizontalSpacing(8)
        pg.setVerticalSpacing(6)
        pg.addWidget(lbl_dim("Frame Size"), 0, 0)
        self.frame_spin3 = QSpinBox()
        self.frame_spin3.setRange(16, 4096)
        self.frame_spin3.setValue(256)
        self.frame_spin3.setSingleStep(64)
        pg.addWidget(self.frame_spin3, 0, 1)
        pg.addWidget(lbl_dim("Hop Size"), 1, 0)
        self.hop_spin3 = QSpinBox()
        self.hop_spin3.setRange(8, 2048)
        self.hop_spin3.setValue(128)
        self.hop_spin3.setSingleStep(64)
        pg.addWidget(self.hop_spin3, 1, 1)
        pg.addWidget(lbl_dim("Window #"), 2, 0)
        self.win_spin = QSpinBox()
        self.win_spin.setRange(0, 0)
        self.win_spin.setValue(0)
        self.win_spin.valueChanged.connect(self._update_window_detail)
        pg.addWidget(self.win_spin, 2, 1)
        sv.addLayout(pg)
        sv.addWidget(hdivider())

        sv.addWidget(section_header("Playback"))
        pb = QHBoxLayout(); pb.setSpacing(6)
        self.btn_play3 = QPushButton("Play")
        self.btn_play3.setObjectName("btn_play")
        self.btn_play3.setEnabled(False)
        self.btn_play3.clicked.connect(lambda: self.audio is not None and sd.play(self.audio, self.sr))
        btn_stop3 = QPushButton("Stop")
        btn_stop3.clicked.connect(sd.stop)
        pb.addWidget(self.btn_play3); pb.addWidget(btn_stop3)
        sv.addLayout(pb)
        sv.addStretch()

        self.btn_run = QPushButton("Run ZCR Analysis")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_analysis)
        sv.addWidget(self.btn_run)

        root.addWidget(sb)

        # ── Content ───────────────────────────────────────────────────────────
        content = QWidget()
        cv = QVBoxLayout(content)
        cv.setContentsMargins(20, 20, 20, 12)
        cv.setSpacing(12)

        # Stat cards
        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.c3_frames  = StatCard("Total Frames",   "")
        self.c3_fs      = StatCard("Frame Length",   "samples")
        self.c3_hs      = StatCard("Hop Size",       "samples")
        self.c3_zcr_w   = StatCard("Window ZCR",     "crossings/sample")
        self.c3_cross_w = StatCard("Crossings",      "in window")
        self.c3_pol_w   = StatCard("Polarity",       "dominant sign")
        for c in (self.c3_frames, self.c3_fs, self.c3_hs,
                  self.c3_zcr_w, self.c3_cross_w, self.c3_pol_w):
            cards.addWidget(c)
        cv.addLayout(cards)

        # Splitter: plots top | table bottom
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background:{C['border']}; height:2px; }}"
        )

        # ── Plot area ─────────────────────────────────────────────────────────
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)

        # Row 1: full signal + zoomed window (side by side)
        self.canvas_top = PlotCanvas(rows=1, cols=2, height=3)
        plot_layout.addWidget(self.canvas_top)

        # Row 2: ZCR bar chart
        self.canvas_bar = PlotCanvas(rows=1, cols=1, height=2.4)
        plot_layout.addWidget(self.canvas_bar)

        splitter.addWidget(plot_widget)

        # ── Table ─────────────────────────────────────────────────────────────
        tbl_widget = QWidget()
        tbl_layout = QVBoxLayout(tbl_widget)
        tbl_layout.setContentsMargins(0, 0, 0, 0)
        tbl_layout.setSpacing(4)

        tbl_hdr = QLabel("FRAME-BY-FRAME ZCR TABLE")
        tbl_hdr.setStyleSheet(
            f"color:{C['text_muted']}; font-size:10px; font-weight:700;"
            f"letter-spacing:1px; padding-top:4px;"
        )
        tbl_layout.addWidget(tbl_hdr)

        self.table = QTableWidget()
        cols = ["Frame", "Start (s)", "End (s)", "Length", "ZCR", "Crossings", "Polarity", "Mean |Amp|"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setMinimumHeight(160)
        self.table.itemSelectionChanged.connect(self._table_row_selected)
        tbl_layout.addWidget(self.table)

        splitter.addWidget(tbl_widget)
        splitter.setSizes([420, 200])

        cv.addWidget(splitter)
        root.addWidget(content)

    # ── Signal sources ────────────────────────────────────────────────────────
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.flac *.ogg *.aiff *.mp3)"
        )
        if not path:
            return
        try:
            audio, sr = sf.read(path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self.audio = audio.astype(np.float32)
            self.sr    = int(sr)
            dur = len(audio) / sr
            self.lbl_file.setText(f"{os.path.basename(path)}\n{dur:.2f}s  |  {sr} Hz")
            self._signal_ready()
        except Exception as exc:
            self.lbl_file.setText(f"Error: {exc}")

    def gen_synth(self):
        sr = 22050
        t  = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
        # Mix voiced + unvoiced regions
        voiced   = 0.6 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t)
        unvoiced = 0.25 * np.random.randn(len(t))
        # Voiced for first 0.75s, unvoiced for rest
        mid = len(t) // 2
        audio = voiced.copy()
        audio[mid:] = unvoiced[mid:]
        peak = np.max(np.abs(audio))
        if peak > 1e-12:
            audio /= peak
        self.audio = audio.astype(np.float32)
        self.sr    = sr
        self.lbl_file.setText("Synthetic: voiced (0-0.75s) + unvoiced (0.75-1.5s)\n1.50s  |  22050 Hz")
        self._signal_ready()

    def _signal_ready(self):
        self.btn_run.setEnabled(True)
        self.btn_play3.setEnabled(True)

    # ── Recording ─────────────────────────────────────────────────────────────
    def toggle_record(self):
        if self._rec and self._rec.isRunning():
            self._rec.stop()
            self._set_rec_idle()
            self.lbl_rec3.setText("Stopped early")
            return
        self.btn_record3.setObjectName("btn_record_active")
        self.btn_record3.setText("Stop Recording")
        self.btn_record3.setStyle(self.btn_record3.style())
        self.lbl_rec3.setText(f"Recording {self.dur_spin3.value()}s ...")
        self.vu_bar3.setValue(0)
        self._rec = RecordThread(self.dur_spin3.value(), self.sr)
        self._rec.finished.connect(self._on_recorded)
        self._rec.vu_update.connect(lambda v: self.vu_bar3.setValue(int(min(100, v * 600))))
        self._rec.start()

    def _set_rec_idle(self):
        self.btn_record3.setObjectName("btn_record")
        self.btn_record3.setText("Start Recording")
        self.btn_record3.setStyle(self.btn_record3.style())

    def _on_recorded(self, audio, sr):
        self.audio = audio
        self.sr    = int(sr)
        self._set_rec_idle()
        self.lbl_rec3.setText(f"Recorded  {len(audio) / sr:.2f}s")
        self.vu_bar3.setValue(0)
        self.btn_play3.setEnabled(True)
        self.btn_run.setEnabled(True)

    # ── Run analysis ──────────────────────────────────────────────────────────
    def run_analysis(self):
        if self.audio is None:
            return
        fs = self.frame_spin3.value()
        hs = self.hop_spin3.value()

        # guard: hop must be < frame
        if hs >= fs:
            hs = fs // 2
            self.hop_spin3.setValue(hs)

        self._frames = compute_zcr_detailed(self.audio, fs, hs, self.sr)
        n = len(self._frames)
        if n == 0:
            return

        # Update win spinner range
        self.win_spin.setRange(0, n - 1)
        self.win_spin.setValue(0)

        # Stat cards (global)
        self.c3_frames.set_text(str(n), C["accent"])
        self.c3_fs.set_text(str(fs), C["accent"])
        self.c3_hs.set_text(str(hs), C["accent"])

        self._fill_table()
        self._draw_zcr_bar()
        self._update_window_detail()   # draws top canvases for frame 0

    # ── Per-frame detail update (called on win_spin change or table click) ────
    def _update_window_detail(self):
        if not self._frames or self.audio is None:
            return
        idx = self.win_spin.value()
        if idx < 0 or idx >= len(self._frames):
            return
        row   = self._frames[idx]
        audio = self.audio
        sr    = self.sr
        fs    = row["frame_length"]
        start = row["start_sample"]
        end   = row["end_sample"]
        frame = audio[start:end]

        # Update stat cards for this window
        pol_col = C["green"] if row["polarity"] == "Positive" else \
                  C["red"]   if row["polarity"] == "Negative" else C["amber"]
        self.c3_zcr_w.set_value(row["zcr"],          "{:.5f}")
        self.c3_cross_w.set_value(float(row["n_crossings"]), "{:.0f}")
        self.c3_pol_w.set_text(row["polarity"], pol_col)

        # Highlight corresponding table row without triggering re-draw
        self.table.blockSignals(True)
        self.table.selectRow(idx)
        self.table.scrollTo(self.table.model().index(idx, 0))
        self.table.blockSignals(False)

        # ── Plot top-left: full signal with window highlight ─────────────────
        self.canvas_top.reset()
        ax0 = self.canvas_top.axes[0]
        t_full = np.arange(len(audio)) / sr
        ax0.plot(t_full, audio, color=C["waveform"], linewidth=0.55, alpha=0.7)

        # Shaded selected window
        ws = start / sr
        we = end   / sr
        ax0.axvspan(ws, we, color=C["accent"], alpha=0.18)
        ax0.axvline(ws, color=C["accent"], linewidth=1.0, alpha=0.8)
        ax0.axvline(we, color=C["accent"], linewidth=1.0, alpha=0.8)
        ax0.axhline(0,  color=C["border_hi"], linewidth=0.5)

        # Mark all ZC positions across full signal
        signs    = np.sign(audio)
        signs[signs == 0] = 1
        zc_idx   = np.where(np.diff(signs) != 0)[0]
        zc_times = zc_idx / sr
        ax0.scatter(zc_times, np.zeros(len(zc_times)),
                    s=8, color=C["red"], zorder=4, alpha=0.4, linewidths=0)

        style_ax(ax0, f"Full Signal  (frame {idx} highlighted)", "Time (s)", "Amplitude")

        # ── Plot top-right: zoomed frame with sample detail ──────────────────
        ax1  = self.canvas_top.axes[1]
        n_s  = len(frame)
        samp = np.arange(n_s)
        t_s  = (start + samp) / sr

        # Color samples by polarity
        pos_mask = frame >= 0
        neg_mask = frame <  0
        if pos_mask.any():
            ax1.vlines(t_s[pos_mask],  0, frame[pos_mask],
                       colors=C["green"], linewidth=1.6, alpha=0.85)
            ax1.scatter(t_s[pos_mask], frame[pos_mask],
                        s=14, color=C["green"], zorder=5, linewidths=0)
        if neg_mask.any():
            ax1.vlines(t_s[neg_mask],  0, frame[neg_mask],
                       colors=C["red"], linewidth=1.6, alpha=0.85)
            ax1.scatter(t_s[neg_mask], frame[neg_mask],
                        s=14, color=C["red"], zorder=5, linewidths=0)

        # Sign function overlay (secondary y-axis)
        ax1b = ax1.twinx()
        ax1b.set_facecolor("none")
        s_fn = np.sign(frame)
        s_fn[s_fn == 0] = 1
        ax1b.step(t_s, s_fn, color=C["amber"], linewidth=0.9,
                  alpha=0.75, where="post", label="sgn(x)")
        ax1b.set_ylim(-2.5, 2.5)
        ax1b.set_yticks([-1, 0, 1])
        ax1b.tick_params(colors=C["text_muted"], labelsize=7)
        ax1b.set_ylabel("sgn(x)", color=C["amber"], fontsize=8)
        ax1b.spines["right"].set_color(C["border"])
        ax1b.spines["right"].set_linewidth(0.7)

        # Mark zero crossings in frame with vertical lines
        f_signs = np.sign(frame)
        f_signs[f_signs == 0] = 1
        f_zc = np.where(np.diff(f_signs) != 0)[0]
        for zci in f_zc:
            xpos = (start + zci + 0.5) / sr
            ax1.axvline(xpos, color=C["cyan"], linewidth=1.0,
                        linestyle="--", alpha=0.85, zorder=3)

        ax1.axhline(0, color=C["border_hi"], linewidth=0.6)
        ax1.legend(
            handles=[
                mpatches.Patch(color=C["green"], label="Positive"),
                mpatches.Patch(color=C["red"],   label="Negative"),
                mpatches.Patch(color=C["cyan"],  label="ZC points"),
                mpatches.Patch(color=C["amber"], label="sgn(x)"),
            ],
            fontsize=7, facecolor=C["surface"], labelcolor=C["text"],
            framealpha=0.9, edgecolor=C["border"], loc="upper right",
            ncol=2
        )
        style_ax(ax1,
                 f"Frame {idx}  |  ZCR={row['zcr']:.5f}  |  N_cross={row['n_crossings']}  |  {row['polarity']}",
                 "Time (s)", "Amplitude")

        self.canvas_top.flush(pad=1.8, h_pad=2.0, w_pad=2.5)

    # ── ZCR bar chart ─────────────────────────────────────────────────────────
    def _draw_zcr_bar(self):
        if not self._frames:
            return
        self.canvas_bar.reset()
        ax  = self.canvas_bar.axes[0]
        idxs = [r["frame_idx"]  for r in self._frames]
        zcrs = [r["zcr"]        for r in self._frames]
        pols = [r["polarity"]   for r in self._frames]

        bar_colors = [
            C["green"] if p == "Positive" else C["red"] if p == "Negative" else C["amber"]
            for p in pols
        ]
        bars = ax.bar(idxs, zcrs, color=bar_colors, alpha=0.80, width=0.8, zorder=2)

        # Highlight selected
        sel = self.win_spin.value()
        if 0 <= sel < len(bars):
            bars[sel].set_edgecolor("white")
            bars[sel].set_linewidth(1.4)

        mean_zcr = float(np.mean(zcrs))
        ax.axhline(mean_zcr, color=C["accent"], linewidth=1.0,
                   linestyle="--", alpha=0.9, label=f"Mean  {mean_zcr:.4f}")
        ax.legend(fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                  framealpha=0.9, edgecolor=C["border"])

        legend_handles = [
            mpatches.Patch(color=C["green"], label="Positive"),
            mpatches.Patch(color=C["red"],   label="Negative"),
            mpatches.Patch(color=C["amber"], label="Mixed"),
        ]
        ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0][-1:],
                  labels=["Positive", "Negative", "Mixed", f"Mean {mean_zcr:.4f}"],
                  fontsize=8, facecolor=C["surface"], labelcolor=C["text"],
                  framealpha=0.9, edgecolor=C["border"], ncol=4)

        style_ax(ax, "ZCR per Frame — color by dominant polarity",
                 "Frame Index", "ZCR (crossings/sample)")
        self.canvas_bar.flush(pad=1.5, h_pad=1.5, w_pad=1.5)

    # ── Table fill ────────────────────────────────────────────────────────────
    def _fill_table(self):
        self.table.setRowCount(0)
        polarity_colors = {
            "Positive": QColor(C["green"]),
            "Negative": QColor(C["red"]),
            "Mixed":    QColor(C["amber"]),
        }
        for row in self._frames:
            r = self.table.rowCount()
            self.table.insertRow(r)
            cells = [
                str(row["frame_idx"]),
                f"{row['start_time_s']:.4f}",
                f"{row['end_time_s']:.4f}",
                str(row["frame_length"]),
                f"{row['zcr']:.6f}",
                str(row["n_crossings"]),
                row["polarity"],
                f"{row['mean_amp']:.5f}",
            ]
            for col, val in enumerate(cells):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 6:   # polarity column
                    item.setForeground(polarity_colors.get(val, QColor(C["text"])))
                    item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                self.table.setItem(r, col, item)

        self.table.resizeColumnsToContents()

    def _table_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.win_spin.setValue(rows[0].row())


# ============================================================
# MAIN WINDOW
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Signal Analyzer")
        self.setMinimumSize(1320, 840)
        self.setStyleSheet(APP_STYLE)
        self._build()

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────────
        topbar = QWidget()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(
            f"background-color:{C['surface']}; border-bottom:1px solid {C['border']};"
        )
        tl = QHBoxLayout(topbar)
        tl.setContentsMargins(20, 0, 20, 0)
        tl.setSpacing(0)

        accent_mark = QFrame()
        accent_mark.setFixedSize(4, 26)
        accent_mark.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {C['accent']},stop:1 {C['accent2']});"
            f"border-radius:2px;"
        )
        tl.addWidget(accent_mark)
        tl.addSpacing(12)

        title = QLabel("Speech Signal Analyzer")
        title.setStyleSheet(
            f"color:{C['text']}; font-size:15px; font-weight:700; letter-spacing:-0.3px;"
        )
        tl.addWidget(title)
        tl.addSpacing(10)

        badge = QLabel("v3.0")
        badge.setStyleSheet(
            f"color:{C['accent']}; font-size:10px; font-weight:700;"
            f"background:{C['surface2']}; border:1px solid {C['border_hi']};"
            f"border-radius:4px; padding:1px 6px;"
        )
        tl.addWidget(badge)
        tl.addStretch()

        meta = QLabel("NumPy  |  SciPy  |  SoundDevice  |  Matplotlib  |  PyQt5")
        meta.setStyleSheet(f"color:{C['text_muted']}; font-size:11px;")
        tl.addWidget(meta)

        root.addWidget(topbar)

        # ── Tabs ─────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setStyleSheet(f"background-color:{C['surface']};")
        self.tabs.addTab(FeatureTab(),      "Signal Feature Extraction")
        self.tabs.addTab(PhonemeTab(),      "Phoneme Analysis")
        self.tabs.addTab(ZCRInspectorTab(), "ZCR Window Inspector")
        root.addWidget(self.tabs)

        # ── Status bar ───────────────────────────────────────────────────────
        sb = self.statusBar()
        sb.setStyleSheet(
            f"background:{C['surface']}; color:{C['text_muted']};"
            f"border-top:1px solid {C['border']}; font-size:11px;"
        )
        sb.showMessage(
            "Ready    |    Default frame: 512    Hop: 256    SR: 22050 Hz    "
            "|    Part 3: ZCR Window Inspector with per-frame polarity table"
        )


# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(C["bg"]))
    pal.setColor(QPalette.WindowText,      QColor(C["text"]))
    pal.setColor(QPalette.Base,            QColor(C["surface"]))
    pal.setColor(QPalette.AlternateBase,   QColor(C["surface2"]))
    pal.setColor(QPalette.Text,            QColor(C["text"]))
    pal.setColor(QPalette.Button,          QColor(C["surface2"]))
    pal.setColor(QPalette.ButtonText,      QColor(C["text"]))
    pal.setColor(QPalette.Highlight,       QColor(C["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Disabled, QPalette.Text,       QColor(C["text_muted"]))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(C["text_muted"]))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
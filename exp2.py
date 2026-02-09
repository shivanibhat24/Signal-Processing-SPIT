"""
Voice Signal Processor Pro - Enhanced Version
A professional audio analysis and processing application
Features: Multi-format support, threading, optimized performance
"""

import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QLabel, QTabWidget, QSpinBox, QComboBox, 
    QGroupBox, QMessageBox, QProgressBar, QSlider, QFrame,
    QStatusBar, QCheckBox, QRadioButton, QButtonGroup, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont
import librosa
import librosa.display
from PIL import Image
import os
import warnings
import gc
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')


class WorkerSignals(QObject):
    """Signals for worker threads"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class AudioLoadWorker(QThread):
    """Worker thread for loading audio files"""
    finished = pyqtSignal(object, int)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        try:
            self.progress.emit(25)
            
            # Detect file type and load accordingly
            file_ext = Path(self.file_path).suffix.lower()
            
            self.progress.emit(50)
            
            # Try multiple loading methods for better compatibility
            audio_data = None
            sample_rate = None
            
            # Method 1: Try soundfile first (works for WAV, FLAC, OGG)
            try:
                audio_data, sample_rate = sf.read(self.file_path, always_2d=False)
                if len(audio_data.shape) > 1:
                    # Convert stereo to mono
                    audio_data = np.mean(audio_data, axis=1)
                print(f"Loaded with soundfile: {self.file_path}")
            except Exception as e1:
                print(f"soundfile failed: {e1}")
                
                # Method 2: Try librosa (works for most formats including MP3, M4A, video files)
                try:
                    audio_data, sample_rate = librosa.load(self.file_path, sr=None, mono=True)
                    print(f"Loaded with librosa: {self.file_path}")
                except Exception as e2:
                    print(f"librosa failed: {e2}")
                    raise Exception(f"Could not load audio file. Tried soundfile and librosa.\nErrors:\n{str(e1)}\n{str(e2)}")
            
            self.progress.emit(75)
            
            # Ensure audio_data is 2D for consistency with rest of code
            if audio_data is not None:
                audio_data = audio_data.reshape(-1, 1)
            
            self.progress.emit(100)
            self.finished.emit(audio_data, sample_rate)
            
        except Exception as e:
            self.error.emit(f"Failed to load audio: {str(e)}")


class AnalysisWorker(QThread):
    """Worker thread for heavy analysis tasks"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, audio_data, sample_rate, task_type, **kwargs):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.task_type = task_type
        self.kwargs = kwargs
        
    def run(self):
        try:
            if self.task_type == 'voice_analysis':
                result = self.analyze_voice()
            elif self.task_type == 'extrema':
                result = self.analyze_extrema()
            elif self.task_type == 'spectrogram':
                result = self.create_spectrogram()
            elif self.task_type == 'segmentation':
                result = self.segment_audio()
            else:
                result = None
                
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}")
    
    def analyze_voice(self):
        """Analyze voice segments"""
        audio_1d = self.audio_data.flatten()
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        self.progress.emit(20)
        
        # Calculate energy
        energy = np.array([
            np.sum(audio_1d[i:i+frame_length]**2)
            for i in range(0, len(audio_1d)-frame_length, hop_length)
        ])
        
        self.progress.emit(40)
        
        # Calculate zero crossing rate
        zcr = np.array([
            np.sum(np.abs(np.diff(np.sign(audio_1d[i:i+frame_length])))) / (2 * frame_length)
            for i in range(0, len(audio_1d)-frame_length, hop_length)
        ])
        
        self.progress.emit(60)
        
        # Normalize
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)
        
        self.progress.emit(80)
        
        # Classification
        silence_threshold = 0.03
        voiced_zcr_threshold = 0.3
        
        segments = np.zeros(len(energy))
        for i in range(len(energy)):
            if energy_norm[i] < silence_threshold:
                segments[i] = 0  # Silence
            elif zcr_norm[i] < voiced_zcr_threshold:
                segments[i] = 1  # Voiced
            else:
                segments[i] = 2  # Unvoiced
        
        self.progress.emit(100)
        
        return {
            'energy_norm': energy_norm,
            'zcr_norm': zcr_norm,
            'segments': segments,
            'hop_length': hop_length,
            'silence_threshold': silence_threshold
        }
    
    def analyze_extrema(self):
        """Analyze extrema"""
        audio_1d = self.audio_data.flatten()
        sensitivity = self.kwargs.get('sensitivity', 10)
        window_size = int(sensitivity * 0.001 * self.sample_rate)
        
        self.progress.emit(50)
        
        maxima_indices = signal.argrelextrema(audio_1d, np.greater, order=window_size)[0]
        minima_indices = signal.argrelextrema(audio_1d, np.less, order=window_size)[0]
        
        self.progress.emit(100)
        
        return {
            'maxima': maxima_indices,
            'minima': minima_indices
        }
    
    def create_spectrogram(self):
        """Create spectrogram"""
        audio_1d = self.audio_data.flatten()
        
        self.progress.emit(50)
        
        D = librosa.stft(audio_1d)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        self.progress.emit(100)
        
        return S_db
    
    def segment_audio(self):
        """Segment audio and separate into voiced, unvoiced, and silence"""
        audio_1d = self.audio_data.flatten()
        silence_thresh = self.kwargs.get('silence_thresh', 0.3)
        min_segment_samples = self.kwargs.get('min_segment_samples', 13230)
        
        self.progress.emit(10)
        
        # Calculate energy and ZCR
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        energy = np.array([
            np.sum(audio_1d[i:i+frame_length]**2)
            for i in range(0, len(audio_1d)-frame_length, hop_length)
        ])
        
        self.progress.emit(30)
        
        # Calculate zero crossing rate
        zcr = np.array([
            np.sum(np.abs(np.diff(np.sign(audio_1d[i:i+frame_length])))) / (2 * frame_length)
            for i in range(0, len(audio_1d)-frame_length, hop_length)
        ])
        
        self.progress.emit(50)
        
        # Normalize
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)
        
        # Classify each frame
        voiced_zcr_threshold = 0.3
        segments_class = np.zeros(len(energy))
        
        for i in range(len(energy)):
            if energy_norm[i] < silence_thresh:
                segments_class[i] = 0  # Silence
            elif zcr_norm[i] < voiced_zcr_threshold:
                segments_class[i] = 1  # Voiced
            else:
                segments_class[i] = 2  # Unvoiced
        
        self.progress.emit(70)
        
        # Extract continuous segments for each type
        def extract_segments(segment_type):
            segments = []
            start_idx = None
            
            for i in range(len(segments_class)):
                if segments_class[i] == segment_type and start_idx is None:
                    start_idx = i * hop_length
                elif segments_class[i] != segment_type and start_idx is not None:
                    end_idx = i * hop_length
                    if end_idx - start_idx >= min_segment_samples:
                        segments.append((start_idx, end_idx))
                    start_idx = None
            
            # Handle last segment
            if start_idx is not None:
                end_idx = len(audio_1d)
                if end_idx - start_idx >= min_segment_samples:
                    segments.append((start_idx, end_idx))
            
            return segments
        
        silence_segments = extract_segments(0)
        voiced_segments = extract_segments(1)
        unvoiced_segments = extract_segments(2)
        
        self.progress.emit(100)
        
        return {
            'silence': silence_segments,
            'voiced': voiced_segments,
            'unvoiced': unvoiced_segments,
            'segments_class': segments_class,
            'hop_length': hop_length
        }


class ThemeManager:
    """Professional theme management system"""
    
    THEMES = {
        'Professional Dark': {
            'background': '#1E1E1E',
            'surface': '#252526',
            'surface_elevated': '#2D2D30',
            'primary': '#0E639C',
            'primary_hover': '#1177BB',
            'secondary': '#3794FF',
            'accent': '#00BCF2',
            'success': '#89D185',
            'warning': '#DCA561',
            'error': '#F48771',
            'text': '#CCCCCC',
            'text_secondary': '#858585',
            'border': '#3E3E42',
            'chart_bg': '#1E1E1E',
            'chart_grid': '#3E3E42'
        },
        'Professional Light': {
            'background': '#FFFFFF',
            'surface': '#F3F3F3',
            'surface_elevated': '#FFFFFF',
            'primary': '#0078D4',
            'primary_hover': '#106EBE',
            'secondary': '#2B88D8',
            'accent': '#00BCF2',
            'success': '#107C10',
            'warning': '#CA5010',
            'error': '#D13438',
            'text': '#323130',
            'text_secondary': '#605E5C',
            'border': '#D1D1D1',
            'chart_bg': '#FFFFFF',
            'chart_grid': '#EDEBE9'
        },
        'Monochrome': {
            'background': '#FAFAFA',
            'surface': '#FFFFFF',
            'surface_elevated': '#FFFFFF',
            'primary': '#212121',
            'primary_hover': '#424242',
            'secondary': '#616161',
            'accent': '#757575',
            'success': '#424242',
            'warning': '#616161',
            'error': '#212121',
            'text': '#212121',
            'text_secondary': '#757575',
            'border': '#E0E0E0',
            'chart_bg': '#FFFFFF',
            'chart_grid': '#E0E0E0'
        },
        'Arctic': {
            'background': '#F0F4F8',
            'surface': '#FFFFFF',
            'surface_elevated': '#FFFFFF',
            'primary': '#0D3B66',
            'primary_hover': '#1A5490',
            'secondary': '#1E5A8E',
            'accent': '#2E8BC0',
            'success': '#52B788',
            'warning': '#E09F3E',
            'error': '#C1666B',
            'text': '#0D3B66',
            'text_secondary': '#627D98',
            'border': '#D9E2EC',
            'chart_bg': '#FFFFFF',
            'chart_grid': '#D9E2EC'
        },
        'Corporate': {
            'background': '#F5F5F5',
            'surface': '#FFFFFF',
            'surface_elevated': '#FFFFFF',
            'primary': '#003D5B',
            'primary_hover': '#005A87',
            'secondary': '#00607A',
            'accent': '#008891',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#E53935',
            'text': '#212121',
            'text_secondary': '#616161',
            'border': '#E0E0E0',
            'chart_bg': '#FFFFFF',
            'chart_grid': '#E0E0E0'
        }
    }
    
    @staticmethod
    def get_stylesheet(theme_name):
        """Generate complete stylesheet for theme"""
        theme = ThemeManager.THEMES.get(theme_name, ThemeManager.THEMES['Professional Dark'])
        
        return f"""
            QMainWindow, QWidget {{
                background-color: {theme['background']};
                color: {theme['text']};
                font-family: 'Segoe UI', 'San Francisco', 'Helvetica Neue', Arial, sans-serif;
            }}
            
            QGroupBox {{
                border: 1px solid {theme['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: {theme['surface']};
                font-weight: 500;
                font-size: 12px;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {theme['text']};
            }}
            
            QLabel {{
                color: {theme['text']};
                background-color: transparent;
            }}
            
            QPushButton {{
                background-color: {theme['primary']};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 6px 16px;
                font-size: 13px;
                font-weight: 500;
                min-height: 32px;
            }}
            
            QPushButton:hover {{
                background-color: {theme['primary_hover']};
            }}
            
            QPushButton:pressed {{
                background-color: {theme['secondary']};
            }}
            
            QPushButton:disabled {{
                background-color: {theme['border']};
                color: {theme['text_secondary']};
            }}
            
            QPushButton[buttonStyle="secondary"] {{
                background-color: {theme['surface_elevated']};
                color: {theme['text']};
                border: 1px solid {theme['border']};
            }}
            
            QPushButton[buttonStyle="secondary"]:hover {{
                background-color: {theme['border']};
            }}
            
            QPushButton[buttonStyle="danger"] {{
                background-color: {theme['error']};
            }}
            
            QPushButton[buttonStyle="success"] {{
                background-color: {theme['success']};
            }}
            
            QTabWidget::pane {{
                border: 1px solid {theme['border']};
                border-radius: 4px;
                background-color: {theme['surface']};
                top: -1px;
            }}
            
            QTabBar::tab {{
                background-color: {theme['surface']};
                color: {theme['text_secondary']};
                border: none;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {theme['surface_elevated']};
                color: {theme['text']};
                border-bottom: 2px solid {theme['primary']};
            }}
            
            QTabBar::tab:hover:!selected {{
                background-color: {theme['border']};
            }}
            
            QComboBox {{
                background-color: {theme['surface_elevated']};
                border: 1px solid {theme['border']};
                border-radius: 3px;
                padding: 6px;
                color: {theme['text']};
                min-height: 24px;
            }}
            
            QComboBox:hover {{
                border-color: {theme['primary']};
            }}
            
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {theme['surface_elevated']};
                border: 1px solid {theme['border']};
                selection-background-color: {theme['primary']};
                selection-color: white;
            }}
            
            QSpinBox {{
                background-color: {theme['surface_elevated']};
                border: 1px solid {theme['border']};
                border-radius: 3px;
                padding: 6px;
                color: {theme['text']};
                min-height: 24px;
            }}
            
            QSpinBox:hover {{
                border-color: {theme['primary']};
            }}
            
            QSlider::groove:horizontal {{
                height: 4px;
                background: {theme['border']};
                border-radius: 2px;
            }}
            
            QSlider::handle:horizontal {{
                background: {theme['primary']};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {theme['primary_hover']};
            }}
            
            QCheckBox {{
                spacing: 6px;
                color: {theme['text']};
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {theme['border']};
                border-radius: 3px;
                background-color: {theme['surface_elevated']};
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {theme['primary']};
                border-color: {theme['primary']};
            }}
            
            QCheckBox::indicator:hover {{
                border-color: {theme['primary']};
            }}
            
            QRadioButton {{
                spacing: 6px;
                color: {theme['text']};
            }}
            
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {theme['border']};
                border-radius: 9px;
                background-color: {theme['surface_elevated']};
            }}
            
            QRadioButton::indicator:checked {{
                background-color: {theme['primary']};
                border-color: {theme['primary']};
            }}
            
            QRadioButton::indicator:hover {{
                border-color: {theme['primary']};
            }}
            
            QStatusBar {{
                background-color: {theme['surface']};
                color: {theme['text_secondary']};
                border-top: 1px solid {theme['border']};
            }}
            
            QProgressBar {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                text-align: center;
                background-color: {theme['surface_elevated']};
                color: {theme['text']};
                font-weight: 500;
            }}
            
            QProgressBar::chunk {{
                background-color: {theme['primary']};
                border-radius: 2px;
            }}
            
            QTextEdit {{
                background-color: {theme['surface_elevated']};
                border: 1px solid {theme['border']};
                border-radius: 3px;
                color: {theme['text']};
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
            }}
            
            QMenuBar {{
                background-color: {theme['surface']};
                color: {theme['text']};
                border-bottom: 1px solid {theme['border']};
            }}
            
            QMenuBar::item:selected {{
                background-color: {theme['primary']};
                color: white;
            }}
            
            QMenu {{
                background-color: {theme['surface_elevated']};
                border: 1px solid {theme['border']};
                color: {theme['text']};
            }}
            
            QMenu::item:selected {{
                background-color: {theme['primary']};
                color: white;
            }}
        """


class PlotCanvas(FigureCanvas):
    """Professional matplotlib canvas"""
    
    def __init__(self, parent=None, width=8, height=5, dpi=100, theme='Professional Dark'):
        self.theme_name = theme
        self.theme = ThemeManager.THEMES[theme]
        
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor=self.theme['chart_bg'])
        self.axes = self.figure.add_subplot(111)
        
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.setup_theme()
        
    def setup_theme(self):
        """Apply professional theme to matplotlib figure"""
        self.figure.patch.set_facecolor(self.theme['chart_bg'])
        self.axes.set_facecolor(self.theme['chart_bg'])
        
        # Professional styling
        for spine in ['top', 'right', 'bottom', 'left']:
            self.axes.spines[spine].set_color(self.theme['border'])
            self.axes.spines[spine].set_linewidth(0.5)
        
        self.axes.tick_params(
            colors=self.theme['text_secondary'], 
            which='both',
            labelsize=9
        )
        
        self.axes.xaxis.label.set_color(self.theme['text'])
        self.axes.yaxis.label.set_color(self.theme['text'])
        self.axes.title.set_color(self.theme['text'])
        
        self.axes.grid(True, alpha=0.2, color=self.theme['chart_grid'], linewidth=0.5)
        
    def update_theme(self, theme_name):
        """Update canvas theme"""
        self.theme_name = theme_name
        self.theme = ThemeManager.THEMES[theme_name]
        self.setup_theme()
        self.draw()


class AudioProcessor(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.audio_data = None
        self.sample_rate = 44100
        self.recording = False
        self.recorded_frames = []
        self.duration = 10
        self.current_theme = 'Professional Dark'
        self.second_signal = None
        self.is_playing = False
        self.current_worker = None
        self.playback_stream = None
        
        self.init_ui()
        self.apply_theme(self.current_theme)
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle('Audio Signal Processor Pro')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_waveform_tab()
        self.create_operations_tab()
        self.create_extrema_tab()
        self.create_voice_analysis_tab()
        self.create_segmentation_tab()
        self.create_spectrogram_tab()
        self.create_converter_tab()
        
        # Status bar
        self.create_status_bar()
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        file_menu.addAction('Load Audio/Video', self.load_audio_file, 'Ctrl+O')
        file_menu.addAction('Save Audio', self.quick_save, 'Ctrl+S')
        file_menu.addSeparator()
        file_menu.addAction('Exit', self.close, 'Ctrl+Q')
        
        # View menu
        view_menu = menubar.addMenu('View')
        theme_menu = view_menu.addMenu('Themes')
        
        for theme_name in ThemeManager.THEMES.keys():
            theme_menu.addAction(theme_name, lambda t=theme_name: self.apply_theme(t))
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        tools_menu.addAction('Quick Analyze', self.quick_analyze, 'Ctrl+A')
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        help_menu.addAction('About', self.show_about)
        
    def create_header(self):
        """Create application header"""
        header = QFrame()
        header.setMaximumHeight(80)
        
        layout = QHBoxLayout()
        
        # Title
        title_layout = QVBoxLayout()
        title = QLabel('Audio Signal Processor Pro')
        title.setFont(QFont('Segoe UI', 18, QFont.Bold))
        title_layout.addWidget(title)
        
        subtitle = QLabel('Professional Audio Analysis & Processing')
        subtitle.setFont(QFont('Segoe UI', 10))
        title_layout.addWidget(subtitle)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Theme selector
        theme_label = QLabel('Theme:')
        layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(ThemeManager.THEMES.keys()))
        self.theme_combo.setCurrentText(self.current_theme)
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        layout.addWidget(self.theme_combo)
        
        header.setLayout(layout)
        return header
        
    def create_control_panel(self):
        """Create main control panel"""
        group = QGroupBox('Audio Controls')
        main_layout = QVBoxLayout()
        
        # Row 1: File operations
        row1 = QHBoxLayout()
        
        self.load_btn = QPushButton('Load Audio/Video')
        self.load_btn.clicked.connect(self.load_audio_file)
        row1.addWidget(self.load_btn)
        
        self.record_btn = QPushButton('Start Recording')
        self.record_btn.setProperty('buttonStyle', 'danger')
        self.record_btn.clicked.connect(self.toggle_recording)
        row1.addWidget(self.record_btn)
        
        row1.addWidget(QLabel('Duration:'))
        self.duration_spin = QSpinBox()
        self.duration_spin.setMinimum(1)
        self.duration_spin.setMaximum(60)
        self.duration_spin.setValue(10)
        self.duration_spin.setSuffix(' sec')
        self.duration_spin.valueChanged.connect(self.update_duration)
        row1.addWidget(self.duration_spin)
        
        row1.addStretch()
        main_layout.addLayout(row1)
        
        # Row 2: Playback controls
        row2 = QHBoxLayout()
        
        self.play_btn = QPushButton('Play')
        self.play_btn.setProperty('buttonStyle', 'success')
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        row2.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        row2.addWidget(self.stop_btn)
        
        row2.addWidget(QLabel('Volume:'))
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.update_volume)
        row2.addWidget(self.volume_slider)
        
        self.volume_label = QLabel('100%')
        self.volume_label.setMinimumWidth(50)
        row2.addWidget(self.volume_label)
        
        main_layout.addLayout(row2)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Info label
        self.info_label = QLabel('No audio loaded - Supports: WAV, FLAC, OGG, MP3, AAC, M4A, MP4, AVI, MOV')
        main_layout.addWidget(self.info_label)
        
        group.setLayout(main_layout)
        return group
        
    def create_status_bar(self):
        """Create status bar with indicators"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        self.status_label = QLabel('Ready')
        status_bar.addWidget(self.status_label, 1)
        
        self.sr_label = QLabel('SR: --')
        status_bar.addPermanentWidget(self.sr_label)
        
        self.duration_label = QLabel('Duration: --')
        status_bar.addPermanentWidget(self.duration_label)
        
        self.size_label = QLabel('Size: --')
        status_bar.addPermanentWidget(self.size_label)
        
    def create_waveform_tab(self):
        """Create waveform visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls = QHBoxLayout()
        
        zoom_in_btn = QPushButton('Zoom In')
        zoom_in_btn.clicked.connect(lambda: self.zoom_waveform(1.5))
        controls.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton('Zoom Out')
        zoom_out_btn.clicked.connect(lambda: self.zoom_waveform(0.67))
        controls.addWidget(zoom_out_btn)
        
        reset_btn = QPushButton('Reset View')
        reset_btn.setProperty('buttonStyle', 'secondary')
        reset_btn.clicked.connect(self.plot_waveform)
        controls.addWidget(reset_btn)
        
        controls.addStretch()
        
        self.show_envelope = QCheckBox('Show Envelope')
        self.show_envelope.stateChanged.connect(self.plot_waveform)
        controls.addWidget(self.show_envelope)
        
        layout.addLayout(controls)
        
        # Canvas
        self.waveform_canvas = PlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.waveform_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Waveform')
        
    def create_operations_tab(self):
        """Create signal operations tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox('Operation Settings')
        controls_layout = QVBoxLayout()
        
        # Operation selection
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel('Select Operation:'))
        
        self.op_button_group = QButtonGroup()
        self.op_add = QRadioButton('Addition')
        self.op_sub = QRadioButton('Subtraction')
        self.op_conv = QRadioButton('Convolution')
        self.op_add.setChecked(True)
        
        self.op_button_group.addButton(self.op_add)
        self.op_button_group.addButton(self.op_sub)
        self.op_button_group.addButton(self.op_conv)
        
        op_layout.addWidget(self.op_add)
        op_layout.addWidget(self.op_sub)
        op_layout.addWidget(self.op_conv)
        op_layout.addStretch()
        
        controls_layout.addLayout(op_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        load_second_btn = QPushButton('Load Second Signal')
        load_second_btn.clicked.connect(self.load_second_signal)
        btn_layout.addWidget(load_second_btn)
        
        apply_btn = QPushButton('Apply Operation')
        apply_btn.setProperty('buttonStyle', 'success')
        apply_btn.clicked.connect(self.apply_operation)
        btn_layout.addWidget(apply_btn)
        
        btn_layout.addStretch()
        controls_layout.addLayout(btn_layout)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Canvas
        self.operations_canvas = PlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.operations_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Operations')
        
    def create_extrema_tab(self):
        """Create extrema analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        controls = QHBoxLayout()
        
        analyze_btn = QPushButton('Analyze Extrema')
        analyze_btn.clicked.connect(self.analyze_extrema)
        controls.addWidget(analyze_btn)
        
        controls.addWidget(QLabel('Sensitivity:'))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(20)
        self.sensitivity_slider.setValue(10)
        controls.addWidget(self.sensitivity_slider)
        
        controls.addStretch()
        
        layout.addLayout(controls)
        
        self.extrema_canvas = PlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.extrema_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Extrema')
        
    def create_voice_analysis_tab(self):
        """Create voice analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        controls = QHBoxLayout()
        
        analyze_btn = QPushButton('Analyze Voice')
        analyze_btn.clicked.connect(self.analyze_voice_segments)
        controls.addWidget(analyze_btn)
        
        export_btn = QPushButton('Export Report')
        export_btn.setProperty('buttonStyle', 'secondary')
        export_btn.clicked.connect(self.export_voice_report)
        controls.addWidget(export_btn)
        
        controls.addStretch()
        
        layout.addLayout(controls)
        
        self.voice_canvas = PlotCanvas(self, width=12, height=8, theme=self.current_theme)
        layout.addWidget(self.voice_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Voice Analysis')
        
    def create_segmentation_tab(self):
        """Create voice segmentation tab with separate graphs"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox('Segmentation Settings')
        controls_layout = QVBoxLayout()
        
        # Parameters
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel('Silence Threshold:'))
        self.silence_threshold = QSlider(Qt.Horizontal)
        self.silence_threshold.setMinimum(1)
        self.silence_threshold.setMaximum(100)
        self.silence_threshold.setValue(30)
        params_layout.addWidget(self.silence_threshold)
        
        params_layout.addWidget(QLabel('Min Segment (ms):'))
        self.min_segment_spin = QSpinBox()
        self.min_segment_spin.setMinimum(100)
        self.min_segment_spin.setMaximum(5000)
        self.min_segment_spin.setValue(300)
        params_layout.addWidget(self.min_segment_spin)
        
        controls_layout.addLayout(params_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        segment_btn = QPushButton('Segment Audio')
        segment_btn.clicked.connect(self.segment_audio)
        btn_layout.addWidget(segment_btn)
        
        export_segments_btn = QPushButton('Export Segments')
        export_segments_btn.setProperty('buttonStyle', 'secondary')
        export_segments_btn.clicked.connect(self.export_segments)
        btn_layout.addWidget(export_segments_btn)
        
        btn_layout.addStretch()
        controls_layout.addLayout(btn_layout)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Canvas for separate graphs
        self.segmentation_canvas = PlotCanvas(self, width=12, height=10, theme=self.current_theme)
        layout.addWidget(self.segmentation_canvas)
        
        # Results
        results_group = QGroupBox('Segmentation Results')
        results_layout = QVBoxLayout()
        
        self.segment_results = QTextEdit()
        self.segment_results.setMaximumHeight(150)
        self.segment_results.setReadOnly(True)
        results_layout.addWidget(self.segment_results)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Segmentation')
        
    def create_spectrogram_tab(self):
        """Create spectrogram tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        controls_group = QGroupBox('Spectrogram Settings')
        controls_layout = QVBoxLayout()
        
        # Row 1
        row1 = QHBoxLayout()
        
        create_btn = QPushButton('Create Spectrogram')
        create_btn.clicked.connect(self.create_spectrogram)
        row1.addWidget(create_btn)
        
        histogram_btn = QPushButton('Generate Histogram')
        histogram_btn.setProperty('buttonStyle', 'secondary')
        histogram_btn.clicked.connect(self.spectrogram_to_histogram)
        row1.addWidget(histogram_btn)
        
        row1.addStretch()
        controls_layout.addLayout(row1)
        
        # Row 2
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Colormap:'))
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        row2.addWidget(self.colormap_combo)
        
        row2.addStretch()
        controls_layout.addLayout(row2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        self.spectrogram_canvas = PlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.spectrogram_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Spectrogram')
        
    def create_converter_tab(self):
        """Create format converter tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Info
        info_group = QGroupBox('Format Conversion')
        info_layout = QVBoxLayout()
        info_text = QLabel(
            'Supported Formats:\n'
            '• WAV - Uncompressed (16/24/32-bit)\n'
            '• FLAC - Lossless compression\n'
            '• OGG - Open-source compressed (Vorbis codec)\n\n'
            'Native Python implementation - No external dependencies required'
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Settings
        settings_group = QGroupBox('Conversion Settings')
        settings_layout = QVBoxLayout()
        
        # Format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel('Output Format:'))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['WAV', 'FLAC', 'OGG'])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        settings_layout.addLayout(format_layout)
        
        # Sample rate
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel('Sample Rate:'))
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(['Keep Original', '8000 Hz', '16000 Hz', '22050 Hz', 
                                '44100 Hz', '48000 Hz', '96000 Hz'])
        sr_layout.addWidget(self.sr_combo)
        sr_layout.addStretch()
        settings_layout.addLayout(sr_layout)
        
        # Bit depth
        bd_layout = QHBoxLayout()
        bd_layout.addWidget(QLabel('Bit Depth:'))
        self.bd_combo = QComboBox()
        self.bd_combo.addItems(['16-bit', '24-bit', '32-bit Float'])
        bd_layout.addWidget(self.bd_combo)
        bd_layout.addStretch()
        settings_layout.addLayout(bd_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Convert button
        convert_btn = QPushButton('Convert & Save')
        convert_btn.setProperty('buttonStyle', 'success')
        convert_btn.setMinimumHeight(40)
        convert_btn.clicked.connect(self.convert_audio_format)
        layout.addWidget(convert_btn)
        
        layout.addStretch()
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Converter')
        
    # Core functionality methods
    
    def apply_theme(self, theme_name):
        """Apply theme to application"""
        self.current_theme = theme_name
        self.setStyleSheet(ThemeManager.get_stylesheet(theme_name))
        
        # Update all canvases
        if hasattr(self, 'waveform_canvas'):
            self.waveform_canvas.update_theme(theme_name)
        if hasattr(self, 'operations_canvas'):
            self.operations_canvas.update_theme(theme_name)
        if hasattr(self, 'extrema_canvas'):
            self.extrema_canvas.update_theme(theme_name)
        if hasattr(self, 'voice_canvas'):
            self.voice_canvas.update_theme(theme_name)
        if hasattr(self, 'segmentation_canvas'):
            self.segmentation_canvas.update_theme(theme_name)
        if hasattr(self, 'spectrogram_canvas'):
            self.spectrogram_canvas.update_theme(theme_name)
            
        self.statusBar().showMessage(f'Theme: {theme_name}', 2000)
        
    def update_duration(self, value):
        """Update recording duration"""
        self.duration = value
        
    def update_volume(self, value):
        """Update volume display"""
        self.volume_label.setText(f'{value}%')
        
    def toggle_recording(self):
        """Toggle audio recording"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start audio recording"""
        self.recording = True
        self.recorded_frames = []
        self.record_btn.setText('Stop Recording')
        self.status_label.setText(f'Recording for {self.duration} seconds...')
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.recorded_frames.append(indata.copy())
        
        try:
            self.stream = sd.InputStream(
                callback=callback, 
                channels=1, 
                samplerate=self.sample_rate,
                blocksize=2048  # Optimized buffer size
            )
            self.stream.start()
            
            self.record_timer = QTimer()
            self.record_timer.timeout.connect(self.update_recording_progress)
            self.record_timer.start(100)
            
            self.recording_start_time = 0
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to start recording:\n{str(e)}')
            self.recording = False
            self.record_btn.setText('Start Recording')
            self.progress_bar.setVisible(False)
        
    def update_recording_progress(self):
        """Update recording progress bar"""
        self.recording_start_time += 0.1
        progress = int((self.recording_start_time / self.duration) * 100)
        self.progress_bar.setValue(min(progress, 100))
        
        if self.recording_start_time >= self.duration:
            self.stop_recording()
            
    def stop_recording(self):
        """Stop audio recording"""
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        self.recording = False
        self.record_btn.setText('Start Recording')
        
        if hasattr(self, 'record_timer'):
            self.record_timer.stop()
        
        self.progress_bar.setVisible(False)
        
        if self.recorded_frames:
            try:
                self.audio_data = np.concatenate(self.recorded_frames, axis=0)
                self.status_label.setText('Recording completed successfully')
                self.play_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.update_audio_info()
                
                # Plot waveform in a non-blocking way
                QTimer.singleShot(100, self.plot_waveform)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to process recording:\n{str(e)}')
        
    def load_audio_file(self):
        """Load audio or video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Audio/Video File', '', 
            'All Supported (*.wav *.flac *.ogg *.mp3 *.aac *.m4a *.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v);;'
            'Audio Files (*.wav *.flac *.ogg *.mp3 *.aac *.m4a);;'
            'Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v);;'
            'All Files (*)'
        )
        
        if file_path:
            print(f"Loading file: {file_path}")
            self.status_label.setText('Loading file...')
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Disable buttons while loading
            self.load_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            
            # Create and start worker thread
            self.current_worker = AudioLoadWorker(file_path)
            self.current_worker.finished.connect(self.on_audio_loaded)
            self.current_worker.error.connect(self.on_load_error)
            self.current_worker.progress.connect(self.progress_bar.setValue)
            self.current_worker.start()
    
    def on_audio_loaded(self, audio_data, sample_rate):
        """Handle audio load completion"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
        self.status_label.setText('Audio loaded successfully')
        self.progress_bar.setVisible(False)
        
        # Re-enable buttons
        self.load_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.update_audio_info()
        
        # Plot waveform in a non-blocking way
        QTimer.singleShot(100, self.plot_waveform)
        
        # Clean up worker
        if self.current_worker:
            self.current_worker = None
        
        # Trigger garbage collection
        gc.collect()
    
    def on_load_error(self, error_msg):
        """Handle audio load error"""
        QMessageBox.critical(self, 'Error', error_msg)
        self.status_label.setText('Failed to load audio')
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        
        if self.current_worker:
            self.current_worker = None
                
    def update_audio_info(self):
        """Update audio information display"""
        if self.audio_data is not None:
            duration = len(self.audio_data) / self.sample_rate
            size = self.audio_data.nbytes / 1024
            
            self.info_label.setText(f'Audio loaded ({len(self.audio_data)} samples)')
            self.sr_label.setText(f'SR: {self.sample_rate} Hz')
            self.duration_label.setText(f'Duration: {duration:.2f}s')
            self.size_label.setText(f'Size: {size:.1f} KB')
            
    def play_audio(self):
        """Play audio with optimized playback"""
        if self.audio_data is None:
            return
            
        # Stop any existing playback
        self.stop_audio()
        
        try:
            volume = self.volume_slider.value() / 100
            audio_to_play = self.audio_data * volume
            
            # Use non-blocking playback
            self.playback_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=2048
            )
            self.playback_stream.start()
            
            # Write audio in chunks
            chunk_size = 2048
            for i in range(0, len(audio_to_play), chunk_size):
                chunk = audio_to_play[i:i+chunk_size]
                self.playback_stream.write(chunk)
            
            self.playback_stream.stop()
            self.playback_stream.close()
            self.playback_stream = None
            
            self.status_label.setText('Playback complete')
            self.is_playing = False
            
        except Exception as e:
            QMessageBox.warning(self, 'Playback Error', f'Error during playback:\n{str(e)}')
            self.is_playing = False
            if self.playback_stream:
                try:
                    self.playback_stream.stop()
                    self.playback_stream.close()
                except:
                    pass
                self.playback_stream = None
            
    def stop_audio(self):
        """Stop audio playback"""
        try:
            sd.stop()
            if self.playback_stream:
                self.playback_stream.stop()
                self.playback_stream.close()
                self.playback_stream = None
        except:
            pass
        
        self.status_label.setText('Playback stopped')
        self.is_playing = False
        
    def plot_waveform(self):
        """Plot audio waveform"""
        if self.audio_data is None:
            return
            
        try:
            self.waveform_canvas.axes.clear()
            
            audio_1d = self.audio_data.flatten()
            
            # Downsample for display if too large
            max_points = 50000
            if len(audio_1d) > max_points:
                step = len(audio_1d) // max_points
                audio_1d = audio_1d[::step]
            else:
                step = 1
            
            time_axis = np.arange(len(audio_1d)) * step / self.sample_rate
            
            theme = ThemeManager.THEMES[self.current_theme]
            self.waveform_canvas.axes.plot(
                time_axis, audio_1d, 
                linewidth=0.5, 
                color=theme['primary'], 
                alpha=0.8
            )
            
            if self.show_envelope.isChecked():
                envelope = np.abs(signal.hilbert(audio_1d))
                self.waveform_canvas.axes.plot(
                    time_axis, envelope, 
                    linewidth=1.0, 
                    color=theme['accent'], 
                    alpha=0.6, 
                    label='Envelope'
                )
                self.waveform_canvas.axes.plot(
                    time_axis, -envelope, 
                    linewidth=1.0, 
                    color=theme['accent'], 
                    alpha=0.6
                )
                self.waveform_canvas.axes.legend(fontsize=9)
            
            self.waveform_canvas.axes.set_xlabel('Time (s)', fontsize=10)
            self.waveform_canvas.axes.set_ylabel('Amplitude', fontsize=10)
            self.waveform_canvas.axes.set_title(
                f'Waveform - {self.sample_rate} Hz', 
                fontsize=11, 
                fontweight='500'
            )
            self.waveform_canvas.setup_theme()
            self.waveform_canvas.draw()
        except Exception as e:
            print(f"Error plotting waveform: {e}")
        
    def zoom_waveform(self, factor):
        """Zoom waveform plot"""
        if self.audio_data is None:
            return
            
        xlim = self.waveform_canvas.axes.get_xlim()
        ylim = self.waveform_canvas.axes.get_ylim()
        
        xcenter = (xlim[0] + xlim[1]) / 2
        ycenter = (ylim[0] + ylim[1]) / 2
        xrange = (xlim[1] - xlim[0]) / factor
        yrange = (ylim[1] - ylim[0]) / factor
        
        self.waveform_canvas.axes.set_xlim(xcenter - xrange/2, xcenter + xrange/2)
        self.waveform_canvas.axes.set_ylim(ycenter - yrange/2, ycenter + yrange/2)
        self.waveform_canvas.draw()
        
    def load_second_signal(self):
        """Load second signal for operations"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Second Signal', '', 
            'Audio Files (*.wav *.flac *.ogg *.mp3 *.aac *.m4a);;All Files (*)'
        )
        
        if file_path:
            try:
                self.second_signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                self.second_signal = self.second_signal.reshape(-1, 1)
                self.status_label.setText('Second signal loaded')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load:\n{str(e)}')
                
    def apply_operation(self):
        """Apply signal operation"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first')
            return
            
        self.operations_canvas.axes.clear()
        theme = ThemeManager.THEMES[self.current_theme]
        
        if self.op_add.isChecked() or self.op_sub.isChecked():
            if self.second_signal is None:
                QMessageBox.warning(self, 'Warning', 'Load second signal')
                return
                
            min_len = min(len(self.audio_data), len(self.second_signal))
            sig1 = self.audio_data[:min_len]
            sig2 = self.second_signal[:min_len]
            
            if self.op_add.isChecked():
                result = sig1 + sig2
                title = 'Addition Result'
            else:
                result = sig1 - sig2
                title = 'Subtraction Result'
                
            time_axis = np.arange(len(result)) / self.sample_rate
            self.operations_canvas.axes.plot(
                time_axis, result, 
                linewidth=0.5,
                color=theme['primary']
            )
            
        elif self.op_conv.isChecked():
            kernel = np.ones(100) / 100
            result = np.convolve(self.audio_data.flatten(), kernel, mode='same')
            
            time_axis = np.arange(len(result)) / self.sample_rate
            self.operations_canvas.axes.plot(
                time_axis, result, 
                linewidth=0.5,
                color=theme['accent']
            )
            title = 'Convolution Result'
            
        self.operations_canvas.axes.set_xlabel('Time (s)', fontsize=10)
        self.operations_canvas.axes.set_ylabel('Amplitude', fontsize=10)
        self.operations_canvas.axes.set_title(title, fontsize=11, fontweight='500')
        self.operations_canvas.setup_theme()
        self.operations_canvas.draw()
        
        self.status_label.setText('Operation applied')
        
    def analyze_extrema(self):
        """Analyze signal extrema"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first')
            return
        
        self.progress_bar.setVisible(True)
        self.status_label.setText('Analyzing extrema...')
        
        # Create worker
        sensitivity = self.sensitivity_slider.value()
        worker = AnalysisWorker(
            self.audio_data, 
            self.sample_rate, 
            'extrema',
            sensitivity=sensitivity
        )
        worker.finished.connect(self.on_extrema_complete)
        worker.error.connect(lambda msg: QMessageBox.critical(self, 'Error', msg))
        worker.progress.connect(self.progress_bar.setValue)
        worker.start()
        
        self.current_worker = worker
    
    def on_extrema_complete(self, result):
        """Handle extrema analysis completion"""
        self.progress_bar.setVisible(False)
        
        if result is None:
            return
        
        self.extrema_canvas.axes.clear()
        
        audio_1d = self.audio_data.flatten()
        time_axis = np.arange(len(audio_1d)) / self.sample_rate
        theme = ThemeManager.THEMES[self.current_theme]
        
        # Downsample for display
        max_points = 50000
        if len(audio_1d) > max_points:
            step = len(audio_1d) // max_points
            display_audio = audio_1d[::step]
            display_time = time_axis[::step]
        else:
            display_audio = audio_1d
            display_time = time_axis
        
        self.extrema_canvas.axes.plot(
            display_time, display_audio, 
            linewidth=0.4, 
            color=theme['text_secondary'], 
            alpha=0.5, 
            label='Signal'
        )
        
        maxima_indices = result['maxima']
        minima_indices = result['minima']
        
        if len(maxima_indices) > 0:
            self.extrema_canvas.axes.scatter(
                time_axis[maxima_indices], 
                audio_1d[maxima_indices],
                c=theme['success'], 
                marker='^', 
                s=30,
                label=f'Maxima ({len(maxima_indices)})', 
                zorder=5
            )
        
        if len(minima_indices) > 0:
            self.extrema_canvas.axes.scatter(
                time_axis[minima_indices], 
                audio_1d[minima_indices],
                c=theme['error'], 
                marker='v', 
                s=30,
                label=f'Minima ({len(minima_indices)})',
                zorder=5
            )
        
        self.extrema_canvas.axes.set_xlabel('Time (s)', fontsize=10)
        self.extrema_canvas.axes.set_ylabel('Amplitude', fontsize=10)
        self.extrema_canvas.axes.set_title('Extrema Analysis', fontsize=11, fontweight='500')
        self.extrema_canvas.axes.legend(fontsize=9)
        self.extrema_canvas.setup_theme()
        self.extrema_canvas.draw()
        
        self.status_label.setText(
            f'Found {len(maxima_indices)} maxima, {len(minima_indices)} minima'
        )
        
        self.current_worker = None
        gc.collect()
        
    def analyze_voice_segments(self):
        """Analyze voice segments"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first')
            return
        
        self.progress_bar.setVisible(True)
        self.status_label.setText('Analyzing voice...')
        
        # Create worker
        worker = AnalysisWorker(
            self.audio_data, 
            self.sample_rate, 
            'voice_analysis'
        )
        worker.finished.connect(self.on_voice_complete)
        worker.error.connect(lambda msg: QMessageBox.critical(self, 'Error', msg))
        worker.progress.connect(self.progress_bar.setValue)
        worker.start()
        
        self.current_worker = worker
    
    def on_voice_complete(self, result):
        """Handle voice analysis completion"""
        self.progress_bar.setVisible(False)
        
        if result is None:
            return
        
        self.voice_canvas.figure.clear()
        
        audio_1d = self.audio_data.flatten()
        
        energy_norm = result['energy_norm']
        zcr_norm = result['zcr_norm']
        segments = result['segments']
        hop_length = result['hop_length']
        silence_threshold = result['silence_threshold']
        
        time_frames = np.arange(len(energy_norm)) * hop_length / self.sample_rate
        time_signal = np.arange(len(audio_1d)) / self.sample_rate
        
        # Downsample signal for display
        max_points = 50000
        if len(audio_1d) > max_points:
            step = len(audio_1d) // max_points
            display_audio = audio_1d[::step]
            display_time = time_signal[::step]
        else:
            display_audio = audio_1d
            display_time = time_signal
        
        theme = ThemeManager.THEMES[self.current_theme]
        
        # Plot 1: Signal
        ax1 = self.voice_canvas.figure.add_subplot(311)
        ax1.plot(display_time, display_audio, linewidth=0.5, color=theme['primary'], alpha=0.7)
        ax1.set_ylabel('Amplitude', fontsize=9)
        ax1.set_title('Original Signal', fontsize=10, fontweight='500')
        ax1.grid(True, alpha=0.2, color=theme['chart_grid'], linewidth=0.5)
        ax1.set_facecolor(theme['chart_bg'])
        ax1.tick_params(labelsize=8, colors=theme['text_secondary'])
        
        # Plot 2: Features
        ax2 = self.voice_canvas.figure.add_subplot(312)
        ax2.plot(time_frames, energy_norm, label='Energy', linewidth=1.5, color=theme['success'])
        ax2.plot(time_frames, zcr_norm, label='ZCR', linewidth=1.5, color=theme['warning'])
        ax2.axhline(
            y=silence_threshold, 
            color=theme['error'], 
            linestyle='--', 
            label='Silence threshold', 
            linewidth=1
        )
        ax2.set_ylabel('Normalized', fontsize=9)
        ax2.set_title('Features', fontsize=10, fontweight='500')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2, color=theme['chart_grid'], linewidth=0.5)
        ax2.set_facecolor(theme['chart_bg'])
        ax2.tick_params(labelsize=8, colors=theme['text_secondary'])
        
        # Plot 3: Classification
        ax3 = self.voice_canvas.figure.add_subplot(313)
        colors_map = {
            0: theme['text_secondary'], 
            1: theme['primary'], 
            2: theme['warning']
        }
        labels_map = {0: 'Silence', 1: 'Voiced', 2: 'Unvoiced'}
        
        for i in range(3):
            mask = segments == i
            if np.any(mask):
                ax3.scatter(
                    time_frames[mask], 
                    segments[mask],
                    c=colors_map[i], 
                    label=labels_map[i], 
                    s=15, 
                    alpha=0.7
                )
        
        ax3.set_xlabel('Time (s)', fontsize=9)
        ax3.set_ylabel('Type', fontsize=9)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Silence', 'Voiced', 'Unvoiced'])
        ax3.set_title('Voice Activity Detection', fontsize=10, fontweight='500')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2, color=theme['chart_grid'], linewidth=0.5)
        ax3.set_facecolor(theme['chart_bg'])
        ax3.tick_params(labelsize=8, colors=theme['text_secondary'])
        
        self.voice_canvas.figure.tight_layout()
        self.voice_canvas.draw()
        
        # Calculate statistics
        silence_pct = np.sum(segments == 0) / len(segments) * 100
        voiced_pct = np.sum(segments == 1) / len(segments) * 100
        unvoiced_pct = np.sum(segments == 2) / len(segments) * 100
        
        self.status_label.setText(
            f'Silence: {silence_pct:.1f}% | Voiced: {voiced_pct:.1f}% | Unvoiced: {unvoiced_pct:.1f}%'
        )
        
        self.voice_stats = {
            'silence': silence_pct,
            'voiced': voiced_pct,
            'unvoiced': unvoiced_pct
        }
        
        self.current_worker = None
        gc.collect()
        
    def export_voice_report(self):
        """Export voice analysis report"""
        if not hasattr(self, 'voice_stats'):
            QMessageBox.warning(self, 'Warning', 'Run voice analysis first')
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Report', '', 'Text Files (*.txt);;All Files (*)'
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write('Voice Analysis Report\n')
                f.write('=' * 50 + '\n\n')
                f.write(f'Silence: {self.voice_stats["silence"]:.2f}%\n')
                f.write(f'Voiced: {self.voice_stats["voiced"]:.2f}%\n')
                f.write(f'Unvoiced: {self.voice_stats["unvoiced"]:.2f}%\n\n')
                f.write(f'Sample Rate: {self.sample_rate} Hz\n')
                f.write(f'Duration: {len(self.audio_data)/self.sample_rate:.2f}s\n')
                
            self.status_label.setText('Report saved')
            
    def segment_audio(self):
        """Segment audio based on silence detection"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first')
            return
        
        self.progress_bar.setVisible(True)
        self.status_label.setText('Segmenting audio...')
        
        silence_thresh = self.silence_threshold.value() / 100.0
        min_segment_ms = self.min_segment_spin.value()
        min_segment_samples = int(min_segment_ms * self.sample_rate / 1000)
        
        # Create worker
        worker = AnalysisWorker(
            self.audio_data, 
            self.sample_rate, 
            'segmentation',
            silence_thresh=silence_thresh,
            min_segment_samples=min_segment_samples
        )
        worker.finished.connect(self.on_segmentation_complete)
        worker.error.connect(lambda msg: QMessageBox.critical(self, 'Error', msg))
        worker.progress.connect(self.progress_bar.setValue)
        worker.start()
        
        self.current_worker = worker
    
    def on_segmentation_complete(self, result):
        """Handle segmentation completion with separate graphs"""
        self.progress_bar.setVisible(False)
        
        if result is None:
            return
        
        silence_segments = result['silence']
        voiced_segments = result['voiced']
        unvoiced_segments = result['unvoiced']
        
        # Store for export
        self.current_segments = {
            'silence': silence_segments,
            'voiced': voiced_segments,
            'unvoiced': unvoiced_segments
        }
        
        # Visualize with 3 separate graphs
        self.segmentation_canvas.figure.clear()
        theme = ThemeManager.THEMES[self.current_theme]
        
        audio_1d = self.audio_data.flatten()
        time_axis = np.arange(len(audio_1d)) / self.sample_rate
        
        # Downsample for display
        max_points = 50000
        if len(audio_1d) > max_points:
            step = len(audio_1d) // max_points
            display_audio = audio_1d[::step]
            display_time = time_axis[::step]
        else:
            display_audio = audio_1d
            display_time = time_axis
            step = 1
        
        # Graph 1: Voiced segments
        ax1 = self.segmentation_canvas.figure.add_subplot(311)
        ax1.plot(
            display_time, display_audio, 
            linewidth=0.3, 
            color=theme['text_secondary'], 
            alpha=0.3
        )
        
        for start, end in voiced_segments:
            ax1.axvspan(
                start / self.sample_rate,
                end / self.sample_rate,
                alpha=0.4,
                color=theme['primary'],
                label='Voiced' if start == voiced_segments[0][0] else ''
            )
        
        ax1.set_ylabel('Amplitude', fontsize=9)
        ax1.set_title(f'Voiced Segments ({len(voiced_segments)} segments)', fontsize=10, fontweight='500')
        ax1.grid(True, alpha=0.2, color=theme['chart_grid'], linewidth=0.5)
        ax1.set_facecolor(theme['chart_bg'])
        ax1.tick_params(labelsize=8, colors=theme['text_secondary'])
        if voiced_segments:
            ax1.legend(fontsize=8)
        
        # Graph 2: Unvoiced segments
        ax2 = self.segmentation_canvas.figure.add_subplot(312)
        ax2.plot(
            display_time, display_audio, 
            linewidth=0.3, 
            color=theme['text_secondary'], 
            alpha=0.3
        )
        
        for start, end in unvoiced_segments:
            ax2.axvspan(
                start / self.sample_rate,
                end / self.sample_rate,
                alpha=0.4,
                color=theme['warning'],
                label='Unvoiced' if start == unvoiced_segments[0][0] else ''
            )
        
        ax2.set_ylabel('Amplitude', fontsize=9)
        ax2.set_title(f'Unvoiced Segments ({len(unvoiced_segments)} segments)', fontsize=10, fontweight='500')
        ax2.grid(True, alpha=0.2, color=theme['chart_grid'], linewidth=0.5)
        ax2.set_facecolor(theme['chart_bg'])
        ax2.tick_params(labelsize=8, colors=theme['text_secondary'])
        if unvoiced_segments:
            ax2.legend(fontsize=8)
        
        # Graph 3: Silence segments
        ax3 = self.segmentation_canvas.figure.add_subplot(313)
        ax3.plot(
            display_time, display_audio, 
            linewidth=0.3, 
            color=theme['text_secondary'], 
            alpha=0.3
        )
        
        for start, end in silence_segments:
            ax3.axvspan(
                start / self.sample_rate,
                end / self.sample_rate,
                alpha=0.4,
                color=theme['error'],
                label='Silence' if start == silence_segments[0][0] else ''
            )
        
        ax3.set_xlabel('Time (s)', fontsize=9)
        ax3.set_ylabel('Amplitude', fontsize=9)
        ax3.set_title(f'Silence Segments ({len(silence_segments)} segments)', fontsize=10, fontweight='500')
        ax3.grid(True, alpha=0.2, color=theme['chart_grid'], linewidth=0.5)
        ax3.set_facecolor(theme['chart_bg'])
        ax3.tick_params(labelsize=8, colors=theme['text_secondary'])
        if silence_segments:
            ax3.legend(fontsize=8)
        
        self.segmentation_canvas.figure.tight_layout()
        self.segmentation_canvas.draw()
        
        # Display results
        total_segments = len(voiced_segments) + len(unvoiced_segments) + len(silence_segments)
        results_text = f'Total Segments: {total_segments}\n\n'
        
        results_text += f'VOICED SEGMENTS ({len(voiced_segments)}):\n'
        for i, (start, end) in enumerate(voiced_segments[:10], 1):
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            duration = end_time - start_time
            results_text += f'  {i}. {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)\n'
        if len(voiced_segments) > 10:
            results_text += f'  ... and {len(voiced_segments) - 10} more\n'
        
        results_text += f'\nUNVOICED SEGMENTS ({len(unvoiced_segments)}):\n'
        for i, (start, end) in enumerate(unvoiced_segments[:10], 1):
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            duration = end_time - start_time
            results_text += f'  {i}. {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)\n'
        if len(unvoiced_segments) > 10:
            results_text += f'  ... and {len(unvoiced_segments) - 10} more\n'
        
        results_text += f'\nSILENCE SEGMENTS ({len(silence_segments)}):\n'
        for i, (start, end) in enumerate(silence_segments[:10], 1):
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            duration = end_time - start_time
            results_text += f'  {i}. {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)\n'
        if len(silence_segments) > 10:
            results_text += f'  ... and {len(silence_segments) - 10} more\n'
        
        self.segment_results.setText(results_text)
        self.status_label.setText(
            f'Segmentation complete: {len(voiced_segments)} voiced, {len(unvoiced_segments)} unvoiced, {len(silence_segments)} silence'
        )
        
        self.current_worker = None
        gc.collect()
        
    def export_segments(self):
        """Export audio segments as separate files"""
        if not hasattr(self, 'current_segments'):
            QMessageBox.warning(self, 'Warning', 'Perform segmentation first')
            return
            
        segments_dict = self.current_segments
        total_segments = (len(segments_dict['voiced']) + 
                         len(segments_dict['unvoiced']) + 
                         len(segments_dict['silence']))
        
        if total_segments == 0:
            QMessageBox.warning(self, 'Warning', 'No segments found')
            return
            
        output_dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        
        if output_dir:
            audio_1d = self.audio_data.flatten()
            
            try:
                # Export voiced segments
                for i, (start, end) in enumerate(segments_dict['voiced'], 1):
                    segment_audio = audio_1d[start:end]
                    output_path = os.path.join(output_dir, f'voiced_{i:03d}.wav')
                    sf.write(output_path, segment_audio, self.sample_rate)
                
                # Export unvoiced segments
                for i, (start, end) in enumerate(segments_dict['unvoiced'], 1):
                    segment_audio = audio_1d[start:end]
                    output_path = os.path.join(output_dir, f'unvoiced_{i:03d}.wav')
                    sf.write(output_path, segment_audio, self.sample_rate)
                
                # Export silence segments
                for i, (start, end) in enumerate(segments_dict['silence'], 1):
                    segment_audio = audio_1d[start:end]
                    output_path = os.path.join(output_dir, f'silence_{i:03d}.wav')
                    sf.write(output_path, segment_audio, self.sample_rate)
                
                QMessageBox.information(
                    self, 
                    'Success', 
                    f'Exported {total_segments} segments to:\n{output_dir}\n\n'
                    f'Voiced: {len(segments_dict["voiced"])}\n'
                    f'Unvoiced: {len(segments_dict["unvoiced"])}\n'
                    f'Silence: {len(segments_dict["silence"])}'
                )
                self.status_label.setText(f'Exported {total_segments} segments')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Export failed:\n{str(e)}')
        
    def create_spectrogram(self):
        """Create spectrogram"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first')
            return
        
        self.progress_bar.setVisible(True)
        self.status_label.setText('Creating spectrogram...')
        
        # Create worker
        worker = AnalysisWorker(
            self.audio_data, 
            self.sample_rate, 
            'spectrogram'
        )
        worker.finished.connect(self.on_spectrogram_complete)
        worker.error.connect(lambda msg: QMessageBox.critical(self, 'Error', msg))
        worker.progress.connect(self.progress_bar.setValue)
        worker.start()
        
        self.current_worker = worker
    
    def on_spectrogram_complete(self, S_db):
        """Handle spectrogram completion"""
        self.progress_bar.setVisible(False)
        
        if S_db is None:
            return
        
        self.current_spectrogram = S_db
        
        self.spectrogram_canvas.axes.clear()
        
        colormap = self.colormap_combo.currentText()
        img = librosa.display.specshow(
            S_db, 
            sr=self.sample_rate, 
            x_axis='time', 
            y_axis='hz', 
            ax=self.spectrogram_canvas.axes,
            cmap=colormap
        )
        
        self.spectrogram_canvas.axes.set_title('Spectrogram', fontsize=11, fontweight='500')
        self.spectrogram_canvas.figure.colorbar(
            img, 
            ax=self.spectrogram_canvas.axes, 
            format='%+2.0f dB'
        )
        
        self.spectrogram_canvas.setup_theme()
        self.spectrogram_canvas.draw()
        self.status_label.setText('Spectrogram created')
        
        self.current_worker = None
        gc.collect()
        
    def spectrogram_to_histogram(self):
        """Convert spectrogram to histogram"""
        if not hasattr(self, 'current_spectrogram'):
            QMessageBox.warning(self, 'Warning', 'Create spectrogram first')
            return
            
        spec_norm = self.current_spectrogram - np.min(self.current_spectrogram)
        spec_norm = (spec_norm / np.max(spec_norm) * 255).astype(np.uint8)
        
        img = Image.fromarray(spec_norm)
        img_rgb = img.convert('RGB')
        histogram = np.array(img).flatten()
        
        self.spectrogram_canvas.figure.clear()
        theme = ThemeManager.THEMES[self.current_theme]
        
        ax1 = self.spectrogram_canvas.figure.add_subplot(121)
        ax1.imshow(img_rgb, aspect='auto', origin='lower')
        ax1.set_title('Spectrogram Image', fontsize=10, fontweight='500')
        ax1.set_xlabel('Time', fontsize=9)
        ax1.set_ylabel('Frequency', fontsize=9)
        
        ax2 = self.spectrogram_canvas.figure.add_subplot(122)
        ax2.hist(
            histogram, 
            bins=50, 
            color=theme['primary'], 
            edgecolor=theme['border'], 
            alpha=0.7
        )
        ax2.set_title('Intensity Histogram', fontsize=10, fontweight='500')
        ax2.set_xlabel('Intensity', fontsize=9)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.grid(True, alpha=0.2, linewidth=0.5)
        
        self.spectrogram_canvas.figure.tight_layout()
        self.spectrogram_canvas.draw()
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 'PNG (*.png);;JPEG (*.jpg)'
        )
        
        if save_path:
            img_rgb.save(save_path)
            self.status_label.setText('Image saved')
    
    def convert_audio_format(self):
        """Convert audio to different format"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first')
            return
            
        output_format = self.format_combo.currentText().lower()
        sr_text = self.sr_combo.currentText()
        
        if sr_text == 'Keep Original':
            target_sr = self.sample_rate
        else:
            target_sr = int(sr_text.split()[0])
        
        audio_to_save = self.audio_data.flatten()
        
        # Resample if needed
        if target_sr != self.sample_rate:
            self.status_label.setText('Resampling audio...')
            QApplication.processEvents()
            audio_to_save = librosa.resample(
                audio_to_save, 
                orig_sr=self.sample_rate, 
                target_sr=target_sr
            )
        
        file_filter = f'{output_format.upper()} (*.{output_format})'
        save_path, _ = QFileDialog.getSaveFileName(
            self, f'Save as {output_format.upper()}', '', file_filter
        )
        
        if not save_path:
            return
            
        if not save_path.endswith(f'.{output_format}'):
            save_path += f'.{output_format}'
        
        try:
            self.status_label.setText('Saving file...')
            QApplication.processEvents()
            
            if output_format == 'wav':
                bd_text = self.bd_combo.currentText()
                if '16-bit' in bd_text:
                    subtype = 'PCM_16'
                elif '24-bit' in bd_text:
                    subtype = 'PCM_24'
                else:
                    subtype = 'FLOAT'
                
                sf.write(save_path, audio_to_save, target_sr, subtype=subtype)
                
            elif output_format == 'flac':
                sf.write(save_path, audio_to_save, target_sr, format='FLAC')
                
            elif output_format == 'ogg':
                sf.write(save_path, audio_to_save, target_sr, format='OGG', subtype='VORBIS')
            
            QMessageBox.information(
                self, 
                'Success', 
                f'Audio converted successfully\n\nFormat: {output_format.upper()}\nSample Rate: {target_sr} Hz'
            )
            self.status_label.setText(f'Converted to {output_format.upper()}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Conversion failed:\n{str(e)}')
            
    def quick_save(self):
        """Quick save as WAV"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'No audio to save')
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Quick Save', '', 'WAV (*.wav)'
        )
        
        if save_path:
            if not save_path.endswith('.wav'):
                save_path += '.wav'
            
            try:
                sf.write(save_path, self.audio_data, self.sample_rate)
                self.status_label.setText('Saved successfully')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Save failed:\n{str(e)}')
            
    def quick_analyze(self):
        """Quick analysis of all features"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first')
            return
            
        # Switch to different tabs and trigger analysis
        self.tabs.setCurrentIndex(2)  # Extrema
        QTimer.singleShot(100, lambda: self.analyze_extrema())
        
        QTimer.singleShot(500, lambda: self.tabs.setCurrentIndex(3))  # Voice
        QTimer.singleShot(600, lambda: self.analyze_voice_segments())
        QTimer.singleShot(1000, lambda: self.tabs.setCurrentIndex(5))  # Spectrogram
        QTimer.singleShot(1100, lambda: self.create_spectrogram())
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, 
            'About',
            '<h2>Audio Signal Processor Pro</h2>'
            '<p><b>Version:</b> 4.0 Enhanced - Fixed</p>'
            '<p><b>Features:</b></p>'
            '<ul>'
            '<li>Multi-format support (WAV, FLAC, OGG, MP3, AAC, M4A)</li>'
            '<li>Video audio extraction (MP4, AVI, MOV, MKV)</li>'
            '<li>Advanced voice analysis and segmentation</li>'
            '<li>Separate graphs for voiced, unvoiced, and silence</li>'
            '<li>Interactive visualizations</li>'
            '<li>Professional themes</li>'
            '<li>Optimized performance with threading</li>'
            '</ul>'
            '<p><b>Technologies:</b> Python, PyQt5, librosa, matplotlib</p>'
        )
    
    def closeEvent(self, event):
        """Clean up on application close"""
        # Stop any audio playback
        self.stop_audio()
        
        # Stop any running workers
        if self.current_worker:
            try:
                self.current_worker.terminate()
                self.current_worker.wait()
            except:
                pass
        
        # Accept the close event
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set default font
    font = QFont('Segoe UI', 10)
    app.setFont(font)
    
    # Create and show main window
    window = AudioProcessor()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

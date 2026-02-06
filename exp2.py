import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QTabWidget, QSpinBox, QComboBox, QGroupBox,
                             QMessageBox, QProgressBar, QSlider, QFrame,
                             QSplitter, QToolButton, QAction, QMenu, QStatusBar,
                             QCheckBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QLinearGradient, QPainter, QPen
import librosa
import librosa.display
from PIL import Image


class ModernButton(QPushButton):
    """Custom styled button with hover effects"""
    def __init__(self, text, color="#2196F3"):
        super().__init__(text)
        self.color = color
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        self.update_style()
        
    def update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.adjust_color(self.color, 20)};
            }}
            QPushButton:pressed {{
                background-color: {self.adjust_color(self.color, -20)};
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
                color: #888888;
            }}
        """)
    
    def adjust_color(self, hex_color, amount):
        """Lighten or darken a color"""
        color = QColor(hex_color)
        h, s, v, a = color.getHsv()
        v = max(0, min(255, v + amount))
        color.setHsv(h, s, v, a)
        return color.name()


class AnimatedProgressBar(QProgressBar):
    """Custom progress bar with gradient and animation"""
    def __init__(self):
        super().__init__()
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignCenter)
        self.update_style()
        
    def update_style(self):
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                text-align: center;
                background-color: #F5F5F5;
                color: #333;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #8BC34A);
                border-radius: 6px;
            }
        """)


class ThemeManager:
    """Manages application themes"""
    
    THEMES = {
        'Light': {
            'background': '#FFFFFF',
            'surface': '#F5F5F5',
            'primary': '#2196F3',
            'secondary': '#FF9800',
            'accent': '#4CAF50',
            'text': '#212121',
            'border': '#E0E0E0',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336',
            'chart_bg': '#FFFFFF',
            'chart_grid': '#E0E0E0'
        },
        'Dark': {
            'background': '#1E1E1E',
            'surface': '#2D2D2D',
            'primary': '#64B5F6',
            'secondary': '#FFB74D',
            'accent': '#81C784',
            'text': '#FFFFFF',
            'border': '#404040',
            'success': '#66BB6A',
            'warning': '#FFA726',
            'error': '#EF5350',
            'chart_bg': '#2D2D2D',
            'chart_grid': '#404040'
        },
        'Ocean': {
            'background': '#E3F2FD',
            'surface': '#BBDEFB',
            'primary': '#0277BD',
            'secondary': '#00ACC1',
            'accent': '#00897B',
            'text': '#01579B',
            'border': '#90CAF9',
            'success': '#00897B',
            'warning': '#FFB300',
            'error': '#D32F2F',
            'chart_bg': '#E3F2FD',
            'chart_grid': '#90CAF9'
        },
        'Sunset': {
            'background': '#FFF3E0',
            'surface': '#FFE0B2',
            'primary': '#F57C00',
            'secondary': '#E64A19',
            'accent': '#7B1FA2',
            'text': '#4E342E',
            'border': '#FFCC80',
            'success': '#689F38',
            'warning': '#F57C00',
            'error': '#D32F2F',
            'chart_bg': '#FFF3E0',
            'chart_grid': '#FFCC80'
        },
        'Forest': {
            'background': '#E8F5E9',
            'surface': '#C8E6C9',
            'primary': '#388E3C',
            'secondary': '#689F38',
            'accent': '#FFA726',
            'text': '#1B5E20',
            'border': '#A5D6A7',
            'success': '#66BB6A',
            'warning': '#FFB300',
            'error': '#E53935',
            'chart_bg': '#E8F5E9',
            'chart_grid': '#A5D6A7'
        }
    }
    
    @staticmethod
    def get_stylesheet(theme_name):
        """Generate complete stylesheet for theme"""
        theme = ThemeManager.THEMES.get(theme_name, ThemeManager.THEMES['Light'])
        
        return f"""
            QMainWindow {{
                background-color: {theme['background']};
            }}
            QWidget {{
                background-color: {theme['background']};
                color: {theme['text']};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QGroupBox {{
                border: 2px solid {theme['border']};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: {theme['surface']};
                font-weight: bold;
                font-size: 13px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {theme['primary']};
            }}
            QLabel {{
                color: {theme['text']};
                background-color: transparent;
            }}
            QTabWidget::pane {{
                border: 2px solid {theme['border']};
                border-radius: 8px;
                background-color: {theme['surface']};
            }}
            QTabBar::tab {{
                background-color: {theme['surface']};
                color: {theme['text']};
                border: 1px solid {theme['border']};
                padding: 10px 20px;
                margin: 2px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {theme['primary']};
                color: white;
            }}
            QTabBar::tab:hover {{
                background-color: {theme['accent']};
                color: white;
            }}
            QComboBox {{
                background-color: {theme['surface']};
                border: 2px solid {theme['border']};
                border-radius: 6px;
                padding: 6px;
                color: {theme['text']};
                min-height: 28px;
            }}
            QComboBox:hover {{
                border-color: {theme['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QSpinBox {{
                background-color: {theme['surface']};
                border: 2px solid {theme['border']};
                border-radius: 6px;
                padding: 6px;
                color: {theme['text']};
                min-height: 28px;
            }}
            QSpinBox:hover {{
                border-color: {theme['primary']};
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: {theme['border']};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {theme['primary']};
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {theme['accent']};
            }}
            QCheckBox {{
                spacing: 8px;
                color: {theme['text']};
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {theme['border']};
                border-radius: 4px;
                background-color: {theme['surface']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {theme['primary']};
                border-color: {theme['primary']};
            }}
            QRadioButton {{
                spacing: 8px;
                color: {theme['text']};
            }}
            QRadioButton::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {theme['border']};
                border-radius: 10px;
                background-color: {theme['surface']};
            }}
            QRadioButton::indicator:checked {{
                background-color: {theme['primary']};
                border-color: {theme['primary']};
            }}
            QStatusBar {{
                background-color: {theme['surface']};
                color: {theme['text']};
                border-top: 2px solid {theme['border']};
            }}
        """


class InteractivePlotCanvas(FigureCanvas):
    """Enhanced matplotlib canvas with interactivity"""
    def __init__(self, parent=None, width=8, height=5, dpi=100, theme='Light'):
        self.theme_name = theme
        self.theme = ThemeManager.THEMES[theme]
        
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor=self.theme['chart_bg'])
        self.axes = self.figure.add_subplot(111)
        
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.setup_theme()
        self.mpl_connect('motion_notify_event', self.on_hover)
        
    def setup_theme(self):
        """Apply theme to matplotlib figure"""
        self.figure.patch.set_facecolor(self.theme['chart_bg'])
        self.axes.set_facecolor(self.theme['chart_bg'])
        self.axes.spines['bottom'].set_color(self.theme['text'])
        self.axes.spines['top'].set_color(self.theme['border'])
        self.axes.spines['right'].set_color(self.theme['border'])
        self.axes.spines['left'].set_color(self.theme['text'])
        self.axes.tick_params(colors=self.theme['text'], which='both')
        self.axes.xaxis.label.set_color(self.theme['text'])
        self.axes.yaxis.label.set_color(self.theme['text'])
        self.axes.title.set_color(self.theme['primary'])
        self.axes.grid(True, alpha=0.3, color=self.theme['chart_grid'])
        
    def update_theme(self, theme_name):
        """Update canvas theme"""
        self.theme_name = theme_name
        self.theme = ThemeManager.THEMES[theme_name]
        self.setup_theme()
        self.draw()
        
    def on_hover(self, event):
        """Show coordinates on hover"""
        if event.inaxes == self.axes:
            self.setToolTip(f'x: {event.xdata:.4f}, y: {event.ydata:.4f}')


class VoiceSignalProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_data = None
        self.sample_rate = 44100
        self.recording = False
        self.recorded_frames = []
        self.duration = 10
        self.current_theme = 'Dark'  # Default theme
        self.second_signal = None
        
        # Animation state
        self.is_playing = False
        self.play_timer = None
        
        self.initUI()
        self.apply_theme(self.current_theme)
        
    def initUI(self):
        self.setWindowTitle('🎙️ Voice Signal Processor Pro')
        self.setGeometry(50, 50, 1600, 950)
        
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
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs)
        
        # Create all tabs
        self.create_waveform_tab()
        self.create_operations_tab()
        self.create_extrema_tab()
        self.create_voice_analysis_tab()
        self.create_spectrogram_tab()
        self.create_converter_tab()
        
        # Status bar with widgets
        self.create_status_bar()
        
    def create_menu_bar(self):
        """Create modern menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('📁 File')
        
        load_action = QAction('🎵 Load Audio', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_audio_file)
        file_menu.addAction(load_action)
        
        save_action = QAction('💾 Save Audio', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.quick_save)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('🚪 Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('👁️ View')
        
        # Theme submenu
        theme_menu = view_menu.addMenu('🎨 Themes')
        for theme_name in ThemeManager.THEMES.keys():
            theme_action = QAction(theme_name, self)
            theme_action.triggered.connect(lambda checked, t=theme_name: self.apply_theme(t))
            theme_menu.addAction(theme_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('🔧 Tools')
        
        analyze_action = QAction('📊 Quick Analyze', self)
        analyze_action.setShortcut('Ctrl+A')
        analyze_action.triggered.connect(self.quick_analyze)
        tools_menu.addAction(analyze_action)
        
        # Help menu
        help_menu = menubar.addMenu('❓ Help')
        
        about_action = QAction('ℹ️ About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_header(self):
        """Create modern header"""
        header = QFrame()
        header.setFrameStyle(QFrame.StyledPanel)
        header.setMaximumHeight(100)
        
        layout = QHBoxLayout()
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title = QLabel('🎙️ Voice Signal Processor Pro')
        title.setFont(QFont('Segoe UI', 20, QFont.Bold))
        title_layout.addWidget(title)
        
        subtitle = QLabel('Advanced Audio Analysis & Processing Suite')
        subtitle.setFont(QFont('Segoe UI', 10))
        title_layout.addWidget(subtitle)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Quick theme switcher
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
        """Create interactive control panel"""
        group = QGroupBox('🎛️ Audio Controls')
        main_layout = QVBoxLayout()
        
        # Row 1: Recording controls
        row1 = QHBoxLayout()
        
        self.record_btn = ModernButton('🔴 Start Recording', '#F44336')
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
        
        self.load_btn = ModernButton('📂 Load File', '#2196F3')
        self.load_btn.clicked.connect(self.load_audio_file)
        row1.addWidget(self.load_btn)
        
        self.play_btn = ModernButton('▶️ Play', '#4CAF50')
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        row1.addWidget(self.play_btn)
        
        self.stop_btn = ModernButton('⏹️ Stop', '#FF9800')
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        row1.addWidget(self.stop_btn)
        
        main_layout.addLayout(row1)
        
        # Row 2: Progress and info
        row2 = QHBoxLayout()
        
        self.progress_bar = AnimatedProgressBar()
        self.progress_bar.setVisible(False)
        row2.addWidget(self.progress_bar)
        
        self.info_label = QLabel('No audio loaded')
        self.info_label.setStyleSheet("padding: 8px; font-weight: bold;")
        row2.addWidget(self.info_label)
        
        main_layout.addLayout(row2)
        
        # Row 3: Volume control
        row3 = QHBoxLayout()
        row3.addWidget(QLabel('🔊 Volume:'))
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.update_volume)
        row3.addWidget(self.volume_slider)
        
        self.volume_label = QLabel('100%')
        self.volume_label.setMinimumWidth(50)
        row3.addWidget(self.volume_label)
        
        main_layout.addLayout(row3)
        
        group.setLayout(main_layout)
        return group
        
    def create_status_bar(self):
        """Create enhanced status bar"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        self.status_label = QLabel('Ready')
        status_bar.addWidget(self.status_label, 1)
        
        # Add sample rate indicator
        self.sr_label = QLabel('SR: --')
        status_bar.addPermanentWidget(self.sr_label)
        
        # Add duration indicator
        self.duration_label = QLabel('Duration: --')
        status_bar.addPermanentWidget(self.duration_label)
        
        # Add file size indicator
        self.size_label = QLabel('Size: --')
        status_bar.addPermanentWidget(self.size_label)
        
    def create_waveform_tab(self):
        """Enhanced waveform tab with zoom controls"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls = QHBoxLayout()
        
        zoom_in_btn = ModernButton('🔍 Zoom In', '#4CAF50')
        zoom_in_btn.clicked.connect(lambda: self.zoom_waveform(1.5))
        controls.addWidget(zoom_in_btn)
        
        zoom_out_btn = ModernButton('🔍 Zoom Out', '#FF9800')
        zoom_out_btn.clicked.connect(lambda: self.zoom_waveform(0.67))
        controls.addWidget(zoom_out_btn)
        
        reset_btn = ModernButton('↺ Reset View', '#2196F3')
        reset_btn.clicked.connect(self.plot_waveform)
        controls.addWidget(reset_btn)
        
        controls.addStretch()
        
        self.show_envelope = QCheckBox('Show Envelope')
        self.show_envelope.stateChanged.connect(self.plot_waveform)
        controls.addWidget(self.show_envelope)
        
        layout.addLayout(controls)
        
        # Canvas
        self.waveform_canvas = InteractivePlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.waveform_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Waveform')
        
    def create_operations_tab(self):
        """Enhanced operations tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox('Operation Settings')
        controls_layout = QVBoxLayout()
        
        # Operation selection with radio buttons
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel('Select Operation:'))
        
        self.op_button_group = QButtonGroup()
        self.op_add = QRadioButton('➕ Addition')
        self.op_sub = QRadioButton('➖ Subtraction')
        self.op_conv = QRadioButton('🔄 Convolution')
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
        
        load_second_btn = ModernButton('📂 Load 2nd Signal', '#2196F3')
        load_second_btn.clicked.connect(self.load_second_signal)
        btn_layout.addWidget(load_second_btn)
        
        apply_btn = ModernButton('✨ Apply Operation', '#4CAF50')
        apply_btn.clicked.connect(self.apply_operation)
        btn_layout.addWidget(apply_btn)
        
        controls_layout.addLayout(btn_layout)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Canvas
        self.operations_canvas = InteractivePlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.operations_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Operations')
        
    def create_extrema_tab(self):
        """Enhanced extrema tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        controls = QHBoxLayout()
        
        analyze_btn = ModernButton('🔍 Analyze Extrema', '#4CAF50')
        analyze_btn.clicked.connect(self.analyze_extrema)
        controls.addWidget(analyze_btn)
        
        controls.addWidget(QLabel('Sensitivity:'))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(20)
        self.sensitivity_slider.setValue(10)
        self.sensitivity_slider.setToolTip('Adjust peak detection sensitivity')
        controls.addWidget(self.sensitivity_slider)
        
        controls.addStretch()
        
        layout.addLayout(controls)
        
        self.extrema_canvas = InteractivePlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.extrema_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Extrema')
        
    def create_voice_analysis_tab(self):
        """Enhanced voice analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        controls = QHBoxLayout()
        
        analyze_btn = ModernButton('🎤 Analyze Voice', '#9C27B0')
        analyze_btn.clicked.connect(self.analyze_voice_segments)
        controls.addWidget(analyze_btn)
        
        export_btn = ModernButton('💾 Export Report', '#FF9800')
        export_btn.clicked.connect(self.export_voice_report)
        controls.addWidget(export_btn)
        
        controls.addStretch()
        
        layout.addLayout(controls)
        
        self.voice_canvas = InteractivePlotCanvas(self, width=12, height=8, theme=self.current_theme)
        layout.addWidget(self.voice_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Voice Analysis')
        
    def create_spectrogram_tab(self):
        """Enhanced spectrogram tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        controls_group = QGroupBox('Spectrogram Settings')
        controls_layout = QVBoxLayout()
        
        # Row 1
        row1 = QHBoxLayout()
        
        create_btn = ModernButton('📊 Create Spectrogram', '#2196F3')
        create_btn.clicked.connect(self.create_spectrogram)
        row1.addWidget(create_btn)
        
        histogram_btn = ModernButton('📈 Histogram', '#FF9800')
        histogram_btn.clicked.connect(self.spectrogram_to_histogram)
        row1.addWidget(histogram_btn)
        
        row1.addStretch()
        controls_layout.addLayout(row1)
        
        # Row 2: Colormap selection
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Colormap:'))
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                                      'hot', 'cool', 'jet', 'rainbow'])
        row2.addWidget(self.colormap_combo)
        
        row2.addStretch()
        controls_layout.addLayout(row2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        self.spectrogram_canvas = InteractivePlotCanvas(self, width=12, height=6, theme=self.current_theme)
        layout.addWidget(self.spectrogram_canvas)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Spectrogram')
        
    def create_converter_tab(self):
        """Enhanced converter tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Info card
        info_group = QGroupBox('ℹ️ Format Conversion')
        info_layout = QVBoxLayout()
        info_text = QLabel(
            '<b>Supported Formats:</b><br>'
            '• WAV - Uncompressed (16/24/32-bit)<br>'
            '• FLAC - Lossless compression<br>'
            '• OGG - Open-source compressed<br><br>'
            '<b>No FFmpeg required!</b>'
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Conversion settings
        settings_group = QGroupBox('⚙️ Conversion Settings')
        settings_layout = QVBoxLayout()
        
        # Format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel('Format:'))
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
                                '44100 Hz', '48000 Hz'])
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
        convert_btn = ModernButton('🔄 Convert & Save', '#4CAF50')
        convert_btn.setMinimumHeight(50)
        convert_btn.clicked.connect(self.convert_audio_format)
        layout.addWidget(convert_btn)
        
        layout.addStretch()
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, 'Converter')
        
    def apply_theme(self, theme_name):
        """Apply theme to entire application"""
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
        if hasattr(self, 'spectrogram_canvas'):
            self.spectrogram_canvas.update_theme(theme_name)
            
        # Update button colors based on theme
        theme = ThemeManager.THEMES[theme_name]
        if hasattr(self, 'record_btn'):
            self.record_btn.color = theme['error']
            self.record_btn.update_style()
        if hasattr(self, 'load_btn'):
            self.load_btn.color = theme['primary']
            self.load_btn.update_style()
        if hasattr(self, 'play_btn'):
            self.play_btn.color = theme['success']
            self.play_btn.update_style()
            
        self.statusBar().showMessage(f'Theme changed to: {theme_name}', 3000)
        
    def update_duration(self, value):
        self.duration = value
        
    def update_volume(self, value):
        """Update volume display"""
        self.volume_label.setText(f'{value}%')
        
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.recording = True
        self.recorded_frames = []
        self.record_btn.setText('⏹️ Stop Recording')
        self.status_label.setText(f'🔴 Recording for {self.duration} seconds...')
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.recorded_frames.append(indata.copy())
        
        self.stream = sd.InputStream(callback=callback, channels=1, 
                                     samplerate=self.sample_rate)
        self.stream.start()
        
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.update_recording_progress)
        self.record_timer.start(100)
        
        self.recording_start_time = 0
        
    def update_recording_progress(self):
        self.recording_start_time += 0.1
        progress = int((self.recording_start_time / self.duration) * 100)
        self.progress_bar.setValue(min(progress, 100))
        
        if self.recording_start_time >= self.duration:
            self.stop_recording()
            
    def stop_recording(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.recording = False
        self.record_btn.setText('🔴 Start Recording')
        self.record_timer.stop()
        self.progress_bar.setVisible(False)
        
        if self.recorded_frames:
            self.audio_data = np.concatenate(self.recorded_frames, axis=0)
            self.status_label.setText('✅ Recording completed!')
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.update_audio_info()
            self.plot_waveform()
        
    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Audio File', '', 
            'Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)'
        )
        
        if file_path:
            try:
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None, mono=True)
                self.audio_data = self.audio_data.reshape(-1, 1)
                self.status_label.setText(f'✅ Loaded: {file_path.split("/")[-1]}')
                self.play_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.update_audio_info()
                self.plot_waveform()
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load audio:\n{str(e)}')
                
    def update_audio_info(self):
        """Update status bar info"""
        if self.audio_data is not None:
            duration = len(self.audio_data) / self.sample_rate
            size = self.audio_data.nbytes / 1024  # KB
            
            self.info_label.setText(f'✅ Audio loaded')
            self.sr_label.setText(f'SR: {self.sample_rate} Hz')
            self.duration_label.setText(f'Duration: {duration:.2f}s')
            self.size_label.setText(f'Size: {size:.1f} KB')
            
    def play_audio(self):
        if self.audio_data is not None:
            volume = self.volume_slider.value() / 100
            sd.play(self.audio_data * volume, self.sample_rate)
            self.status_label.setText('▶️ Playing audio...')
            self.is_playing = True
            
    def stop_audio(self):
        sd.stop()
        self.status_label.setText('⏹️ Playback stopped')
        self.is_playing = False
        
    def plot_waveform(self):
        if self.audio_data is None:
            return
            
        self.waveform_canvas.axes.clear()
        
        audio_1d = self.audio_data.flatten()
        time_axis = np.arange(len(audio_1d)) / self.sample_rate
        
        # Plot waveform
        theme = ThemeManager.THEMES[self.current_theme]
        self.waveform_canvas.axes.plot(time_axis, audio_1d, linewidth=0.8, 
                                       color=theme['primary'], alpha=0.7)
        
        # Optionally show envelope
        if self.show_envelope.isChecked():
            envelope = np.abs(signal.hilbert(audio_1d))
            self.waveform_canvas.axes.plot(time_axis, envelope, linewidth=1.5, 
                                          color=theme['accent'], alpha=0.5, 
                                          label='Envelope')
            self.waveform_canvas.axes.plot(time_axis, -envelope, linewidth=1.5, 
                                          color=theme['accent'], alpha=0.5)
            self.waveform_canvas.axes.legend()
        
        self.waveform_canvas.axes.set_xlabel('Time (s)', fontweight='bold')
        self.waveform_canvas.axes.set_ylabel('Amplitude', fontweight='bold')
        self.waveform_canvas.axes.set_title(f'Waveform - {self.sample_rate} Hz', 
                                           fontweight='bold', fontsize=12)
        self.waveform_canvas.setup_theme()
        self.waveform_canvas.draw()
        
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
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Second Signal', '', 
            'Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)'
        )
        
        if file_path:
            try:
                self.second_signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                self.second_signal = self.second_signal.reshape(-1, 1)
                self.status_label.setText(f'✅ Second signal loaded')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load:\n{str(e)}')
                
    def apply_operation(self):
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first!')
            return
            
        self.operations_canvas.axes.clear()
        theme = ThemeManager.THEMES[self.current_theme]
        
        if self.op_add.isChecked() or self.op_sub.isChecked():
            if self.second_signal is None:
                QMessageBox.warning(self, 'Warning', 'Load second signal!')
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
            self.operations_canvas.axes.plot(time_axis, result, linewidth=0.8,
                                            color=theme['primary'])
            
        elif self.op_conv.isChecked():
            kernel = np.ones(100) / 100
            result = np.convolve(self.audio_data.flatten(), kernel, mode='same')
            
            time_axis = np.arange(len(result)) / self.sample_rate
            self.operations_canvas.axes.plot(time_axis, result, linewidth=0.8,
                                            color=theme['accent'])
            title = 'Convolution Result'
            
        self.operations_canvas.axes.set_xlabel('Time (s)', fontweight='bold')
        self.operations_canvas.axes.set_ylabel('Amplitude', fontweight='bold')
        self.operations_canvas.axes.set_title(title, fontweight='bold', fontsize=12)
        self.operations_canvas.setup_theme()
        self.operations_canvas.draw()
        
        self.status_label.setText(f'✅ Operation applied')
        
    def analyze_extrema(self):
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first!')
            return
            
        self.extrema_canvas.axes.clear()
        
        audio_1d = self.audio_data.flatten()
        sensitivity = self.sensitivity_slider.value()
        window_size = int(sensitivity * 0.001 * self.sample_rate)
        
        maxima_indices = signal.argrelextrema(audio_1d, np.greater, order=window_size)[0]
        minima_indices = signal.argrelextrema(audio_1d, np.less, order=window_size)[0]
        
        time_axis = np.arange(len(audio_1d)) / self.sample_rate
        theme = ThemeManager.THEMES[self.current_theme]
        
        self.extrema_canvas.axes.plot(time_axis, audio_1d, linewidth=0.6, 
                                     color=theme['primary'], alpha=0.5, label='Signal')
        
        if len(maxima_indices) > 0:
            self.extrema_canvas.axes.scatter(time_axis[maxima_indices], 
                                            audio_1d[maxima_indices],
                                            c=theme['success'], marker='^', s=50,
                                            label=f'Maxima ({len(maxima_indices)})', 
                                            zorder=5)
        
        if len(minima_indices) > 0:
            self.extrema_canvas.axes.scatter(time_axis[minima_indices], 
                                            audio_1d[minima_indices],
                                            c=theme['error'], marker='v', s=50,
                                            label=f'Minima ({len(minima_indices)})',
                                            zorder=5)
        
        self.extrema_canvas.axes.set_xlabel('Time (s)', fontweight='bold')
        self.extrema_canvas.axes.set_ylabel('Amplitude', fontweight='bold')
        self.extrema_canvas.axes.set_title('Extrema Analysis', fontweight='bold', fontsize=12)
        self.extrema_canvas.axes.legend()
        self.extrema_canvas.setup_theme()
        self.extrema_canvas.draw()
        
        self.status_label.setText(f'✅ Found {len(maxima_indices)} maxima, {len(minima_indices)} minima')
        
    def analyze_voice_segments(self):
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first!')
            return
            
        self.voice_canvas.figure.clear()
        
        audio_1d = self.audio_data.flatten()
        frame_length = int(0.025 * self.sample_rate)
        hop_length = int(0.010 * self.sample_rate)
        
        energy = np.array([
            np.sum(audio_1d[i:i+frame_length]**2)
            for i in range(0, len(audio_1d)-frame_length, hop_length)
        ])
        
        zcr = np.array([
            np.sum(np.abs(np.diff(np.sign(audio_1d[i:i+frame_length])))) / (2 * frame_length)
            for i in range(0, len(audio_1d)-frame_length, hop_length)
        ])
        
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)
        
        silence_threshold = 0.03
        voiced_zcr_threshold = 0.3
        
        segments = np.zeros(len(energy))
        for i in range(len(energy)):
            if energy_norm[i] < silence_threshold:
                segments[i] = 0
            elif zcr_norm[i] < voiced_zcr_threshold:
                segments[i] = 1
            else:
                segments[i] = 2
        
        time_frames = np.arange(len(energy)) * hop_length / self.sample_rate
        time_signal = np.arange(len(audio_1d)) / self.sample_rate
        
        theme = ThemeManager.THEMES[self.current_theme]
        
        ax1 = self.voice_canvas.figure.add_subplot(311)
        ax1.plot(time_signal, audio_1d, linewidth=0.6, color=theme['primary'])
        ax1.set_ylabel('Amplitude', fontweight='bold')
        ax1.set_title('Original Signal', fontweight='bold')
        ax1.grid(True, alpha=0.3, color=theme['chart_grid'])
        ax1.set_facecolor(theme['chart_bg'])
        
        ax2 = self.voice_canvas.figure.add_subplot(312)
        ax2.plot(time_frames, energy_norm, label='Energy', linewidth=2, 
                color=theme['success'])
        ax2.plot(time_frames, zcr_norm, label='ZCR', linewidth=2,
                color=theme['warning'])
        ax2.axhline(y=silence_threshold, color=theme['error'], linestyle='--', 
                   label='Silence threshold', linewidth=1.5)
        ax2.set_ylabel('Normalized', fontweight='bold')
        ax2.set_title('Features', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, color=theme['chart_grid'])
        ax2.set_facecolor(theme['chart_bg'])
        
        ax3 = self.voice_canvas.figure.add_subplot(313)
        colors_map = {0: theme['border'], 1: theme['primary'], 2: theme['warning']}
        labels_map = {0: 'Silence', 1: 'Voiced', 2: 'Unvoiced'}
        
        for i in range(3):
            mask = segments == i
            if np.any(mask):
                ax3.scatter(time_frames[mask], segments[mask],
                           c=colors_map[i], label=labels_map[i], s=20, alpha=0.7)
        
        ax3.set_xlabel('Time (s)', fontweight='bold')
        ax3.set_ylabel('Type', fontweight='bold')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Silence', 'Voiced', 'Unvoiced'])
        ax3.set_title('Voice Activity Detection', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, color=theme['chart_grid'])
        ax3.set_facecolor(theme['chart_bg'])
        
        self.voice_canvas.figure.tight_layout()
        self.voice_canvas.draw()
        
        silence_pct = np.sum(segments == 0) / len(segments) * 100
        voiced_pct = np.sum(segments == 1) / len(segments) * 100
        unvoiced_pct = np.sum(segments == 2) / len(segments) * 100
        
        self.status_label.setText(
            f'Silence: {silence_pct:.1f}% | Voiced: {voiced_pct:.1f}% | Unvoiced: {unvoiced_pct:.1f}%'
        )
        
        # Store for export
        self.voice_stats = {
            'silence': silence_pct,
            'voiced': voiced_pct,
            'unvoiced': unvoiced_pct
        }
        
    def export_voice_report(self):
        """Export voice analysis report"""
        if not hasattr(self, 'voice_stats'):
            QMessageBox.warning(self, 'Warning', 'Run voice analysis first!')
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Report', '', 'Text Files (*.txt);;All Files (*)'
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write('=== Voice Analysis Report ===\n\n')
                f.write(f'Silence: {self.voice_stats["silence"]:.2f}%\n')
                f.write(f'Voiced: {self.voice_stats["voiced"]:.2f}%\n')
                f.write(f'Unvoiced: {self.voice_stats["unvoiced"]:.2f}%\n')
                f.write(f'\nSample Rate: {self.sample_rate} Hz\n')
                f.write(f'Duration: {len(self.audio_data)/self.sample_rate:.2f}s\n')
                
            self.status_label.setText(f'✅ Report saved')
        
    def create_spectrogram(self):
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first!')
            return
            
        self.spectrogram_canvas.axes.clear()
        
        audio_1d = self.audio_data.flatten()
        D = librosa.stft(audio_1d)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        self.current_spectrogram = S_db
        
        colormap = self.colormap_combo.currentText()
        img = librosa.display.specshow(S_db, sr=self.sample_rate, x_axis='time', 
                                       y_axis='hz', ax=self.spectrogram_canvas.axes,
                                       cmap=colormap)
        
        self.spectrogram_canvas.axes.set_title('Spectrogram', fontweight='bold', fontsize=12)
        self.spectrogram_canvas.figure.colorbar(img, ax=self.spectrogram_canvas.axes, 
                                                format='%+2.0f dB')
        
        self.spectrogram_canvas.setup_theme()
        self.spectrogram_canvas.draw()
        self.status_label.setText('✅ Spectrogram created')
        
    def spectrogram_to_histogram(self):
        if not hasattr(self, 'current_spectrogram'):
            QMessageBox.warning(self, 'Warning', 'Create spectrogram first!')
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
        ax1.set_title('Spectrogram Image', fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        
        ax2 = self.spectrogram_canvas.figure.add_subplot(122)
        ax2.hist(histogram, bins=50, color=theme['primary'], edgecolor=theme['text'], alpha=0.7)
        ax2.set_title('Intensity Histogram', fontweight='bold')
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        self.spectrogram_canvas.figure.tight_layout()
        self.spectrogram_canvas.draw()
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 'PNG (*.png);;JPEG (*.jpg)'
        )
        
        if save_path:
            img_rgb.save(save_path)
            self.status_label.setText(f'✅ Image saved')
    
    def convert_audio_format(self):
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first!')
            return
            
        output_format = self.format_combo.currentText().lower()
        sr_text = self.sr_combo.currentText()
        
        if sr_text == 'Keep Original':
            target_sr = self.sample_rate
        else:
            target_sr = int(sr_text.split()[0])
        
        audio_to_save = self.audio_data.flatten()
        
        if target_sr != self.sample_rate:
            audio_to_save = librosa.resample(audio_to_save, 
                                            orig_sr=self.sample_rate, 
                                            target_sr=target_sr)
        
        file_filter = f'{output_format.upper()} (*.{output_format})'
        save_path, _ = QFileDialog.getSaveFileName(
            self, f'Save as {output_format.upper()}', '', file_filter
        )
        
        if not save_path:
            return
            
        if not save_path.endswith(f'.{output_format}'):
            save_path += f'.{output_format}'
        
        try:
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
                self, 'Success', 
                f'✅ Audio converted!\n\nFormat: {output_format.upper()}\nSample Rate: {target_sr} Hz'
            )
            self.status_label.setText(f'✅ Converted to {output_format.upper()}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed:\n{str(e)}')
            
    def quick_save(self):
        """Quick save as WAV"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'No audio to save!')
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Quick Save', '', 'WAV (*.wav)'
        )
        
        if save_path:
            if not save_path.endswith('.wav'):
                save_path += '.wav'
            sf.write(save_path, self.audio_data, self.sample_rate)
            self.status_label.setText('✅ Saved')
            
    def quick_analyze(self):
        """Quick analysis of all features"""
        if self.audio_data is None:
            QMessageBox.warning(self, 'Warning', 'Load audio first!')
            return
            
        self.analyze_extrema()
        self.analyze_voice_segments()
        self.create_spectrogram()
        self.status_label.setText('✅ Quick analysis complete!')
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, 'About',
            '<h2>🎙️ Voice Signal Processor Pro</h2>'
            '<p><b>Version:</b> 2.0</p>'
            '<p><b>Features:</b></p>'
            '<ul>'
            '<li>Multi-format support (WAV, FLAC, OGG, MP3)</li>'
            '<li>Advanced voice analysis</li>'
            '<li>Interactive visualizations</li>'
            '<li>Multiple themes</li>'
            '<li>No FFmpeg required</li>'
            '</ul>'
            '<p><b>Created with:</b> Python, PyQt5, librosa, matplotlib</p>'
        )


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont('Segoe UI', 10)
    app.setFont(font)
    
    window = VoiceSignalProcessor()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
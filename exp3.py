"""
Advanced Spectrogram Analyzer with Qt GUI - Version 2.0 (Bug-Fixed & Enhanced)
Features: Time-Frequency Analysis, Comparative Analysis, and Creative Add-ons

Improvements:
- Fixed comparative analysis bugs
- Added proper window-specific parameters
- Improved error handling
- Enhanced UI/UX
- Better code organization and best practices
"""

import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QPushButton, QTabWidget, QScrollArea,
                             QGroupBox, QFormLayout, QCheckBox, QSlider, 
                             QListWidget, QMessageBox, QFileDialog, QTextEdit,
                             QSplitter, QTableWidget, QTableWidgetItem, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from datetime import datetime
import copy


class SpectrogramCanvas(FigureCanvas):
    """Custom canvas for spectrogram display with improved rendering"""
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.colorbar = None
        super(SpectrogramCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
    def plot_spectrogram(self, f, t, Sxx, title="Spectrogram", vmin=None, vmax=None):
        """Plot spectrogram with colorbar"""
        self.axes.clear()
        
        # Remove old colorbar if exists - with proper exception handling
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except (AttributeError, KeyError, ValueError):
                # Colorbar already removed or figure cleared
                pass
            finally:
                self.colorbar = None
        
        # Convert to dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)
        
        # Set vmin/vmax if not provided
        if vmin is None:
            vmin = np.percentile(Sxx_dB, 5)
        if vmax is None:
            vmax = np.percentile(Sxx_dB, 95)
        
        im = self.axes.pcolormesh(t, f, Sxx_dB, 
                                   shading='gouraud', cmap='viridis',
                                   vmin=vmin, vmax=vmax)
        self.axes.set_ylabel('Frequency [Hz]')
        self.axes.set_xlabel('Time [sec]')
        self.axes.set_title(title)
        self.axes.grid(True, alpha=0.3)
        
        self.colorbar = self.fig.colorbar(im, ax=self.axes, label='Power/Frequency [dB/Hz]')
        self.fig.tight_layout()
        self.draw()
        
    def clear_plot(self):
        """Clear the plot"""
        self.axes.clear()
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.draw()


class SignalGeneratorWidget(QWidget):
    """Widget for signal generation parameters"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout()
        
        # Frequency inputs for each segment
        self.freq1 = QDoubleSpinBox()
        self.freq1.setRange(1, 4000)
        self.freq1.setValue(100)
        self.freq1.setSuffix(" Hz")
        self.freq1.setDecimals(1)
        
        self.freq2 = QDoubleSpinBox()
        self.freq2.setRange(1, 4000)
        self.freq2.setValue(300)
        self.freq2.setSuffix(" Hz")
        self.freq2.setDecimals(1)
        
        self.freq3 = QDoubleSpinBox()
        self.freq3.setRange(1, 4000)
        self.freq3.setValue(600)
        self.freq3.setSuffix(" Hz")
        self.freq3.setDecimals(1)
        
        # Duration
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 10)
        self.duration.setValue(3.0)
        self.duration.setSuffix(" s")
        self.duration.setDecimals(1)
        
        # Sampling frequency
        self.fs = QSpinBox()
        self.fs.setRange(1000, 48000)
        self.fs.setValue(8000)
        self.fs.setSuffix(" Hz")
        self.fs.setSingleStep(1000)
        
        layout.addRow("Frequency 1 (0-1s):", self.freq1)
        layout.addRow("Frequency 2 (1-2s):", self.freq2)
        layout.addRow("Frequency 3 (2-3s):", self.freq3)
        layout.addRow("Total Duration:", self.duration)
        layout.addRow("Sampling Frequency:", self.fs)
        
        self.setLayout(layout)
        
    def get_params(self):
        """Get current signal parameters as dict"""
        return {
            'freq1': self.freq1.value(),
            'freq2': self.freq2.value(),
            'freq3': self.freq3.value(),
            'duration': self.duration.value(),
            'fs': self.fs.value()
        }


class WindowParametersWidget(QGroupBox):
    """Widget for window-specific parameters with dynamic UI"""
    
    # Signal emitted when window type changes
    windowTypeChanged = pyqtSignal(str)
    
    def __init__(self, title="Window-Specific Parameters"):
        super().__init__(title)
        self.init_ui()
        
    def init_ui(self):
        self.main_layout = QFormLayout()
        
        # Store parameter widgets
        self.param_widgets = {}
        
        # Hamming alpha parameter
        self.hamming_alpha = QDoubleSpinBox()
        self.hamming_alpha.setRange(0, 1)
        self.hamming_alpha.setValue(0.54)
        self.hamming_alpha.setSingleStep(0.01)
        self.hamming_alpha.setToolTip("0.54 = Standard Hamming, 0.5 = Hann window")
        self.param_widgets['hamming_alpha'] = ('Alpha (α):', self.hamming_alpha)
        
        # Kaiser beta parameter
        self.kaiser_beta = QDoubleSpinBox()
        self.kaiser_beta.setRange(0, 20)
        self.kaiser_beta.setValue(8.6)
        self.kaiser_beta.setSingleStep(0.1)
        self.kaiser_beta.setToolTip("Higher β = more side-lobe suppression (0=rect, 5=moderate, 8.6=good, 15=strong)")
        self.param_widgets['kaiser_beta'] = ('Beta (β):', self.kaiser_beta)
        
        # Tukey alpha parameter
        self.tukey_alpha = QDoubleSpinBox()
        self.tukey_alpha.setRange(0, 1)
        self.tukey_alpha.setValue(0.5)
        self.tukey_alpha.setSingleStep(0.05)
        self.tukey_alpha.setToolTip("Taper fraction (0=rect, 1=Hann)")
        self.param_widgets['tukey_alpha'] = ('Alpha (α):', self.tukey_alpha)
        
        # Gaussian std parameter
        self.gaussian_std = QDoubleSpinBox()
        self.gaussian_std.setRange(0.1, 10)
        self.gaussian_std.setValue(1.0)
        self.gaussian_std.setSingleStep(0.1)
        self.gaussian_std.setToolTip("Standard deviation in samples")
        self.param_widgets['gaussian_std'] = ('Std Dev (σ):', self.gaussian_std)
        
        # Blackman-Harris nterm parameter
        self.blackman_harris_nterm = QSpinBox()
        self.blackman_harris_nterm.setRange(3, 5)
        self.blackman_harris_nterm.setValue(4)
        self.blackman_harris_nterm.setToolTip("Number of terms (3-5, more = better suppression)")
        self.param_widgets['blackman_harris_nterm'] = ('Terms:', self.blackman_harris_nterm)
        
        self.setLayout(self.main_layout)
        
    def update_for_window(self, window_name):
        """Update displayed parameters based on window type"""
        # Clear current parameters
        for i in reversed(range(self.main_layout.count())): 
            self.main_layout.itemAt(i).widget().setParent(None)
        
        # Add relevant parameters based on window type
        if window_name == 'hamming':
            label, widget = self.param_widgets['hamming_alpha']
            self.main_layout.addRow(label, widget)
        elif window_name == 'kaiser':
            label, widget = self.param_widgets['kaiser_beta']
            self.main_layout.addRow(label, widget)
        elif window_name == 'tukey':
            label, widget = self.param_widgets['tukey_alpha']
            self.main_layout.addRow(label, widget)
        elif window_name == 'gaussian':
            label, widget = self.param_widgets['gaussian_std']
            self.main_layout.addRow(label, widget)
        else:
            # No specific parameters for this window
            info_label = QLabel("This window has no adjustable parameters")
            info_label.setStyleSheet("color: gray; font-style: italic;")
            self.main_layout.addRow(info_label)
    
    def get_window_params(self, window_name):
        """Get parameters for specific window type"""
        if window_name == 'hamming':
            return {'alpha': self.hamming_alpha.value()}
        elif window_name == 'kaiser':
            return {'beta': self.kaiser_beta.value()}
        elif window_name == 'tukey':
            return {'alpha': self.tukey_alpha.value()}
        elif window_name == 'gaussian':
            return {'std': self.gaussian_std.value()}
        else:
            return {}


class SpectrogramParametersWidget(QWidget):
    """Widget for spectrogram parameters with improved organization"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Window parameters group
        window_group = QGroupBox("Window Configuration")
        window_layout = QFormLayout()
        
        # Window type
        self.window_type = QComboBox()
        self.window_type.addItems([
            'hamming', 'hann', 'blackman', 'bartlett', 'kaiser',
            'tukey', 'boxcar', 'triang', 'parzen', 'bohman'
        ])
        self.window_type.currentTextChanged.connect(self.on_window_type_changed)
        
        # Window length
        self.window_length = QSpinBox()
        self.window_length.setRange(32, 4096)
        self.window_length.setValue(256)
        self.window_length.setSingleStep(32)
        self.window_length.setToolTip("Longer windows = better frequency resolution")
        
        # Overlap
        self.overlap = QSpinBox()
        self.overlap.setRange(0, 2048)
        self.overlap.setValue(128)
        self.overlap.setToolTip("Higher overlap = smoother spectrogram")
        
        # Overlap percentage slider
        self.overlap_pct = QSlider(Qt.Horizontal)
        self.overlap_pct.setRange(0, 95)
        self.overlap_pct.setValue(50)
        self.overlap_pct.setTickPosition(QSlider.TicksBelow)
        self.overlap_pct.setTickInterval(10)
        
        # Overlap percentage label
        self.overlap_pct_label = QLabel("50%")
        self.overlap_pct_label.setAlignment(Qt.AlignRight)
        
        # FFT size
        self.nfft = QSpinBox()
        self.nfft.setRange(64, 8192)
        self.nfft.setValue(512)
        self.nfft.setSingleStep(64)
        self.nfft.setToolTip("Higher FFT = more frequency bins")
        
        window_layout.addRow("Window Type:", self.window_type)
        window_layout.addRow("Window Length:", self.window_length)
        window_layout.addRow("Overlap (samples):", self.overlap)
        
        overlap_pct_layout = QHBoxLayout()
        overlap_pct_layout.addWidget(self.overlap_pct)
        overlap_pct_layout.addWidget(self.overlap_pct_label)
        window_layout.addRow("Overlap (%):", overlap_pct_layout)
        
        window_layout.addRow("FFT Size:", self.nfft)
        
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        # Window-specific parameters
        self.window_params_widget = WindowParametersWidget()
        layout.addWidget(self.window_params_widget)
        
        # Connect overlap synchronization
        self.overlap_pct.valueChanged.connect(self.sync_overlap_from_pct)
        self.overlap.valueChanged.connect(self.sync_overlap_from_samples)
        self.window_length.valueChanged.connect(self.update_max_overlap)
        
        # Initialize with hamming window parameters
        self.window_params_widget.update_for_window('hamming')
        
        self.setLayout(layout)
        
    def on_window_type_changed(self, window_name):
        """Handle window type change"""
        self.window_params_widget.update_for_window(window_name)
        
    def sync_overlap_from_pct(self, value):
        """Sync overlap samples from percentage"""
        samples = int(self.window_length.value() * value / 100)
        self.overlap.blockSignals(True)
        self.overlap.setValue(samples)
        self.overlap.blockSignals(False)
        self.overlap_pct_label.setText(f"{value}%")
        
    def sync_overlap_from_samples(self, value):
        """Sync overlap percentage from samples"""
        if self.window_length.value() > 0:
            pct = int(value * 100 / self.window_length.value())
            self.overlap_pct.blockSignals(True)
            self.overlap_pct.setValue(pct)
            self.overlap_pct.blockSignals(False)
            self.overlap_pct_label.setText(f"{pct}%")
        
    def update_max_overlap(self, value):
        """Update maximum overlap when window length changes"""
        self.overlap.setMaximum(value - 1)
        # Update overlap to maintain percentage
        current_pct = self.overlap_pct.value()
        self.sync_overlap_from_pct(current_pct)
        
    def get_window(self):
        """Get the window function with proper parameters"""
        window_name = self.window_type.currentText()
        N = self.window_length.value()
        window_params = self.window_params_widget.get_window_params(window_name)
        
        try:
            if window_name == 'hamming':
                alpha = window_params.get('alpha', 0.54)
                return signal.windows.general_hamming(N, alpha)
            elif window_name == 'hann':
                return signal.windows.hann(N)
            elif window_name == 'blackman':
                return signal.windows.blackman(N)
            elif window_name == 'bartlett':
                return signal.windows.bartlett(N)
            elif window_name == 'kaiser':
                beta = window_params.get('beta', 8.6)
                return signal.windows.kaiser(N, beta)
            elif window_name == 'tukey':
                alpha = window_params.get('alpha', 0.5)
                return signal.windows.tukey(N, alpha)
            elif window_name == 'boxcar':
                return signal.windows.boxcar(N)
            elif window_name == 'triang':
                return signal.windows.triang(N)
            elif window_name == 'parzen':
                return signal.windows.parzen(N)
            elif window_name == 'bohman':
                return signal.windows.bohman(N)
            else:
                return signal.windows.hamming(N)
        except Exception as e:
            QMessageBox.warning(None, "Window Error", 
                              f"Error creating window: {str(e)}\nUsing default Hamming window.")
            return signal.windows.hamming(N)
    
    def get_params(self):
        """Get all spectrogram parameters as dict"""
        window_name = self.window_type.currentText()
        params = {
            'window_type': window_name,
            'window_length': self.window_length.value(),
            'overlap': self.overlap.value(),
            'overlap_pct': self.overlap_pct.value(),
            'nfft': self.nfft.value(),
            'window_params': self.window_params_widget.get_window_params(window_name)
        }
        return params


class SpectrogramResult:
    """Class to store spectrogram computation results"""
    def __init__(self, f, t, Sxx, params, signal_params, timestamp=None):
        self.f = np.array(f)
        self.t = np.array(t)
        self.Sxx = np.array(Sxx)
        self.params = copy.deepcopy(params)  # Deep copy to avoid reference issues
        self.signal_params = copy.deepcopy(signal_params)
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def get_resolution(self):
        """Calculate frequency and time resolution"""
        freq_res = self.f[1] - self.f[0] if len(self.f) > 1 else 0
        time_res = self.t[1] - self.t[0] if len(self.t) > 1 else 0
        return freq_res, time_res
        
    def get_max_intensity(self):
        """Get maximum intensity"""
        return np.max(10 * np.log10(self.Sxx + 1e-10))
        
    def get_mean_intensity(self):
        """Get mean intensity"""
        return np.mean(10 * np.log10(self.Sxx + 1e-10))
    
    def get_display_name(self):
        """Get a display name for the result"""
        window_str = self.params['window_type']
        if self.params.get('window_params'):
            # Add window parameters to display
            wp = self.params['window_params']
            param_strs = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                         for k, v in wp.items()]
            if param_strs:
                window_str += f"({','.join(param_strs)})"
        
        return (f"{self.timestamp} - {window_str} "
                f"[W:{self.params['window_length']}, "
                f"O:{self.params['overlap_pct']}%, "
                f"FFT:{self.params['nfft']}]")


class ComparativeAnalysisTab(QWidget):
    """Tab for comparing multiple spectrograms with bug fixes"""
    def __init__(self):
        super().__init__()
        self.results = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header_label = QLabel("Comparative Spectrogram Analysis")
        header_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2E86AB;")
        
        # List of added spectrograms
        list_label = QLabel("Added Spectrograms (Select 2 or more to compare):")
        list_label.setStyleSheet("font-weight: bold;")
        
        self.result_list = QListWidget()
        self.result_list.setSelectionMode(QListWidget.MultiSelection)
        self.result_list.setAlternatingRowColors(True)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_compare = QPushButton("Compare Selected")
        self.btn_compare.clicked.connect(self.compare_spectrograms)
        self.btn_compare.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1F5F7A;
            }
        """)
        
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self.remove_selected)
        
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_all)
        
        btn_layout.addWidget(self.btn_compare)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch()
        
        # Comparison display
        comparison_label = QLabel("Comparison Analysis Report:")
        comparison_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        self.comparison_text.setMaximumHeight(200)
        font = QFont("Courier", 9)
        self.comparison_text.setFont(font)
        
        # Comparison plots
        self.comparison_canvas = SpectrogramCanvas(self, width=12, height=8)
        self.comparison_toolbar = NavigationToolbar(self.comparison_canvas, self)
        
        layout.addWidget(header_label)
        layout.addWidget(list_label)
        layout.addWidget(self.result_list)
        layout.addLayout(btn_layout)
        layout.addWidget(comparison_label)
        layout.addWidget(self.comparison_text)
        layout.addWidget(self.comparison_toolbar)
        layout.addWidget(self.comparison_canvas)
        
        self.setLayout(layout)
        
    def add_result(self, result):
        """Add a spectrogram result for comparison with validation"""
        try:
            # Validate result data
            if result is None:
                raise ValueError("Result is None")
            
            if not hasattr(result, 'f') or not hasattr(result, 't') or not hasattr(result, 'Sxx'):
                raise ValueError("Result missing required attributes (f, t, or Sxx)")
            
            if result.f is None or result.t is None or result.Sxx is None:
                raise ValueError("Result contains None values")
            
            if len(result.f) == 0 or len(result.t) == 0 or result.Sxx.size == 0:
                raise ValueError("Result contains empty data arrays")
            
            # Deep copy the result to avoid reference issues
            result_copy = SpectrogramResult(
                result.f.copy(), 
                result.t.copy(), 
                result.Sxx.copy(),
                copy.deepcopy(result.params), 
                copy.deepcopy(result.signal_params),
                result.timestamp
            )
            
            # Validate copied data
            if np.any(np.isnan(result_copy.Sxx)) or np.any(np.isinf(result_copy.Sxx)):
                raise ValueError("Result contains NaN or Inf values")
            
            self.results.append(result_copy)
            
            # Add to list with better formatting
            display_name = result_copy.get_display_name()
            item = QListWidgetItem(display_name)
            
            # Color code by window type
            window_colors = {
                'hamming': QColor(100, 200, 100),
                'hann': QColor(100, 150, 200),
                'blackman': QColor(200, 100, 100),
                'kaiser': QColor(200, 150, 100),
                'bartlett': QColor(150, 100, 200),
                'tukey': QColor(150, 200, 150),
                'boxcar': QColor(200, 200, 100),
                'triang': QColor(100, 200, 200),
                'parzen': QColor(200, 100, 200),
                'bohman': QColor(150, 150, 200),
            }
            color = window_colors.get(result.params['window_type'], QColor(150, 150, 150))
            item.setBackground(color)
            
            self.result_list.addItem(item)
            
            QMessageBox.information(self, "Added Successfully", 
                                  f"Spectrogram #{len(self.results)} added to comparison list!\n\n"
                                  f"Window: {result.params['window_type']}\n"
                                  f"Resolution: Δf={result.get_resolution()[0]:.4f} Hz, "
                                  f"Δt={result.get_resolution()[1]:.4f} s")
                                  
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", 
                               f"Invalid result data: {str(e)}\n\n"
                               f"Please regenerate the spectrogram and try again.")
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to add result: {str(e)}\n\n"
                               f"This is likely a bug. Please report it.")
            import traceback
            traceback.print_exc()
    
    def remove_selected(self):
        """Remove selected spectrograms from list"""
        selected_items = self.result_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select spectrograms to remove")
            return
        
        # Remove from back to front to maintain indices
        for item in selected_items:
            row = self.result_list.row(item)
            self.result_list.takeItem(row)
            del self.results[row]
        
        QMessageBox.information(self, "Removed", 
                              f"{len(selected_items)} spectrogram(s) removed")
        
    def clear_all(self):
        """Clear all results"""
        if not self.results:
            return
            
        reply = QMessageBox.question(self, "Confirm Clear", 
                                    "Are you sure you want to clear all spectrograms?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.results.clear()
            self.result_list.clear()
            self.comparison_text.clear()
            self.comparison_canvas.clear_plot()
            QMessageBox.information(self, "Cleared", "All spectrograms cleared")
        
    def compare_spectrograms(self):
        """Compare selected spectrograms with comprehensive analysis and debugging"""
        selected_indices = [self.result_list.row(item) 
                           for item in self.result_list.selectedItems()]
        
        if len(selected_indices) < 2:
            QMessageBox.warning(self, "Selection Error", 
                              "Please select at least 2 spectrograms to compare")
            return
        
        try:
            # Build comparison text with extensive analysis
            comparison_text = "=" * 80 + "\n"
            comparison_text += "COMPREHENSIVE COMPARATIVE SPECTROGRAM ANALYSIS\n"
            comparison_text += "=" * 80 + "\n\n"
            
            selected_results = [self.results[i] for i in selected_indices]
            
            # Signal parameters comparison
            comparison_text += "SIGNAL PARAMETERS:\n"
            comparison_text += "-" * 80 + "\n"
            sp = selected_results[0].signal_params
            comparison_text += f"Frequencies: {sp['freq1']:.1f} Hz, {sp['freq2']:.1f} Hz, {sp['freq3']:.1f} Hz\n"
            comparison_text += f"Duration: {sp['duration']:.1f} s | Sampling Rate: {sp['fs']} Hz\n"
            comparison_text += f"Note: All spectrograms use the same signal for fair comparison\n\n"
            
            # Window configuration comparison
            comparison_text += "WINDOW CONFIGURATIONS:\n"
            comparison_text += "-" * 80 + "\n"
            for idx, result in enumerate(selected_results):
                p = result.params
                comparison_text += f"\nSpectrogram #{idx + 1} ({result.timestamp}):\n"
                comparison_text += f"  Window Type: {p['window_type']}\n"
                
                # Add window-specific parameters with explanations
                if p.get('window_params'):
                    comparison_text += f"  Window Parameters:\n"
                    for key, val in p['window_params'].items():
                        comparison_text += f"    • {key} = {val}\n"
                        # Add explanation for each parameter
                        if key == 'alpha' and p['window_type'] == 'hamming':
                            comparison_text += f"      (Standard Hamming uses α=0.54, Hann uses α=0.5)\n"
                        elif key == 'beta' and p['window_type'] == 'kaiser':
                            if val < 5:
                                comparison_text += f"      (Low β: More frequency resolution, less suppression)\n"
                            elif val < 10:
                                comparison_text += f"      (Medium β: Balanced trade-off)\n"
                            else:
                                comparison_text += f"      (High β: Maximum side-lobe suppression)\n"
                        elif key == 'alpha' and p['window_type'] == 'tukey':
                            comparison_text += f"      (α=0: rectangular, α=1: Hann window)\n"
                
                comparison_text += f"  Window Length: {p['window_length']} samples\n"
                comparison_text += f"  Overlap: {p['overlap']} samples ({p['overlap_pct']}%)\n"
                comparison_text += f"  FFT Size: {p['nfft']}\n"
                
                # Calculate hop size
                hop_size = p['window_length'] - p['overlap']
                comparison_text += f"  Hop Size: {hop_size} samples\n"
            
            # DETAILED RESOLUTION ANALYSIS
            comparison_text += "\n\n" + "=" * 80 + "\n"
            comparison_text += "RESOLUTION ANALYSIS (Critical for Time-Frequency Localization)\n"
            comparison_text += "=" * 80 + "\n\n"
            
            comparison_text += "FREQUENCY RESOLUTION:\n"
            comparison_text += "-" * 80 + "\n"
            comparison_text += "Frequency resolution determines ability to separate close frequencies.\n"
            comparison_text += "Formula: Δf = Sampling_Rate / FFT_Size\n"
            comparison_text += "Better (smaller Δf) = Can distinguish frequencies closer together\n\n"
            
            comparison_text += f"{'Spec':<6} {'Δf (Hz)':<12} {'FFT Size':<12} {'Window':<12} {'Analysis':<40}\n"
            comparison_text += "-" * 80 + "\n"
            
            freq_resolutions = []
            for idx, result in enumerate(selected_results):
                freq_res, _ = result.get_resolution()
                freq_resolutions.append(freq_res)
                p = result.params
                
                # Determine quality based on signal frequencies
                min_freq_sep = min(sp['freq2'] - sp['freq1'], sp['freq3'] - sp['freq2'])
                if freq_res < min_freq_sep / 5:
                    quality = "Excellent - Can clearly separate signal frequencies"
                elif freq_res < min_freq_sep / 2:
                    quality = "Good - Adequate frequency separation"
                else:
                    quality = "Poor - May not separate frequencies well"
                
                comparison_text += f"#{idx+1:<5} {freq_res:<12.4f} {p['nfft']:<12} {p['window_length']:<12} {quality:<40}\n"
            
            comparison_text += f"\nBest Frequency Resolution: Spec #{np.argmin(freq_resolutions) + 1} "
            comparison_text += f"(Δf = {min(freq_resolutions):.4f} Hz)\n"
            comparison_text += f"Worst Frequency Resolution: Spec #{np.argmax(freq_resolutions) + 1} "
            comparison_text += f"(Δf = {max(freq_resolutions):.4f} Hz)\n"
            comparison_text += f"Resolution Range: {max(freq_resolutions)/min(freq_resolutions):.2f}x difference\n"
            
            # Frequency resolution recommendations
            comparison_text += "\nFREQUENCY RESOLUTION INSIGHTS:\n"
            comparison_text += "• Larger FFT size → Better frequency resolution (more frequency bins)\n"
            comparison_text += "• Longer window → Better effective frequency resolution\n"
            comparison_text += f"• To resolve {sp['freq1']:.0f} Hz and {sp['freq2']:.0f} Hz: "
            comparison_text += f"Need Δf < {sp['freq2'] - sp['freq1']:.1f} Hz\n"
            
            # TIME RESOLUTION ANALYSIS
            comparison_text += "\n\nTIME RESOLUTION:\n"
            comparison_text += "-" * 80 + "\n"
            comparison_text += "Time resolution determines ability to localize events in time.\n"
            comparison_text += "Formula: Δt = Hop_Size / Sampling_Rate = (Window_Length - Overlap) / Fs\n"
            comparison_text += "Better (smaller Δt) = Can detect rapid changes more accurately\n\n"
            
            comparison_text += f"{'Spec':<6} {'Δt (s)':<12} {'Hop Size':<12} {'Window':<12} {'Analysis':<40}\n"
            comparison_text += "-" * 80 + "\n"
            
            time_resolutions = []
            for idx, result in enumerate(selected_results):
                _, time_res = result.get_resolution()
                time_resolutions.append(time_res)
                p = result.params
                hop_size = p['window_length'] - p['overlap']
                
                # Determine quality for detecting 1-second transitions
                if time_res < 0.01:  # 10ms
                    quality = "Excellent - Very precise temporal localization"
                elif time_res < 0.05:  # 50ms
                    quality = "Good - Can detect transitions clearly"
                elif time_res < 0.1:  # 100ms
                    quality = "Adequate - Transitions visible but blurred"
                else:
                    quality = "Poor - Transitions may be very blurred"
                
                comparison_text += f"#{idx+1:<5} {time_res:<12.4f} {hop_size:<12} {p['window_length']:<12} {quality:<40}\n"
            
            comparison_text += f"\nBest Time Resolution: Spec #{np.argmin(time_resolutions) + 1} "
            comparison_text += f"(Δt = {min(time_resolutions):.4f} s)\n"
            comparison_text += f"Worst Time Resolution: Spec #{np.argmax(time_resolutions) + 1} "
            comparison_text += f"(Δt = {max(time_resolutions):.4f} s)\n"
            comparison_text += f"Resolution Range: {max(time_resolutions)/min(time_resolutions):.2f}x difference\n"
            
            # Time resolution recommendations
            comparison_text += "\nTIME RESOLUTION INSIGHTS:\n"
            comparison_text += "• Shorter window → Better time resolution (less temporal averaging)\n"
            comparison_text += "• Higher overlap → Smoother but doesn't improve fundamental resolution\n"
            comparison_text += "• For 1-second transitions: Any Δt < 0.1s should be adequate\n"
            
            # UNCERTAINTY PRINCIPLE ANALYSIS
            comparison_text += "\n\nTIME-FREQUENCY UNCERTAINTY PRINCIPLE:\n"
            comparison_text += "-" * 80 + "\n"
            comparison_text += "Heisenberg-Gabor Limit: Δf × Δt ≥ 1/(4π) ≈ 0.0796\n"
            comparison_text += "Smaller product = Better localization (closer to theoretical limit)\n\n"
            
            comparison_text += f"{'Spec':<6} {'Δf (Hz)':<12} {'Δt (s)':<12} {'Product':<12} {'Quality':<30}\n"
            comparison_text += "-" * 80 + "\n"
            
            products = []
            for idx, result in enumerate(selected_results):
                freq_res, time_res = result.get_resolution()
                product = freq_res * time_res
                products.append(product)
                
                # Compare to theoretical limit
                theoretical_limit = 1 / (4 * np.pi)
                ratio = product / theoretical_limit
                
                if ratio < 1.5:
                    quality = "EXCELLENT (Near optimal)"
                elif ratio < 3:
                    quality = "Very Good"
                elif ratio < 5:
                    quality = "Good"
                else:
                    quality = "Moderate"
                
                comparison_text += f"#{idx+1:<5} {freq_res:<12.4f} {time_res:<12.4f} {product:<12.6f} {quality:<30}\n"
            
            comparison_text += f"\nTheoretical Minimum: 0.0796 (Uncertainty Limit)\n"
            comparison_text += f"Best Achieved: Spec #{np.argmin(products) + 1} (Product = {min(products):.6f})\n"
            comparison_text += f"Worst Achieved: Spec #{np.argmax(products) + 1} (Product = {max(products):.6f})\n"
            
            # DETAILED INTENSITY ANALYSIS
            comparison_text += "\n\n" + "=" * 80 + "\n"
            comparison_text += "INTENSITY ANALYSIS (Signal Power and Dynamic Range)\n"
            comparison_text += "=" * 80 + "\n\n"
            
            comparison_text += "POWER SPECTRAL DENSITY:\n"
            comparison_text += "-" * 80 + "\n"
            comparison_text += "Intensity measured in dB/Hz (decibels per Hertz)\n"
            comparison_text += "Higher values = Stronger signal at that frequency\n"
            comparison_text += "Dynamic Range = Max - Min intensity (how much variation)\n\n"
            
            comparison_text += f"{'Spec':<6} {'Max (dB/Hz)':<14} {'Mean (dB/Hz)':<14} {'Min (dB/Hz)':<14} {'Dyn Range':<12}\n"
            comparison_text += "-" * 80 + "\n"
            
            max_intensities = []
            mean_intensities = []
            dynamic_ranges = []
            
            for idx, result in enumerate(selected_results):
                Sxx_dB = 10 * np.log10(result.Sxx + 1e-10)
                max_int = np.max(Sxx_dB)
                mean_int = np.mean(Sxx_dB)
                min_int = np.min(Sxx_dB)
                dyn_range = max_int - min_int
                
                max_intensities.append(max_int)
                mean_intensities.append(mean_int)
                dynamic_ranges.append(dyn_range)
                
                comparison_text += f"#{idx+1:<5} {max_int:<14.2f} {mean_int:<14.2f} {min_int:<14.2f} {dyn_range:<12.2f}\n"
            
            comparison_text += "\nINTENSITY ANALYSIS:\n"
            comparison_text += f"• Highest Peak: Spec #{np.argmax(max_intensities) + 1} ({max(max_intensities):.2f} dB/Hz)\n"
            comparison_text += f"• Highest Average: Spec #{np.argmax(mean_intensities) + 1} ({max(mean_intensities):.2f} dB/Hz)\n"
            comparison_text += f"• Largest Dynamic Range: Spec #{np.argmax(dynamic_ranges) + 1} ({max(dynamic_ranges):.2f} dB)\n"
            
            # Intensity interpretation
            comparison_text += "\nINTENSITY INSIGHTS:\n"
            comparison_text += "• Dynamic range shows contrast between signal and noise floor\n"
            comparison_text += "• Higher peaks = Better signal detection capability\n"
            comparison_text += "• Mean intensity indicates overall energy distribution\n"
            comparison_text += "• Window type affects side-lobe levels (spectral leakage)\n"
            
            # SPECTRAL LEAKAGE ANALYSIS
            comparison_text += "\n\nSPECTRAL LEAKAGE ANALYSIS:\n"
            comparison_text += "-" * 80 + "\n"
            comparison_text += "Spectral leakage causes energy to 'leak' into adjacent frequencies\n"
            comparison_text += "Better windowing reduces leakage but may widen main lobe\n\n"
            
            for idx, result in enumerate(selected_results):
                p = result.params
                window = p['window_type']
                
                comparison_text += f"Spec #{idx+1} - {window.capitalize()} Window:\n"
                
                # Window characteristics
                if window == 'hamming':
                    comparison_text += "  • Side-lobe level: -43 dB (Good general purpose)\n"
                    comparison_text += "  • Main lobe width: 1.3 bins (Moderate)\n"
                elif window == 'hann':
                    comparison_text += "  • Side-lobe level: -31 dB (Moderate leakage)\n"
                    comparison_text += "  • Main lobe width: 1.5 bins (Moderate)\n"
                elif window == 'blackman':
                    comparison_text += "  • Side-lobe level: -58 dB (Excellent suppression)\n"
                    comparison_text += "  • Main lobe width: 1.7 bins (Wider, less precise)\n"
                elif window == 'bartlett':
                    comparison_text += "  • Side-lobe level: -27 dB (High leakage)\n"
                    comparison_text += "  • Main lobe width: 1.3 bins (Moderate)\n"
                elif window == 'kaiser':
                    beta = p.get('window_params', {}).get('beta', 8.6)
                    if beta < 5:
                        comparison_text += f"  • Side-lobe level: ~-30 dB (β={beta}: Moderate)\n"
                    elif beta < 10:
                        comparison_text += f"  • Side-lobe level: ~-50 dB (β={beta}: Good)\n"
                    else:
                        comparison_text += f"  • Side-lobe level: ~-80 dB (β={beta}: Excellent)\n"
                    comparison_text += "  • Main lobe width: Adjustable with β\n"
                elif window == 'boxcar':
                    comparison_text += "  • Side-lobe level: -13 dB (Very high leakage!)\n"
                    comparison_text += "  • Main lobe width: 0.9 bins (Narrow but leaky)\n"
                else:
                    comparison_text += "  • Specialized window with specific characteristics\n"
                
                comparison_text += "\n"
            
            # RECOMMENDATIONS
            comparison_text += "=" * 80 + "\n"
            comparison_text += "RECOMMENDATIONS & USE CASES\n"
            comparison_text += "=" * 80 + "\n\n"
            
            # Best for frequency resolution
            best_freq_idx = np.argmin(freq_resolutions)
            comparison_text += f"[FREQUENCY PRECISION] Use Spectrogram #{best_freq_idx + 1}\n"
            comparison_text += f"  Δf = {freq_resolutions[best_freq_idx]:.4f} Hz\n"
            comparison_text += "  Best for: Separating closely spaced frequencies\n"
            comparison_text += f"  Window: {selected_results[best_freq_idx].params['window_type']}\n"
            comparison_text += f"  FFT Size: {selected_results[best_freq_idx].params['nfft']}\n\n"
            
            # Best for time resolution
            best_time_idx = np.argmin(time_resolutions)
            comparison_text += f"[TIME PRECISION] Use Spectrogram #{best_time_idx + 1}\n"
            comparison_text += f"  Δt = {time_resolutions[best_time_idx]:.4f} s\n"
            comparison_text += "  Best for: Detecting rapid transients and transitions\n"
            comparison_text += f"  Window Length: {selected_results[best_time_idx].params['window_length']} samples\n"
            comparison_text += f"  Overlap: {selected_results[best_time_idx].params['overlap_pct']}%\n\n"
            
            # Best overall balance
            best_product_idx = np.argmin(products)
            comparison_text += f"[BALANCED] Use Spectrogram #{best_product_idx + 1}\n"
            comparison_text += f"  Δf × Δt = {products[best_product_idx]:.6f}\n"
            comparison_text += "  Best for: General purpose time-frequency analysis\n"
            comparison_text += f"  Closest to uncertainty limit (ratio: {products[best_product_idx]/(1/(4*np.pi)):.2f}x)\n\n"
            
            # Best for SNR
            best_snr_idx = np.argmax(max_intensities)
            comparison_text += f"[SIGNAL DETECTION] Use Spectrogram #{best_snr_idx + 1}\n"
            comparison_text += f"  Peak Intensity = {max_intensities[best_snr_idx]:.2f} dB/Hz\n"
            comparison_text += "  Best for: Detecting weak signals in noise\n"
            comparison_text += f"  Window: {selected_results[best_snr_idx].params['window_type']}\n\n"
            
            # Trade-off summary
            comparison_text += "TRADE-OFF SUMMARY:\n"
            comparison_text += "-" * 80 + "\n"
            for idx, result in enumerate(selected_results):
                freq_res, time_res = result.get_resolution()
                product = products[idx]
                
                comparison_text += f"\nSpec #{idx+1} Trade-offs:\n"
                comparison_text += f"  Frequency Resolution: {freq_res:.4f} Hz "
                comparison_text += f"({'Best' if idx == best_freq_idx else 'Worse' if idx == np.argmax(freq_resolutions) else 'Mid'})\n"
                comparison_text += f"  Time Resolution: {time_res:.4f} s "
                comparison_text += f"({'Best' if idx == best_time_idx else 'Worse' if idx == np.argmax(time_resolutions) else 'Mid'})\n"
                comparison_text += f"  Balance: {product:.6f} "
                comparison_text += f"({'Optimal' if idx == best_product_idx else 'Good' if product < 0.002 else 'Moderate'})\n"
            
            comparison_text += "\n" + "=" * 80 + "\n"
            comparison_text += "END OF COMPARATIVE ANALYSIS\n"
            comparison_text += "=" * 80 + "\n"
            
            self.comparison_text.setText(comparison_text)
            
            # Plot side-by-side comparison
            self.plot_comparison(selected_results)
            
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", 
                               f"Error during comparison: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def plot_comparison(self, results):
        """Plot multiple spectrograms side by side with robust error handling"""
        try:
            # Clear figure completely
            self.comparison_canvas.fig.clear()
            n_plots = len(results)
            
            if n_plots == 0:
                QMessageBox.warning(self, "No Data", "No spectrograms to plot")
                return
            
            # Validate data
            for idx, result in enumerate(results):
                if result.Sxx is None or len(result.Sxx) == 0:
                    QMessageBox.warning(self, "Invalid Data", 
                                      f"Spectrogram #{idx+1} contains no data")
                    return
            
            # Find global vmin and vmax for consistent color scale
            all_Sxx_dB = []
            for result in results:
                try:
                    Sxx_dB = 10 * np.log10(result.Sxx + 1e-10)
                    all_Sxx_dB.append(Sxx_dB)
                except Exception as e:
                    QMessageBox.critical(self, "Data Error", 
                                       f"Error processing spectrogram data: {str(e)}")
                    return
            
            # Calculate color scale limits
            try:
                all_values = np.concatenate([s.flatten() for s in all_Sxx_dB])
                vmin = np.percentile(all_values, 5)
                vmax = np.percentile(all_values, 95)
            except Exception as e:
                # Fallback to individual scales
                vmin, vmax = None, None
                print(f"Warning: Could not compute global color scale: {e}")
            
            # Create subplots
            for idx, result in enumerate(results):
                try:
                    ax = self.comparison_canvas.fig.add_subplot(n_plots, 1, idx + 1)
                    
                    # Plot with error handling
                    im = ax.pcolormesh(result.t, result.f, all_Sxx_dB[idx],
                                      shading='gouraud', cmap='viridis',
                                      vmin=vmin, vmax=vmax)
                    
                    ax.set_ylabel('Freq [Hz]', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    if idx == n_plots - 1:
                        ax.set_xlabel('Time [sec]', fontsize=9)
                    else:
                        ax.set_xticklabels([])
                    
                    # Create title with key parameters
                    p = result.params
                    title = f"#{idx+1}: {p['window_type']}"
                    if p.get('window_params'):
                        params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in p['window_params'].items()])
                        title += f"({params_str})"
                    title += f" | W:{p['window_length']}, O:{p['overlap_pct']}%, FFT:{p['nfft']}"
                    
                    ax.set_title(title, fontsize=9, pad=3)
                    
                    # Add colorbar with error handling
                    try:
                        cbar = self.comparison_canvas.fig.colorbar(im, ax=ax, label='dB/Hz')
                        cbar.ax.tick_params(labelsize=8)
                    except Exception as e:
                        print(f"Warning: Could not add colorbar for plot {idx+1}: {e}")
                        
                except Exception as e:
                    QMessageBox.critical(self, "Plot Error", 
                                       f"Error plotting spectrogram #{idx+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
            
            # Final layout adjustment
            try:
                self.comparison_canvas.fig.tight_layout()
                self.comparison_canvas.draw()
            except Exception as e:
                print(f"Warning: Layout adjustment failed: {e}")
                # Try to draw anyway
                try:
                    self.comparison_canvas.draw()
                except:
                    QMessageBox.critical(self, "Render Error", 
                                       "Failed to render comparison plots")
            
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", 
                               f"Unexpected error in plot_comparison: {str(e)}")
            import traceback
            traceback.print_exc()


# CREATIVE ADD-ON 1: Real-time Signal Animator
class SignalAnimatorTab(QWidget):
    """Real-time signal animation showing frequency transitions"""
    def __init__(self):
        super().__init__()
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.current_time = 0
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Real-Time Signal Animation")
        header.setStyleSheet("font-size: 14pt; font-weight: bold; color: #E63946;")
        
        # Control panel
        control_group = QGroupBox("Animation Controls")
        control_layout = QHBoxLayout()
        
        self.btn_play = QPushButton("Play Animation")
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #06A77D;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #048A65;
            }
        """)
        self.btn_play.clicked.connect(self.toggle_animation)
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_animation)
        
        self.speed_label = QLabel("Speed: 50x")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"Speed: {v}x"))
        
        control_layout.addWidget(QLabel("Animation Speed:"))
        control_layout.addWidget(self.speed_slider)
        control_layout.addWidget(self.speed_label)
        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.btn_reset)
        control_group.setLayout(control_layout)
        
        # Canvas
        self.anim_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.anim_toolbar = NavigationToolbar(self.anim_canvas, self)
        
        layout.addWidget(header)
        layout.addWidget(control_group)
        layout.addWidget(self.anim_toolbar)
        layout.addWidget(self.anim_canvas)
        
        self.setLayout(layout)
        
    def set_signal_data(self, t, x, f, t_spec, Sxx):
        """Set signal data for animation"""
        self.t = t
        self.x = x
        self.f = f
        self.t_spec = t_spec
        self.Sxx = Sxx
        self.current_time = 0
        self.reset_animation()
        
    def toggle_animation(self):
        """Toggle animation playback"""
        if not hasattr(self, 't'):
            QMessageBox.warning(self, "No Data", "Generate a spectrogram first!")
            return
            
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("Play Animation")
            self.is_playing = False
        else:
            interval = max(10, int(1000 / self.speed_slider.value()))
            self.timer.start(interval)
            self.btn_play.setText("Pause")
            self.is_playing = True
            
    def reset_animation(self):
        """Reset animation to start"""
        self.current_time = 0
        if hasattr(self, 't'):
            self.update_animation()
            
    def update_animation(self):
        """Update animation frame"""
        if not hasattr(self, 't'):
            return
            
        try:
            # Find current time index
            idx = int(self.current_time * len(self.t) / self.t[-1])
            if idx >= len(self.t):
                idx = 0
                self.current_time = 0
                
            # Clear and redraw
            self.anim_canvas.figure.clear()
            
            # Top: Time domain signal with moving cursor
            ax1 = self.anim_canvas.figure.add_subplot(311)
            ax1.plot(self.t, self.x, 'b-', alpha=0.5, linewidth=1)
            ax1.axvline(self.t[idx], color='r', linewidth=2, label=f'Current: t={self.t[idx]:.3f}s')
            ax1.set_ylabel('Amplitude')
            ax1.set_title('Signal in Time Domain', fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([self.t[0], self.t[-1]])
            
            # Middle: Instantaneous frequency view
            ax2 = self.anim_canvas.figure.add_subplot(312)
            window_size = int(0.1 * len(self.t))  # 100ms window
            start_idx = max(0, idx - window_size)
            end_idx = min(len(self.t), idx + window_size)
            ax2.plot(self.t[start_idx:end_idx], self.x[start_idx:end_idx], 'g-', linewidth=2)
            ax2.axvline(self.t[idx], color='r', linewidth=2, linestyle='--')
            ax2.set_ylabel('Amplitude')
            ax2.set_title(f'Zoomed View (±100ms window)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Bottom: Spectrogram with time marker
            ax3 = self.anim_canvas.figure.add_subplot(313)
            Sxx_dB = 10 * np.log10(self.Sxx + 1e-10)
            im = ax3.pcolormesh(self.t_spec, self.f, Sxx_dB,
                               shading='gouraud', cmap='viridis')
            ax3.axvline(self.t[idx], color='r', linewidth=2, linestyle='--', label='Current time')
            ax3.set_ylabel('Frequency [Hz]')
            ax3.set_xlabel('Time [sec]')
            ax3.set_title('Spectrogram', fontweight='bold')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            self.anim_canvas.figure.colorbar(im, ax=ax3, label='dB/Hz')
            
            self.anim_canvas.figure.tight_layout()
            self.anim_canvas.draw()
            
            # Advance time
            self.current_time += 0.05
            
        except Exception as e:
            print(f"Animation error: {e}")


# CREATIVE ADD-ON 2: 3D Waterfall Display
class WaterfallTab(QWidget):
    """3D Waterfall plot of spectrogram"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("3D Waterfall Visualization")
        header.setStyleSheet("font-size: 14pt; font-weight: bold; color: #457B9D;")
        
        # Controls
        control_group = QGroupBox("3D View Controls")
        control_layout = QFormLayout()
        
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(0, 360)
        self.rotation_slider.setValue(30)
        self.rotation_slider.valueChanged.connect(self.update_plot)
        
        self.elevation_slider = QSlider(Qt.Horizontal)
        self.elevation_slider.setRange(0, 90)
        self.elevation_slider.setValue(30)
        self.elevation_slider.valueChanged.connect(self.update_plot)
        
        self.rotation_label = QLabel("30°")
        self.elevation_label = QLabel("30°")
        
        self.rotation_slider.valueChanged.connect(lambda v: self.rotation_label.setText(f"{v}°"))
        self.elevation_slider.valueChanged.connect(lambda v: self.elevation_label.setText(f"{v}°"))
        
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(self.rotation_slider)
        rotation_layout.addWidget(self.rotation_label)
        
        elevation_layout = QHBoxLayout()
        elevation_layout.addWidget(self.elevation_slider)
        elevation_layout.addWidget(self.elevation_label)
        
        control_layout.addRow("Rotation (Azimuth):", rotation_layout)
        control_layout.addRow("Elevation:", elevation_layout)
        
        control_group.setLayout(control_layout)
        
        # Canvas
        self.waterfall_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.waterfall_toolbar = NavigationToolbar(self.waterfall_canvas, self)
        
        layout.addWidget(header)
        layout.addWidget(control_group)
        layout.addWidget(self.waterfall_toolbar)
        layout.addWidget(self.waterfall_canvas)
        
        self.setLayout(layout)
        
    def set_data(self, f, t, Sxx):
        """Set spectrogram data"""
        self.f = f
        self.t = t
        self.Sxx = Sxx
        self.update_plot()
        
    def update_plot(self):
        """Update 3D waterfall plot"""
        if not hasattr(self, 'f'):
            return
            
        try:
            self.waterfall_canvas.figure.clear()
            ax = self.waterfall_canvas.figure.add_subplot(111, projection='3d')
            
            # Create meshgrid
            T, F = np.meshgrid(self.t, self.f)
            Z = 10 * np.log10(self.Sxx + 1e-10)
            
            # Plot surface
            surf = ax.plot_surface(T, F, Z, cmap='viridis', 
                                  linewidth=0, antialiased=True, alpha=0.9)
            
            ax.set_xlabel('Time [s]', fontweight='bold')
            ax.set_ylabel('Frequency [Hz]', fontweight='bold')
            ax.set_zlabel('Power [dB/Hz]', fontweight='bold')
            ax.set_title('3D Waterfall Plot', fontweight='bold', fontsize=12)
            
            # Set view angle
            ax.view_init(elev=self.elevation_slider.value(), 
                        azim=self.rotation_slider.value())
            
            self.waterfall_canvas.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='dB/Hz')
            self.waterfall_canvas.figure.tight_layout()
            self.waterfall_canvas.draw()
            
        except Exception as e:
            print(f"3D plot error: {e}")


# CREATIVE ADD-ON 3: Audio Export and Playback
class AudioExportTab(QWidget):
    """Export signal as audio and play it back"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("Audio Export & Analysis")
        header.setStyleSheet("font-size: 14pt; font-weight: bold; color: #F77F00;")
        
        # Info
        info_group = QGroupBox("About Audio Export")
        info_layout = QVBoxLayout()
        info_label = QLabel(
            "Export the generated non-stationary signal as a WAV audio file.\n"
            "You'll be able to HEAR the frequency transitions from 100Hz → 300Hz → 600Hz!"
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        
        # Export controls
        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout()
        
        self.amplification = QDoubleSpinBox()
        self.amplification.setRange(0.01, 1.0)
        self.amplification.setValue(0.3)
        self.amplification.setSingleStep(0.05)
        self.amplification.setToolTip("Volume level (0.3 recommended for safety)")
        
        self.btn_export = QPushButton("Export as WAV File")
        self.btn_export.clicked.connect(self.export_audio)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #F77F00;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #DC6600;
            }
        """)
        
        export_layout.addRow("Amplification (Volume):", self.amplification)
        export_layout.addRow("", self.btn_export)
        export_group.setLayout(export_layout)
        
        # Signal properties display
        props_group = QGroupBox("Signal Properties")
        props_layout = QVBoxLayout()
        
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(2)
        self.properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties_table.horizontalHeader().setStretchLastSection(True)
        
        props_layout.addWidget(self.properties_table)
        props_group.setLayout(props_layout)
        
        layout.addWidget(header)
        layout.addWidget(info_group)
        layout.addWidget(export_group)
        layout.addWidget(props_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def set_signal_data(self, t, x, fs):
        """Set signal data"""
        self.t = t
        self.x = x
        self.fs = fs
        
        # Update properties table
        properties = [
            ("Duration", f"{t[-1]:.3f} seconds"),
            ("Sampling Rate", f"{fs} Hz"),
            ("Total Samples", str(len(x))),
            ("Max Amplitude", f"{np.max(np.abs(x)):.4f}"),
            ("Min Amplitude", f"{np.min(x):.4f}"),
            ("RMS", f"{np.sqrt(np.mean(x**2)):.4f}"),
            ("Peak-to-Peak", f"{np.ptp(x):.4f}"),
            ("File Size (est.)", f"{len(x)*2/1024:.1f} KB")
        ]
        
        self.properties_table.setRowCount(len(properties))
        for i, (prop, value) in enumerate(properties):
            self.properties_table.setItem(i, 0, QTableWidgetItem(prop))
            self.properties_table.setItem(i, 1, QTableWidgetItem(value))
            
    def export_audio(self):
        """Export signal as WAV file"""
        if not hasattr(self, 'x'):
            QMessageBox.warning(self, "No Data", "Generate a signal first!")
            return
        
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Audio File", 
                f"spectrogram_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                "WAV Files (*.wav)"
            )
            
            if filename:
                # Normalize and amplify
                x_normalized = self.x / np.max(np.abs(self.x))
                x_amplified = x_normalized * self.amplification.value()
                
                # Clipping protection
                x_amplified = np.clip(x_amplified, -1.0, 1.0)
                
                # Convert to 16-bit PCM
                x_int16 = np.int16(x_amplified * 32767)
                
                # Write WAV file
                wavfile.write(filename, self.fs, x_int16)
                
                QMessageBox.information(self, "Export Successful", 
                                      f"Audio exported successfully!\n\n"
                                      f"File: {filename}\n"
                                      f"Duration: {self.t[-1]:.2f}s\n"
                                      f"Sample Rate: {self.fs}Hz\n"
                                      f"Bit Depth: 16-bit PCM\n\n"
                                      f"You can now play this file to hear the frequency transitions!")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export audio: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window with improved error handling"""
    def __init__(self):
        super().__init__()
        self.current_result = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Advanced Spectrogram Analyzer v2.0")
        self.setGeometry(100, 100, 1500, 950)
        
        # Set application icon (if available)
        # self.setWindowIcon(QIcon('icon.png'))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title = QLabel("Spectrogram Analyzer")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #1D3557;")
        left_layout.addWidget(title)
        
        # Signal parameters
        signal_group = QGroupBox("Signal Configuration")
        signal_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        signal_layout = QVBoxLayout()
        self.signal_params = SignalGeneratorWidget()
        signal_layout.addWidget(self.signal_params)
        signal_group.setLayout(signal_layout)
        
        # Spectrogram parameters
        spec_group = QGroupBox("Spectrogram Configuration")
        spec_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        spec_layout = QVBoxLayout()
        self.spec_params = SpectrogramParametersWidget()
        spec_layout.addWidget(self.spec_params)
        spec_group.setLayout(spec_layout)
        
        # Generate button
        self.btn_generate = QPushButton("Generate Spectrogram")
        self.btn_generate.clicked.connect(self.generate_spectrogram)
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #06A77D;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #048A65;
            }
            QPushButton:pressed {
                background-color: #036B4E;
            }
        """)
        
        # Add to comparison button
        self.btn_add_comparison = QPushButton("Add to Comparative Analysis")
        self.btn_add_comparison.clicked.connect(self.add_to_comparison)
        self.btn_add_comparison.setEnabled(False)
        self.btn_add_comparison.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1F5F7A;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        
        # Scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(signal_group)
        scroll_layout.addWidget(spec_group)
        scroll_layout.addWidget(self.btn_generate)
        scroll_layout.addWidget(self.btn_add_comparison)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        
        left_layout.addWidget(scroll)
        left_panel.setMaximumWidth(420)
        left_panel.setMinimumWidth(380)
        
        # Right panel - Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2E86AB;
                color: white;
                font-weight: bold;
            }
        """)
        
        # Main spectrogram tab
        main_tab = QWidget()
        main_layout_tab = QVBoxLayout(main_tab)
        
        self.main_canvas = SpectrogramCanvas(self, width=10, height=6)
        self.main_toolbar = NavigationToolbar(self.main_canvas, self)
        
        main_layout_tab.addWidget(self.main_toolbar)
        main_layout_tab.addWidget(self.main_canvas)
        
        # Results info
        info_label = QLabel("Analysis Information:")
        info_label.setStyleSheet("font-weight: bold;")
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(180)
        font = QFont("Courier", 9)
        self.info_text.setFont(font)
        
        main_layout_tab.addWidget(info_label)
        main_layout_tab.addWidget(self.info_text)
        
        self.tabs.addTab(main_tab, "Main Spectrogram")
        
        # Comparative analysis tab
        self.comparison_tab = ComparativeAnalysisTab()
        self.tabs.addTab(self.comparison_tab, "Comparative Analysis")
        
        # Creative add-ons
        self.animator_tab = SignalAnimatorTab()
        self.tabs.addTab(self.animator_tab, "Signal Animator")
        
        self.waterfall_tab = WaterfallTab()
        self.tabs.addTab(self.waterfall_tab, "3D Waterfall")
        
        self.audio_tab = AudioExportTab()
        self.tabs.addTab(self.audio_tab, "Audio Export")
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready | Load parameters and click 'Generate Spectrogram'")
        
    def generate_signal(self):
        """Generate the piecewise non-stationary signal"""
        params = self.signal_params.get_params()
        fs = params['fs']
        duration = params['duration']
        freq1 = params['freq1']
        freq2 = params['freq2']
        freq3 = params['freq3']
        
        # Time vector
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Generate piecewise signal
        x = np.zeros_like(t)
        segment_duration = duration / 3
        
        # Segment 1
        mask1 = t < segment_duration
        x[mask1] = np.sin(2 * np.pi * freq1 * t[mask1])
        
        # Segment 2
        mask2 = (t >= segment_duration) & (t < 2 * segment_duration)
        x[mask2] = np.sin(2 * np.pi * freq2 * t[mask2])
        
        # Segment 3
        mask3 = t >= 2 * segment_duration
        x[mask3] = np.sin(2 * np.pi * freq3 * t[mask3])
        
        return t, x, fs, params
        
    def generate_spectrogram(self):
        """Generate and display spectrogram with error handling"""
        try:
            self.statusBar().showMessage("Generating spectrogram...")
            QApplication.processEvents()  # Update UI
            
            # Generate signal
            t, x, fs, signal_params = self.generate_signal()
            
            # Get spectrogram parameters
            spec_params = self.spec_params.get_params()
            window = self.spec_params.get_window()
            nperseg = spec_params['window_length']
            noverlap = spec_params['overlap']
            nfft = spec_params['nfft']
            
            # Validate parameters
            if noverlap >= nperseg:
                QMessageBox.warning(self, "Invalid Parameters", 
                                  "Overlap must be less than window length!")
                self.statusBar().showMessage("Error: Invalid parameters")
                return
            
            # Compute spectrogram
            f, t_spec, Sxx = signal.spectrogram(x, fs, window=window, 
                                               nperseg=nperseg, 
                                               noverlap=noverlap, 
                                               nfft=nfft)
            
            # Create result object
            self.current_result = SpectrogramResult(f, t_spec, Sxx, spec_params, signal_params)
            
            # Plot
            window_info = spec_params['window_type']
            if spec_params.get('window_params'):
                params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in spec_params['window_params'].items()])
                window_info += f" ({params_str})"
            
            title = f"Spectrogram - {window_info}"
            self.main_canvas.plot_spectrogram(f, t_spec, Sxx, title)
            
            # Update info
            freq_res, time_res = self.current_result.get_resolution()
            info_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ SPECTROGRAM ANALYSIS - Generated at: {self.current_result.timestamp}      ║
╚══════════════════════════════════════════════════════════════════════════════╝

SIGNAL PARAMETERS:
  • Sampling Frequency: {fs} Hz
  • Duration: {t[-1]:.3f} seconds
  • Segment Frequencies: {signal_params['freq1']:.1f} Hz, {signal_params['freq2']:.1f} Hz, {signal_params['freq3']:.1f} Hz
  • Total Samples: {len(x)}

WINDOW CONFIGURATION:
  • Window Type: {spec_params['window_type']}"""
            
            if spec_params.get('window_params'):
                info_text += "\n  • Window Parameters: " + ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                                        for k, v in spec_params['window_params'].items()])
            
            info_text += f"""
  • Window Length: {spec_params['window_length']} samples
  • Overlap: {spec_params['overlap']} samples ({spec_params['overlap_pct']}%)
  • FFT Size: {spec_params['nfft']} points

RESOLUTION METRICS:
  • Frequency Resolution: {freq_res:.6f} Hz
  • Time Resolution: {time_res:.6f} seconds
  • Time-Frequency Product: {freq_res * time_res:.6f}
  • Uncertainty Limit: {1/(4*np.pi):.6f} (theoretical minimum)

INTENSITY ANALYSIS:
  • Maximum Intensity: {self.current_result.get_max_intensity():.2f} dB/Hz
  • Mean Intensity: {self.current_result.get_mean_intensity():.2f} dB/Hz
  • Dynamic Range: {self.current_result.get_max_intensity() - np.min(10*np.log10(Sxx+1e-10)):.2f} dB

QUALITY ASSESSMENT:
"""
            
            # Add quality assessment
            product = freq_res * time_res
            if product < 0.0005:
                info_text += "  [EXCELLENT] Excellent time-frequency localization (near theoretical limit)\n"
            elif product < 0.001:
                info_text += "  [VERY GOOD] Very good time-frequency localization\n"
            elif product < 0.002:
                info_text += "  [GOOD] Good time-frequency localization\n"
            else:
                info_text += "  [MODERATE] Moderate time-frequency localization (consider adjusting parameters)\n"
            
            # Frequency resolution check
            min_freq = min(signal_params['freq1'], signal_params['freq2'], signal_params['freq3'])
            if freq_res < min_freq / 10:
                info_text += "  [OK] Frequency resolution sufficient for signal separation\n"
            else:
                info_text += "  [WARNING] Consider longer window or larger FFT for better frequency resolution\n"
            
            # Time resolution check
            if time_res < 0.05:  # 50ms
                info_text += "  [OK] Time resolution good for transition detection\n"
            else:
                info_text += "  [WARNING] Consider shorter window for better time resolution\n"
            
            info_text += "\n" + "─" * 80
            
            self.info_text.setText(info_text)
            
            # Enable comparison button
            self.btn_add_comparison.setEnabled(True)
            
            # Update creative add-ons
            self.animator_tab.set_signal_data(t, x, f, t_spec, Sxx)
            self.waterfall_tab.set_data(f, t_spec, Sxx)
            self.audio_tab.set_signal_data(t, x, fs)
            
            self.statusBar().showMessage("Spectrogram generated successfully!", 5000)
            
            QMessageBox.information(self, "Success", 
                                  "Spectrogram generated successfully!\n\n"
                                  "Explore the tabs:\n"
                                  "• Signal Animator - See real-time evolution\n"
                                  "• 3D Waterfall - Interactive 3D visualization\n"
                                  "• Audio Export - Hear the frequency transitions\n"
                                  "• Comparative Analysis - Compare multiple configurations")
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", 
                               f"Error generating spectrogram:\n{str(e)}")
            self.statusBar().showMessage("Error occurred during generation")
            import traceback
            traceback.print_exc()
        
    def add_to_comparison(self):
        """Add current result to comparative analysis"""
        if self.current_result is None:
            QMessageBox.warning(self, "No Result", 
                              "Generate a spectrogram first!")
            return
        
        try:
            self.comparison_tab.add_result(self.current_result)
            self.tabs.setCurrentWidget(self.comparison_tab)
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to add to comparison: {str(e)}")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application metadata
    app.setApplicationName("Advanced Spectrogram Analyzer")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Signal Processing Lab")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
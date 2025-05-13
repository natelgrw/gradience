"""
peak_decoder.py

A Python script that utilizes packages like MSConvert, XCMS, PeakPantheR,
MZMine3, and MOCCA2 to create peak decoder objects aiding in the extraction of
peak locations and peak widths in LC-MS ultraviolet and mass spectra.

Each peak decoder object contains essential functions to process LC-MS experiment 
files and return a list of peaks represented as tuples of length 2. The first and
second elements represent the starting and ending x coordinates of the peak.

Additionally, other supplemental functions exist to extract chromatogram wavelength
and absorbance data using MOCCA2.

Author: natelgrw
Date Created: 4/22/2025
Date Last Edited: 5/8/2025
"""
import pandas as pd
from mocca2.analysis import load_mzml, detect_peaks

class MoccaPeakDecoder:
    def __init__(self, mzml_path: str, wavelength: int = 214):
        """
        Initializes a peak decoder object with a .mzML file path and wavelength.
        """
        self.mzml_path = mzml_path
        self.wavelength = wavelength
        self.dataset = load_mzml(mzml_path)

        self.time = self.dataset.uv_signal["time"]
        absorbance_col = f"absorbance_{wavelength}"
        if absorbance_col not in self.dataset.uv_signal.columns:
            raise ValueError(f"Wavelength of {wavelength} nm not present in UV data")
        
        self.signal = self.dataset.uv_signal[absorbance_col]
        self.peaks = detect_peaks(self.time, self.signal)

    def get_uv_trace(self):
        """
        Returns a full UV chromatogram trace as a dictionary of time and absorbance arrays.
        """
        return {
            "time": self.time.values,
            "absorbance": self.signal.values
        }
    
    def get_uv_peaks(self):
        """
        Returns a list of detected peaks as (start_time, end_time) tuples.
        """
        return [(pk["start_time"], pk["end_time"]) for pk in self.peaks]
    
    def get_peak_widths(self):
        """
        Returns a list of peak widths in the given chromatogram.
        """
        return [pk["end_time"] - pk["start_time"] for pk in self.peaks]
    
    def get_peak_areas(self):
        """
        Returns a list of peak areas from MOCCA2's internal peak integration function.
        """
        return [pk["area"] for pk in self.peaks]
    
    def get_peak_maxima(self):
        """
        Returns a list of max intensities for each peak.
        """
        return [pk["max_intensity"] for pk in self.peaks]
    
    def get_peak_midpoints(self):
        """
        Returns a list of peak midpoints for the chromatogram.
        """
        peak_midpoints = []
        uv_peaks = self.get_uv_peaks()
        for pk in uv_peaks:
            midpoint = (pk[1] + pk[2]) / 2
            peak_midpoints.append(midpoint)
        return peak_midpoints
    
    def get_distances(self):
        """
        Returns a list of distances between peaks in the chromatogram.
        An empty list is returned if there are less than 2 peaks.
        """
        distances = []
        peak_locs = self.get_peak_midpoints()
        if len(peak_locs) < 2:
            return []
        for i in range(len(peak_locs) - 1):
            dist = peak_locs[i + 1] - peak_locs[i]
            distances.append(dist)
        return distances
    
    def get_min_distance(self):
        """
        Returns the minimum distance between 2 peaks in the chromatogram.
        False is returned if there are less than 2 peaks.
        """
        distances = self.get_distances()
        return min(distances) if len(distances) > 0 else False
    
    def get_max_distance(self):
        """
        Returns the maximum distance between 2 peaks in the chromatogram.
        False is returned if there are less than 2 peaks.
        """
        distances = self.get_distances()
        return max(distances) if len(distances) > 0 else False





"""
peak_decoder.py

A Python script that utilizes MOCCA2 to create peak decoder objects aiding in the extraction of
peak locations and peak widths in LC-MS ultraviolet and mass spectra. This script is compatible
with raw data files from Agilent, Schimadzu, and Waters machines.

Each peak decoder object contains essential functions to process LC-MS experiment 
files and return a list of peaks represented as tuples of length 2. The first and
second elements represent the starting and ending x coordinates of the peak.

Additionally, other supplemental functions exist to extract chromatogram wavelength
and absorbance data using MOCCA2.

Author: natelgrw
Created: 4/22/2025
Last Edited: 5/8/2025
"""

#TODO: implement maxima and relative maxima calculations
#TODO: implement lambda max data for peaks
#TODO: implement plotting functions for 1D, 2D, and lambda max representations

from typing import Literal, List, Tuple
from mocca2 import Chromatogram
from mocca2.deconvolution.peak_models import PeakModel
import numpy as np

class MoccaPeakDecoder:
    def __init__(self, 
                 file_path: str, 
                 rxn_type: str,
                 reactants: List[str],
                 solvents: List[str],
                 wavelength: Tuple[int, int] | None = None,
                 method: Literal['asls', 'arpls', 'flatfit'] = "flatfit",
                 time: tuple[int | None, int | None] = None):
        """
        Initializes a MoccaPeakDecoder object and accompanying chromatogram
        with a specified raw file data path, optional wavelength specifications, 
        baseline corrections, and time specifications. Also includes key reaction
        data on solvents and reactants.
        """
        self.file_path = file_path
        self.rxn_type = rxn_type
        self.reactants = reactants
        self.solvents = solvents
        self.wavelength = wavelength
        self.method = method
        self.time = time
        self.chromatogram = Chromatogram(sample=file_path)

        if self.wavelength != None:
            self.chromatogram.extract_wavelength(self.wavelength[0], self.wavelength[1])

        if self.time != None:
            self.chromatogram.extract_time(self.time[0], self.time[1], inplace=True)

        self.chromatogram.correct_baseline(self.method)

    def get_peaks(self,
                  deconvolve_algo: PeakModel | Literal['BiGaussian', 'BiGaussianTrailing', 'FraserSuzuki', 'Bemg'],
                  min_deconvolve_r2: float,
                  concentration_relax: bool,
                  max_num_peaks: int,
                  contraction_algo: Literal['mean', 'max', 'weighted_mean'] = 'mean',
                  min_h: float = 10.0,
                  min_time: float | None = None,
                  max_time: float | None = None,
                  ):
        """
        Calls find_peaks() and deconvolve_peaks() on the Chromatogram object
        derived from the input file path, returning a list of Peak objects stored 
        within the Chromatogram.
        """
        self.peak_params = {
        "deconvolution_model": str(deconvolve_algo),
        "min_r2": min_deconvolve_r2,
        "relax_concentration_constraints": concentration_relax,
        "max_components": max_num_peaks,
        "contraction_algorithm": contraction_algo,
        "min_height": min_h,
        "min_elution_time": min_time,
        "max_elution_time": max_time
        }

        self.chromatogram.find_peaks(contraction=contraction_algo, min_height=min_h, 
                                     min_elution_time=min_time, max_elution_time=max_time)
        self.chromatogram.deconvolve_peaks(model=deconvolve_algo, min_r2=min_deconvolve_r2, 
                                           relaxe_concs=concentration_relax, max_comps=max_num_peaks)

        return self.chromatogram.peaks

    def get_peak_times(self):
        """
        Iterates through all DeconvolvedPeak objects on the Chromatogram object, and
        returns a list of tuples with index 2 Each tuple representing a component peak,
        with the first index representing the start and the second representing the end
        of the peak.

        All subpeak components of a DeconvolvedPeak object are represented as tuples. If
        a DeconvolvedPeak object is marked as unresolved, subpeaks cannot be accurately
        modeled, and the whole DeconvolvedPeak is just treated as a singular tuple.
        """
        time_ranges = []
        time_axis = self.chromatogram.time

        for peak in self.chromatogram.peaks:
            # case 1: deconvolved peak can be resolved
            if peak.resolved and peak.components:
                for component in peak.components:
                    conc = component.concentration
                    tup_peak_indices = (conc > 1e-6).nonzero()[0]

                    if tup_peak_indices.size == 0:
                        continue

                    start_idx = peak.left + tup_peak_indices[0]
                    end_idx = peak.left + tup_peak_indices[-1]

                    time_ranges.append((time_axis[start_idx], time_axis[end_idx]))
            # case 2: deconvolved peak cannot be resolved
            else:
                start_idx = peak.left
                end_idx = peak.right

                time_ranges.append((time_axis[start_idx], time_axis[end_idx]))

        return time_ranges

    def get_peak_areas(self):
        """
        Iterates through all DeconvolvedPeak objects on the Chromatogram object, and
        returns a list of float values. Each float represents the integral
        underneath each component peak in a Chromatogram.

        If a DeconvolvedPeak is unresolvable, the entire DeconvolvedPeak is treated
        as a singular peak, and the integral is estimated manually to a large degree
        of accuracy.
        """
        areas = []
        time_axis = self.chromatogram.time
        data = self.chromatogram.data

        for peak in self.chromatogram.peaks:
            # case 1: deconvolved peak can be resolved
            if peak.resolved and peak.components:
                for component in peak.components:
                    areas.append(component.integral)
            # case 2: deconvolved peak cannot be resolved
            else:
                peak_data = data[:, peak.left : peak.right + 1]
                signal = peak_data.sum(axis=0)

                # approximate time increment
                dt = time_axis[1] - time_axis[0]

                area = np.trapezoid(signal, dx=dt)
                areas.append(area)

        return areas
    
    def get_maxima(self):
        pass

    def get_lambda_max(self):
        pass

    def get_min_peak_distance(self):
        """
        Returns the minimum distance between two peaks in a Chromatogram.
        """
        peak_distances = []
        peak_data = self.get_peak_times()

        for i in range(len(peak_data) - 1):
            first = peak_data[i][1]
            second = peak_data[i+1][0]
            difference = second - first
            peak_distances.append(difference)

        return min(peak_distances)

    def get_summary(self):
        """
        Returns a comprehensive dictionary representative of data in a
        Chromatogram. Includes information about peak widths, peak areas,
        peak maxima, and a templates for compound identification.
        """
        times = self.get_peak_times()
        areas = self.get_peak_areas()
        min_peak_distance = self.get_min_peak_distance()

        return {
            "reaction": self.rxn_type,
            "reactants": self.reactants,
            "solvents": self.solvents,
            "lambda_range": self.wavelength,
            "baseline_method": self.method,
            "time_range": self.time,
            "peak_picking": self.peak_params,
            "peaks": [

            ],
            "min_distance": min_peak_distance
        }
    
    def plot_chromatogram_1d(self):
        pass

    def plot_chromatogram_2d(self):
        pass

    def plot_lambda_absorption(self):
        pass

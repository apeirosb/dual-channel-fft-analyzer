#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dual_channel_fft_analyzer.py

Dual Channel FFT Analyzer

References:
    [1] Herlufsen, H. (1984) Dual Channel FFT Analysis (Part I). Brüel & Kjær Technical
        Review No. 1-1984
    [2] Herlufsen, H. (1984) Dual Channel FFT Analysis (Part II). Brüel & Kjær
        Technical Review No. 2-1984
    [3] Heinzel, G., Rüdiger, A., & Schilling, R. (2002). Spectrum and spectral density
        estimation by the Discrete Fourier transform (DFT), including a comprehensive
        list of window functions and some new at-top windows.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, rfft
from scipy.io.wavfile import read
from scipy.signal import windows


class FFTAnalyzer:
    def __init__(self, config) -> None:
        """FFTAnalyzer class initialization

        Args:
            config (dict): FFT Analyzer settings
        """
        # parse FFT settings
        self.verbose = config["verbose"]
        self.nfft = config["nfft"]
        self.overlap = config["overlap"]
        self.win = windows.get_window(config["window"], config["nfft"], fftbins=True)
        self.spectrumtype = config["spectrumtype"]
        self.delay = config["delay"]
        # placeholder for audio signals and output
        self.fs = None
        self.input_a = None
        self.input_b = None
        self.output = None

    def __str__(self) -> str:
        """Prints summary of the analyzer settings

        Returns:
            str: analyzer settings
        """
        info = (
            "\n---------- DUAL-CHANNEL FFT ANALYZER -----------\n"
            f"Window length: \t\t\t{self.nfft} samples\n"
            f"Overlap: \t\t\t{self.overlap}%\n"
            f"FFT length: \t\t\t{self.nfft} samples\n"
            f"Spectrum type: \t\t\t{self.spectrumtype}\n"
            f"Delay: \t\t\t\t{self.delay} s\n"
            "------------------------------------------------\n"
        )
        return info

    def run(self, input_a: str, input_b: str) -> int:
        """Run analyzer

        Args:
            input_a (str): path to the audio file corresponding to input A
            input_b (str): path to the audio file corresponding to input B

        Returns:
            int: 0 if OK
        """
        # read audio signals
        self._read_input(input_a, input_b)

        # dual-channel FFT analysis
        result = self._dual_channel_fft()

        # analysis info
        if self.verbose:
            info = (
                "\n-------------- FREQUENCY ANALYSIS --------------\n"
                f"Window length:\t\t\t{self.nfft} samples\n"
                f"Overlap:\t\t\t{self.overlap}%\n"
                f"FFT length:\t\t\t{self.nfft} samples\n"
                f"Number of time segments:\t{result['info']['num_time_segments']}\n"
                f"Time res. (analysis):\t\t{result['info']['time_res_ana']:.3f} s\n"
                f"Time res. (grid):\t\t{result['info']['time_res_grid']:.3f} s\n"
                f"Freq. res. (analysis):\t\t{result['info']['freq_res_ana']:.2f} Hz\n"
                f"Freq. res. (grid):\t\t{result['info']['freq_res_grid']:.2f} Hz\n"
                "------------------------------------------------\n"
            )
            print(info)

        return 0

    def _read_input(self, input_a: str, input_b: str) -> int:
        """Reads input audio (.wav) files

        Args:
            input_a (str): path to the audio file corresponding to input A
            input_b (str): path to the audio file corresponding to input B

        Raises:
            ValueError: if sample rate does not match for both inputs

        Returns:
            int: 0 if OK
        """
        # read audio signals
        self.fs, self.input_a = read(input_a)
        fs, self.input_b = read(input_b)
        # check that sampling frequency matches for both signals
        if self.fs != fs:
            raise ValueError("the sample rate of the two inputs does not match")
        # signal conditioning
        self.input_a = self._signal_conditioning(self.input_a)
        self.input_b = self._signal_conditioning(self.input_b)
        return 0

    @staticmethod
    def _signal_conditioning(signal: np.ndarray) -> np.ndarray:
        """Input signal conditioning (scaling, etc.)

        Args:
            signal (np.ndarray): audio signal

        Raises:
            ValueError: if signal has more than one channel

        Returns:
            np.ndarray: conditioned audio signal
        """
        # force conversion into an array instead of a vector
        nchannels = signal.shape[1] if len(signal.shape) > 1 else 1
        if nchannels > 1:
            raise ValueError("input signals must be single channel")
        signal = signal.reshape(-1, nchannels)
        # do scaling (between -1 and 1, to float64)
        scaling = {"uint8": 2**7, "int16": 2**15, "int32": 2**31, "float32": 1}
        signal_type = signal.dtype.name
        signal = (
            signal / scaling[signal_type] - 1
            if signal_type == "uint8"
            else signal / scaling[signal_type]
        )
        # force type
        signal = signal.astype(np.float64)

        return signal

    def _dual_channel_fft(self) -> dict:
        """Dual Channel FFT Analysis

        Returns:
            dict: dict containing all the output values
        """
        # compensate delay for channel B
        if self.delay > 0:
            delay_samples = round(self.delay * self.fs)
            self.input_a = self.input_a[delay_samples:, :]

        # truncate signals to the same length
        signal_length = min(self.input_a.size, self.input_b.size)
        self.input_a = self.input_a[:signal_length, :]
        self.input_b = self.input_b[:signal_length, :]

        # analysis: instant autospectra and cross-spectra ##############################
        spectra = {
            x
            + y: self._crossspectrum(
                getattr(self, f"input_{x}"),
                getattr(self, f"input_{y}"),
                self.nfft,
                self.overlap,
                self.win,
                self.fs,
            )
            for x in ["a", "b"]
            for y in ["a", "b"]
        }

        # averaging ####################################################################
        g = {
            x + y: spectra[x + y]["spectrum"].mean(axis=1)
            for x in ["a", "b"]
            for y in ["a", "b"]
        }

        # post-processing ##############################################################

        # coherence and coherent/non-coherent output power
        coherence = (np.power(np.abs(g["ab"]), 2) / (g["aa"] * g["bb"])).real.astype(
            np.float64
        )
        coherent_output_power = (coherence * g["bb"]).real.astype(np.float64)
        non_coherent_output_power = ((1 - coherence) * g["bb"]).real.astype(np.float64)

        # signal-to-noise ratio (SNR)
        snr = coherence / (1 - coherence)

        # cross-correlation (Rab)
        r_ab = irfft(self._one_sided_2_two_sided_correction(g["ab"], self.nfft))

        # autocorrelation (Raa, Rbb)
        r_aa = irfft(self._one_sided_2_two_sided_correction(g["aa"], self.nfft))
        r_bb = irfft(self._one_sided_2_two_sided_correction(g["bb"], self.nfft))
        lags = np.arange(0, self.nfft)

        # frequeny responses
        h1_freq_resp = g["ab"] / g["aa"]
        h2_freq_resp = g["bb"] / g["ab"].conj()

        # impulse responses
        h1 = irfft(h1_freq_resp)
        h2 = irfft(h2_freq_resp)

        # format output
        self.output = {
            "autospectrum_a": self._scale_spectrum(g["aa"]).real.astype(np.float64),
            "autospectrum_b": self._scale_spectrum(g["bb"]).real.astype(np.float64),
            "cross_spectrum": self._scale_spectrum(g["ab"]),
            "freq_axis": spectra["ab"]["freq_axis"],
            "coherence": coherence,
            "coherent_output_power": self._scale_spectrum(coherent_output_power),
            "non_coherent_output_power": self._scale_spectrum(
                non_coherent_output_power
            ),
            "snr": snr,
            "autocorrelation_a": r_aa,
            "autocorrelation_b": r_bb,
            "cross_correlation": r_ab,
            "lags": lags,
            "freq_resp_h1": h1_freq_resp,
            "freq_resp_h2": h2_freq_resp,
            "impulse_resp_h1": h1,
            "impulse_resp_h2": h2,
            "info": spectra["ab"]["info"],
        }

        return self.output

    @staticmethod
    def _crossspectrum(
        x: np.ndarray,
        y: np.ndarray,
        nfft: int,
        overlap: float,
        win: np.ndarray,
        fs: float,
    ) -> dict:
        """Compute the cross-spectrum of two time-domain signals

        Args:
            x (np.ndarray): first input signal
            y (np.ndarray): second input signal
            nfft (int): FFT length
            overlap (float): overlapping between time segments
            win (np.ndarray): anaysis window
            fs (float): sampling frequency

        Returns:
            dict: cross-spectrum, time axis, freq. axis and some metadata
        """
        # we ensure we take only the first channel
        x = x[:, 0]
        y = y[:, 0]

        # some preliminary computations
        wlen = win.size
        nfft = nfft if nfft > wlen else wlen
        hop = int(np.round(wlen * (1 - overlap / 100)))
        signal_length = min(x.size, y.size)
        duration = signal_length / fs
        num_time_segments = int(1 + np.fix((signal_length - wlen) / hop))
        num_unique_bins = int(np.ceil((nfft + 1) / 2))

        # allocation of STFT matrix
        stft = np.zeros((num_unique_bins, num_time_segments), dtype=np.complex64)

        idx = 0
        for k in range(num_time_segments):
            # apply window to the segment
            xw = np.multiply(x[idx : idx + wlen], win)
            yw = np.multiply(y[idx : idx + wlen], win)
            # fft (rfft gives the single-sided FFT)
            x_fft = rfft(xw, nfft)
            y_fft = rfft(yw, nfft)
            spectrum = np.multiply(x_fft.conj(), y_fft)
            # correction for one-sided spectrum to distribute power at 'positive'
            # frequencies only
            if np.remainder(nfft, 2):
                # odd Nfft, Nyquist point not included
                spectrum[1:] = spectrum[1:] * 2
            else:
                # even Nfft, Nyquist point included
                spectrum[1:-1] = spectrum[1:-1] * 2
            # update STFT matrix
            stft[:, k] = spectrum
            # update index
            idx += hop

        # time and frequency axis
        time_axis = np.arange(wlen / 2, wlen / 2 + num_time_segments * hop, hop) / fs
        freq_axis = np.arange(0, num_unique_bins) * fs / nfft

        # some metadata/info
        metadata = {
            "time_res_ana": wlen / fs,
            "freq_res_ana": fs / wlen,
            "time_res_grid": duration / num_time_segments,
            "freq_res_grid": fs / nfft,
            "window_overlap_pctg": 100 * (wlen - hop) / wlen,
            "num_time_segments": num_time_segments,
            "nfft": nfft,
        }

        return {
            "spectrum": stft,
            "time_axis": time_axis,
            "freq_axis": freq_axis,
            "info": metadata,
        }

    def _scale_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Scale the spectrum for 'power' or 'psd'

        Args:
            spectrum (np.ndarray): input spectrum

        Raises:
            ValueError: if spectrumtype value is wrong

        Returns:
            np.ndarray: scaled spectrum
        """
        # sums for scalation
        s1 = self.win.sum()
        s2 = np.power(self.win, 2).sum()

        # scaling factor
        if self.spectrumtype == "power":
            scaling_factor = s1**2
        elif self.spectrumtype == "psd":
            scaling_factor = s2 * self.fs
        else:
            raise ValueError("spectrumtype value is wrong")
        scaled_spectrum = spectrum / scaling_factor

        return scaled_spectrum

    @staticmethod
    def _one_sided_2_two_sided_correction(
        spectrum: np.ndarray, nfft: int
    ) -> np.ndarray:
        """Correction from one-sided spectrum to two-sided

        Note that the output spectrum only contains the positive frequencies as expected
        by irfft(), we just apply the power correction to the one-sided bins.

        Args:
            spectrum (np.ndarray): input one-sided spectrum
            nfft (int): FFT length

        Returns:
            np.ndarray: corrected spectrum
        """
        spectrum2 = spectrum.copy()
        if np.remainder(nfft, 2):
            # odd Nfft, Nyquist point not included
            spectrum2[1:] = spectrum[1:] / 2
        else:
            # even Nfft, Nyquist point included
            spectrum2[1:-1] = spectrum[1:-1] / 2

        return spectrum2

    def plot(self):
        if self.output is None:
            raise RuntimeError(
                "Analysis not done. Execute 'run()' before calling 'plot()'"
            )
        else:
            # matplotlib general settings
            mpl.rcParams["lines.linewidth"] = 1.2
            mpl.rcParams["font.size"] = 10

            # Figure 1: Frequency domain ###############################################

            fig1, axs1 = plt.subplots(3, 2, figsize=(12, 7), tight_layout=True)
            fig1.suptitle(
                "Dual Channel FFT Analyzer - Frequency domain", fontweight="bold"
            )

            # autospectrum
            axs1[0, 0].plot(
                self.output["freq_axis"],
                10 * np.log10(self.output["autospectrum_a"]),
                color="sienna",
                label="Input A",
            )
            axs1[0, 0].plot(
                self.output["freq_axis"],
                10 * np.log10(self.output["autospectrum_b"]),
                color="springgreen",
                label="Input B",
            )
            axs1[0, 0].set_title("Autospectrum")
            axs1[0, 0].legend()

            # cross-spectrum
            axs1[0, 1].plot(
                self.output["freq_axis"],
                10 * np.log10(np.abs(self.output["cross_spectrum"])),
                color="steelblue",
            )
            axs1[0, 1].set_title("Cross-Spectrum")

            # coherence
            axs1[1, 0].plot(
                self.output["freq_axis"], self.output["coherence"], color="black"
            )
            axs1[1, 0].set_title("Coherence")

            # signal-to-noise ratio
            axs1[1, 1].plot(
                self.output["freq_axis"],
                10 * np.log10(self.output["snr"]),
                color="black",
            )
            axs1[1, 1].set_title("Signal-To-Noise Ratio (SNR)")

            # coherent/non-coherent output power
            axs1[2, 0].plot(
                self.output["freq_axis"],
                10 * np.log10(self.output["coherent_output_power"]),
                color="salmon",
                label="coherent",
            )
            axs1[2, 0].plot(
                self.output["freq_axis"],
                10 * np.log10(self.output["non_coherent_output_power"]),
                color="gold",
                label="non-coherent",
            )
            axs1[2, 0].set_title("Ouput power")
            axs1[2, 0].legend()

            # frequency responses
            axs1[2, 1].plot(
                self.output["freq_axis"],
                20 * np.log10(np.abs(self.output["freq_resp_h1"])),
                color="chartreuse",
                label="H1",
            )
            axs1[2, 1].plot(
                self.output["freq_axis"],
                20 * np.log10(np.abs(self.output["freq_resp_h2"])),
                color="deeppink",
                label="H2",
            )
            axs1[2, 1].set_title("Frequency response")
            axs1[2, 1].legend()

            # more formatting
            [
                axs1[x, y].set_xlabel("Frequency (Hz)")
                for x in range(3)
                for y in range(2)
            ]
            [axs1[x, y].set_xscale("log") for x in range(3) for y in range(2)]
            [
                axs1[x, y].grid(alpha=0.3, which="both")
                for x in range(3)
                for y in range(2)
            ]
            [
                axs1[x].set_ylabel(
                    "PSD (dBov/Hz)" if self.spectrumtype == "psd" else "Power (dBov)"
                )
                for x in [(0, 0), (0, 1), (2, 0)]
            ]
            [axs1[x].set_ylabel("dB") for x in [(1, 1), (2, 1)]]

            plt.show(block=False)

            # Figure 2: Time domain ####################################################

            fig2, axs2 = plt.subplots(2, 2, figsize=(12, 7), tight_layout=True)
            fig2.suptitle("Dual Channel FFT Analyzer - Time domain", fontweight="bold")

            # autocorrelation
            axs2[0, 0].plot(
                self.output["lags"],
                self.output["autocorrelation_a"],
                color="sienna",
                label="Input A",
            )
            axs2[0, 0].set_title("Autocorrelation: A")

            axs2[0, 1].plot(
                self.output["lags"],
                self.output["autocorrelation_b"],
                color="springgreen",
                label="Input B",
            )
            axs2[0, 1].set_title("Autocorrelation: B")

            # cross-correlation
            axs2[1, 0].plot(
                self.output["lags"],
                self.output["cross_correlation"],
                color="steelblue",
            )
            axs2[1, 0].set_title("Cross-correlation")

            # impulse response
            axs2[1, 1].plot(
                self.output["impulse_resp_h1"],
                color="chartreuse",
                label="h1",
            )
            axs2[1, 1].plot(
                self.output["impulse_resp_h2"],
                color="deeppink",
                label="h2",
            )
            axs2[1, 1].set_title("Impulse response")
            axs2[1, 1].set_xlabel("samples")
            axs2[1, 1].legend()

            # more formatting
            [
                axs2[x, y].grid(alpha=0.3, which="both")
                for x in range(2)
                for y in range(2)
            ]
            [axs2[x].set_xlabel("lag (samples)") for x in [(0, 0), (0, 1), (1, 0)]]

            plt.show()


if __name__ == "__main__":
    # example
    config = {
        "verbose": True,
        "nfft": 4096,
        "window": "hann",
        "overlap": 50,
        "spectrumtype": "psd",
        "delay": 0.0003125,  # 5 samples at fs=16 kHz
    }
    analyzer = FFTAnalyzer(config)
    print(analyzer)
    analyzer.run("data/input_a.wav", "data/input_b.wav")
    analyzer.plot()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_dual_channel_fft_analyzer.py
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from .context import dual_channel_fft_analyzer


# some fixtures for the tests
@pytest.fixture
def settings():
    config = {
        "verbose": True,
        "nfft": 1024,
        "window": "hann",
        "spectrumtype": "psd",
        "overlap": 50,
        "delay": 0,
    }
    return config


@pytest.fixture
def input_a():
    return "data/input_a.wav"


@pytest.fixture
def input_b():
    return "data/input_b.wav"


@pytest.fixture
def input_fs8k():
    return "data/input_fs8k.wav"


@pytest.fixture
def test_vector_1():
    return np.expand_dims(
        np.array([-0.7407, -0.0482, -0.1884, 0.9999, -1.0000, -0.5281, -0.3260]),
        axis=1,
    )


@pytest.fixture
def test_vector_2():
    return np.expand_dims(
        np.array([0.0949, -0.6100, 1.0000, -0.6525, 0.0718, 0.3367, 0.8633]),
        axis=1,
    )


@pytest.fixture
def analyzer(settings):
    return dual_channel_fft_analyzer.FFTAnalyzer(settings)


# init() ###############################################################################


class TestFFTAnalyzer:
    def test_init(self, analyzer, settings):
        """Test initialization of FFTAnalyzer class

        Args:
            analyzer (FFTAnalyzer): FFTAnalyzer object
            settings (dict): Settings
        """
        assert analyzer.verbose == settings["verbose"]
        assert analyzer.nfft == settings["nfft"]
        assert analyzer.overlap == settings["overlap"]
        assert analyzer.win.size == settings["nfft"]
        assert analyzer.spectrumtype == settings["spectrumtype"]
        assert analyzer.delay == settings["delay"]

    def test_window(self, analyzer, settings):
        assert 1 == 1


# _read_inputs() #######################################################################


class TestReadInput:
    def test_read_input(self, analyzer, input_a, input_b):
        analyzer._read_input(input_a, input_b)
        assert analyzer.input_a.shape == (960000, 1)
        assert analyzer.input_b.shape == (960199, 1)

    def test_exception_fs(self, analyzer, input_a, input_fs8k):
        with pytest.raises(ValueError):
            analyzer._read_input(input_a, input_fs8k)


# _signal_conditioning() ###############################################################


class TestSignalConditioning:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        self.__class__.input = [
            np.array([0], dtype="uint8"),
            np.array([-32768], dtype="int16"),
            np.array([-2147483648], dtype="int32"),
            np.array([-1.0], dtype="float32"),
        ]

    def test_scaling(self, analyzer):
        for x in [analyzer._signal_conditioning(x) for x in self.input]:
            assert x == np.array([[-1.0]], dtype=np.float64)

    def test_format(self, analyzer):
        for x in [analyzer._signal_conditioning(x) for x in self.input]:
            assert x.shape == (1, 1)
            assert x.dtype == np.float64

    def test_exception_nchannels(self, analyzer):
        x = np.array([[1, 2], [1, 2]], dtype=np.float64)
        with pytest.raises(ValueError):
            analyzer._signal_conditioning(x)


# _crossspectrum() #####################################################################


class TestSpectrum:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        config = {
            "verbose": False,
            "nfft": 4,
            "window": "hann",
            "spectrumtype": "psd",
            "overlap": 50,
            "delay": 0,
        }
        self.__class__.analyzer = dual_channel_fft_analyzer.FFTAnalyzer(config)

    def test_crossspectrum(self, test_vector_1, test_vector_2):
        output = self.analyzer._crossspectrum(
            test_vector_1,
            test_vector_2,
            self.analyzer.nfft,
            self.analyzer.overlap,
            self.analyzer.win,
            fs=1000,
        )
        assert_allclose(
            output["spectrum"],
            np.array(
                [
                    [0.10599719 + 0.0j, 0.06578901 + 0.0j],
                    [-0.3990721 + 1.040093j, -0.8993488 + 0.8794896j],
                    [-1.0835578 - 0.0j, -0.28388622 - 0.0j],
                ],
                dtype=np.complex64,
            ),
            rtol=1e-07,
        )

    def test_scale_spectrum(self):
        # psd scalation
        self.analyzer.win = np.array([1, 1, 1])
        self.analyzer.fs = 2
        spectrum = np.array([6, 6, 6], dtype=np.complex64)
        scaled_spectrum = self.analyzer._scale_spectrum(spectrum)
        assert_array_equal(scaled_spectrum, np.array([1, 1, 1]))

        # power scalation
        self.analyzer.spectrumtype = "power"
        spectrum = np.array([9, 9, 9], dtype=np.complex64)
        scaled_spectrum = self.analyzer._scale_spectrum(spectrum)
        assert_array_equal(scaled_spectrum, np.array([1, 1, 1]))

    def test_one2twosides_correction(self):
        spectrum = np.array([1, 1, 1], dtype=np.float64)
        # even nfft
        assert_array_equal(
            self.analyzer._one_sided_2_two_sided_correction(spectrum, nfft=2),
            np.array([1, 0.5, 1]),
        )
        # odd nfft
        assert_array_equal(
            self.analyzer._one_sided_2_two_sided_correction(spectrum, nfft=3),
            np.array([1, 0.5, 0.5]),
        )


# _dual_channel_fft() ##################################################################


class TestDualChannelFFT:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        config = {
            "verbose": False,
            "nfft": 4,
            "window": "hann",
            "spectrumtype": "psd",
            "overlap": 50,
            "delay": 2 / 1000,
        }
        self.__class__.analyzer = dual_channel_fft_analyzer.FFTAnalyzer(config)

    def test_dual_channel_fft(self, test_vector_1, test_vector_2):
        # same vector at both inputs to simplify (with some delay to compensate)
        self.analyzer.input_a = np.concatenate((np.zeros((2, 1)), test_vector_1))
        self.analyzer.input_b = test_vector_1
        self.analyzer.fs = 1000
        self.analyzer.win = np.ones(
            (self.analyzer.nfft,)
        )  # rectangular window to simplify

        expected = {
            "autospectrum": np.array([6.42532905e-05, 1.09925692e-03, 7.86709134e-04]),
            "cross_spectrum": np.array(
                [6.4253290e-05 + 0.0j, 1.0992569e-03 + 0.0j, 7.8670913e-04 + 0.0j],
                dtype=np.complex64,
            ),
            "freq_axis": np.array([0.0, 250.0, 500.0]),
            "coherence": np.ones((3,)),
            "snr": np.full((3,), np.inf),
            "correlation": np.array([1.9502192, -0.7224558, -0.24829453, -0.7224558]),
            "lags": np.array([0, 1, 2, 3]),
            "freq_resp": np.ones((3,), dtype=np.complex64),
            "impulse_resp": np.array([1, 0, 0, 0]),
        }

        output = self.analyzer._dual_channel_fft()

        tolerance = 1e-12

        assert_allclose(
            output["autospectrum_a"],
            expected["autospectrum"],
            atol=tolerance,
        )

        assert_allclose(
            output["autospectrum_b"],
            expected["autospectrum"],
            atol=tolerance,
        )

        assert_allclose(
            output["cross_spectrum"],
            expected["cross_spectrum"],
            atol=tolerance,
        )

        assert_allclose(
            output["freq_axis"],
            expected["freq_axis"],
            atol=tolerance,
        )

        assert_allclose(
            output["coherence"],
            expected["coherence"],
            atol=tolerance,
        )

        assert_allclose(
            output["coherent_output_power"],
            expected["autospectrum"],
            atol=tolerance,
        )

        assert_allclose(
            output["non_coherent_output_power"],
            np.zeros((3,)),
            atol=tolerance,
        )

        assert_allclose(
            output["snr"],
            expected["snr"],
            atol=tolerance,
        )

        assert_allclose(
            output["autocorrelation_a"],
            expected["correlation"],
            atol=tolerance,
        )

        assert_allclose(
            output["autocorrelation_b"],
            expected["correlation"],
            atol=tolerance,
        )

        assert_allclose(
            output["cross_correlation"],
            expected["correlation"],
            atol=tolerance,
        )

        assert_allclose(
            output["lags"],
            expected["lags"],
            atol=tolerance,
        )

        assert_allclose(
            output["freq_resp_h1"],
            expected["freq_resp"],
            atol=tolerance,
        )

        assert_allclose(
            output["freq_resp_h2"],
            expected["freq_resp"],
            atol=tolerance,
        )

        assert_allclose(
            output["impulse_resp_h1"],
            expected["impulse_resp"],
            atol=tolerance,
        )

        assert_allclose(
            output["impulse_resp_h2"],
            expected["impulse_resp"],
            atol=tolerance,
        )

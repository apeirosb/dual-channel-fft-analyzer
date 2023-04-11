#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI for the Dual Channel FFT Analyzer
"""

import argparse
import dual_channel_fft_analyzer


def parse_arguments():
    """Parses arguments from the command line interface

    Returns:
        dict: Dictionary with all the config parameters
    """

    # CLI
    parser = argparse.ArgumentParser(description="Dual Channel FFT Analyzer")
    parser.add_argument(
        "input_a",
        type=str,
        help="path to the recording (wav) used as input for Channel A",
    )
    parser.add_argument(
        "input_b",
        type=str,
        help="path to the recording (wav) used as input for Channel B",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose mode (default: False)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="plot results (default: False)",
    )
    parser.add_argument(
        "--nfft", type=int, default=512, help="FFT length (default: 512)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="percentage to overlap between segments when averaging (default: 50)",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        help=(
            "window type used for the analysis. It can be any Scipy-compatible window "
            "(default: hann)"
        ),
    )
    parser.add_argument(
        "--spectrumtype",
        type=str,
        choices=["power", "psd"],
        default="psd",
        help="power spectrum scaling (default: psd)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="propagation delay to compensate in ms (default: 0)",
    )

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    # CLI parser
    args = parse_arguments()

    # run FFTAnalyzer
    fft = dual_channel_fft_analyzer.FFTAnalyzer(args)
    fft.run(args["input_a"], args["input_b"])
    if args["plot"]:
        fft.plot()

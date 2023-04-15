# -*- coding: utf-8 -*-
import numpy as np
from sidekit.frontend.features import framing
from sidekit.frontend.vad import pre_emphasis


def ltas(input_sig, fs=16000, fc=0, win_time=0.02, shift_time=0.01):
    """
    Extracts long-term spectral statistics of given speech signal.

    Args:
        input_sig: Speech signal.
        fs: Sample rate of the signal.
        fc: F cut frequency.
        win_time: Length of the sliding window in seconds.
        shift_time: Shift between two analyses.

    Returns:
        Mean and variance statistics of fourier magnitude spectrum.
    """
    # Add this value on elements to avoid dividing values to zero.
    eps = 2.2204e-16

    # Pre-emphasis on signal.
    input_sig = pre_emphasis(input_sig, 0.97)

    # Calculate frame and overlap length in terms of samples.
    window_len = int(round(win_time * fs))
    overlap = window_len - int(shift_time * fs)

    # Split signal into frames.
    framed_sig = framing(sig=input_sig, win_size=window_len, win_shift=window_len - overlap,
                         context=(0, 0), pad="zeros")

    # Windowing.
    window = np.hamming(window_len)
    windowed_sig = framed_sig * window

    # Number of fft points.
    n_fft = 2 ** int(np.ceil(np.log2(window_len)))

    # Low frequency cutting variables.
    start = 0 if fc == 0 else int((fc * n_fft) / fs)
    end = int(n_fft / 2) + 1
    n_elements = int(end - start)

    # A placeholder for spectrum.
    spectrum = np.zeros([framed_sig.shape[0], n_elements])

    # N point FFT of signal.
    fft = np.fft.rfft(windowed_sig, n_fft)

    # Fourier magnitude spectrum computation.
    log_magnitude = np.log(abs(fft) + eps)

    # Crop components according to fc.
    for frame in range(windowed_sig.shape[0]):
        spectrum[frame] = log_magnitude[frame][start:end]

    # Concatenate mean and variance statistics.
    mu = np.mean(spectrum, axis=0)
    sigma = np.std(spectrum, axis=0)

    return np.concatenate((mu, sigma))

import librosa
import matplotlib
import numpy
import sidekit

# Genuine file.
path_to_genuine_file = '.../T_1001506.wav'

y, sr = librosa.load(path_to_genuine_file)

# A pre-computed power spectrogram.
D = numpy.abs(librosa.stft(y)) ** 2
S = librosa.feature.melspectrogram(S=D)

# Passing through arguments to the Mel filters.
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
# Plot spectrogram.
matplotlib.pyplot.plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=numpy.max), y_axis='mel',
                         fmax=8000, x_axis='time')
matplotlib.pyplot.plt.colorbar(format='%+2.0f dB')
matplotlib.pyplot.plt.title('Mel Spectrogram')
matplotlib.pyplot.plt.tight_layout()

# Spoof file.
path_to_genuine_file = '.../T_1001514.wav'

# A pre-computed power spectrogram.
D = numpy.abs(librosa.stft(y)) ** 2
S = librosa.feature.melspectrogram(S=D)

# Passing through arguments to the Mel filters.
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
# Plot spectrogram.
matplotlib.pyplot.plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=numpy.max), y_axis='mel',
                         fmax=8000, x_axis='time')
matplotlib.pyplot.plt.colorbar(format='%+2.0f dB')
matplotlib.pyplot.plt.title('Mel Spectrogram')
matplotlib.pyplot.plt.tight_layout()

x, fs, _ = sidekit.frontend.io.read_wav('.../sample.wav')

input_sig = x
win_time = 0.02
shift_time = 0.01

# Add this value on elements to avoid dividing values to zero.
eps = 2.2204e-16

# Pre-emphasis on signal.
input_sig = sidekit.frontend.vad.pre_emphasis(input_sig, 0.97)

# Calculate frame and overlap length in terms of samples.
window_len = int(round(win_time * fs))
overlap = window_len - int(shift_time * fs)

# Split signal into frames.
framed_sig = sidekit.frontend.features.framing(sig=input_sig,
                                               win_size=window_len,
                                               win_shift=window_len - overlap,
                                               context=(0, 0),
                                               pad='zeros')

# Windowing.
window = numpy.hamming(window_len)
windowed_sig = framed_sig * window

# Number of fft points.
n_fft = 2 ** int(numpy.ceil(numpy.log2(window_len)))

# N point FFT of signal.
fft = numpy.fft.rfft(windowed_sig, n_fft)
a_fft = abs(fft)

# Fourier magnitude spectrum computation.
log_magnitude = numpy.log(abs(fft) + eps)

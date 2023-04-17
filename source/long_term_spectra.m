function feats = long_term_spectra(x, fs)
    % Calculate long-term average spectra (LTAS) features from an input signal x
    % with a given sampling rate fs.

    % Set frame length and frame shift for windowing.
    frame_length = 150 * fs / 1000;
    frame_shift = frame_length / 2;

    % Window the input signal into overlapping frames.
    frames = enframe(x + eps, hamming(frame_length), frame_shift)';

    % Pre-emphasis on each frame.
    % B = [1 -0.97];

    % Set the FFT size.
    nfft = 2 ^ nextpow2(frame_length);

    % Initialize LTAS matrix.
    LTAS = zeros(nfft / 2 + 1, size(frames, 2));

    % Calculate LTAS for each frame.
    for k = 1:size(frames, 2)
        % y = filter(B, 1, frames(:, k));
        y = frames(:, k);
        Y = abs(fft(y, nfft));
        LTAS(:, k) = Y(1:nfft / 2 + 1);
    end

    % Calculate mean and standard deviation of log-spectra.
    mu = mean(log(LTAS + eps), 2);
    sigma = std(log(LTAS + eps), 0, 2);

    % Combine mean and standard deviation as output features.
    feats = [mu; sigma]';
end

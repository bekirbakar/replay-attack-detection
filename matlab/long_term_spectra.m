function feats = long_term_spectra(x,fs)

frame_length = 150*fs/1000;
frame_shift = frame_length/2;
frames = enframe(x+eps,hamming(frame_length),frame_shift)'; % frames

Pre-emphasis on each frame.
B = [1 -0.97];

nfft = 2^nextpow2(frame_length);

LTAS = zeros(nfft/2+1,size(frames,2));

for k = 1:size(frames,2)
    %y = filter(B,1,frames(:,k));
    y = frames(:,k);
    Y = abs(fft(y,nfft));
    LTAS(:,k) = Y(1:nfft/2+1);
end

mu = mean(log(LTAS+eps),2);

feats = mu';

sigma = std(log(LTAS+eps),0,2);

feats = [mu;sigma]';

end

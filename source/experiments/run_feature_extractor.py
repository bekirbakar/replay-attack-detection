# -*- coding: utf-8 -*-
from sidekit.frontend.io import read_wav

from source.features import *
from source.preprocessing import *


def run_feature_extractor(data, path_to_feats, feature_types, subset_list,
                          normalize=False):
    for feature_type in feature_types:
        for subset in subset_list:
            feat_dir = path_to_feats + feature_type + "/" + subset
            path_to_wav = data["path_to_wav"]
            file_list = data["file_list"]
            labels = data["labels"]

            # Loop through files.
            print(f"\nExtracting {feature_type} {subset} features.")
            for file in range(len(file_list)):
                x, fs, _ = read_wav(path_to_wav + file_list[file])

                # Select feature type.
                if feature_type == "frames":
                    feat = extract_frames(input_sig=x, fs=16000,
                                          win_time=0.02, shift_time=0.01)
                elif feature_type == "mfcc":
                    feat = mfcc(input_sig=x, lowfreq=100, maxfreq=8000,
                                nlinfilt=0, nlogfilt=24, nwin=0.02, fs=16000,
                                nceps=13, shift=0.01, get_spec=False,
                                get_mspec=False, prefac=0.97)[0]
                elif feature_type == "power_spectrum":
                    feat = power_spectrum(x, fs, win_time=0.02, shift=0.01)[0]
                elif feature_type == "ltas":
                    feat = long_term_spectra(x, 16000)
                    # fc=0, win_time=0.05, shift_time=0.01)
                else:
                    raise AssertionError("\nFeature type not understood.")

                # Normalization.
                if normalize is not False:
                    if normalize == "cmvn":
                        feat = cmvn(feat.copy())
                    elif normalize == "cvn":
                        feat = cvn(feat.copy())
                    elif normalize == "cms":
                        feat = cms(feat.copy())

                # Save file.
                filename = file_list[file].replace(".wav", ".pkl.gz")
                path_to_feat = f"{feat_dir}/{filename}"
                with open(path_to_feat, "wb+") as fp:
                    pickle.dump([feat, labels[file]], fp)

            print("Done")

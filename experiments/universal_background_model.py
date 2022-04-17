# -*- coding: utf-8 -*-
import numpy
import sidekit

import source.helpers as helpers

# Environment Variables
path_to_database = ""
train_protocol_file = ""
dev_protocol_file = ""

# Train Protocol
protocol = numpy.genfromtxt(train_protocol_file, delimiter=" ", dtype=str)

# Get File Lists
file_list = []
for i in range(0, len(protocol)):
    buffer = protocol[i, 0]
    file_list.append(buffer[0:9])

# Get Labels
labels = []
for i in range(0, len(protocol)):
    labels.append(protocol[i, 1])

# Indices of Genuine and Spoof Files
genuineIdx = []
spoofIdx = []
for i in range(0, len(labels)):
    if "genuine" in labels[i]:
        genuineIdx.append(i)
    elif "spoof" in labels[i]:
        spoofIdx.append(i)

extractor = sidekit.FeaturesExtractor(
    audio_filename_structure=path_to_database + "train/" + "{}.wav",
    feature_filename_structure=path_to_database + "train/feat/" + "{}.h5",
    sampling_frequency=16000,
    lower_frequency=0,
    higher_frequency=8000,
    filter_bank="lin",
    filter_bank_size=30,
    window_size=0.02,
    shift=0.01,
    ceps_number=20,
    vad=None,
    snr=40,
    pre_emphasis=0.97,
    save_param=["cep"],
    keep_all_features=True
)

server = sidekit.FeaturesServer(
    features_extractor=None,
    feature_filename_structure=path_to_database + "train/feat/" + "{}.h5",
    sources=None,
    dataset_list=["cep"],
    mask=None,
    feat_norm=None,
    global_cmvn=None,
    dct_pca=False,
    dct_pca_config=None,
    sdc=False,
    sdc_config=None,
    delta=True,
    double_delta=True,
    delta_filter=None,
    context=None,
    traps_dct_nb=None,
    rasta=False,
    keep_all_features=True
)

# Delete old files in directory.
helpers.delete_files(path_to_database + "train/feat", ".h5")

# Extract features for GENUINE training data and store.
for i in range(0, int(len(genuineIdx))):
    extractor.save(file_list[genuineIdx[i]])

# Extract features for SPOOF training data and store.
for i in range(0, int(len(spoofIdx))):
    extractor.save(file_list[spoofIdx[i]])

# UBM
ubm_genuine_list = []
for i in range(0, len(genuineIdx)):
    ubm_genuine_list.append(file_list[genuineIdx[i]])

ubm_spoof_list = []
for i in range(0, len(spoofIdx)):
    ubm_spoof_list.append(file_list[spoofIdx[i]])

ubm = sidekit.Mixture()
spoof = sidekit.Mixture()

# Delete old files in directory.
helpers.delete_files(path_to_database + "train/feat/ubm", ".h5")

ubm.EM_split(features_server=server,
             feature_list=ubm_genuine_list,
             distrib_nb=512,
             iterations=(1, 2, 2, 4, 4, 4, 4),
             llk_gain=0.01,
             num_thread=1,
             save_partial=True,
             ceil_cov=10,
             floor_cov=1e-2)
ubm.write(path_to_database + "train/feat/ubm/genuine.h5")

spoof.EM_split(features_server=server,
               feature_list=ubm_spoof_list,
               distrib_nb=512,
               iterations=(1, 2, 2, 4, 4, 4, 4),
               llk_gain=0.01,
               num_thread=1,
               save_partial=True,
               ceil_cov=10,
               floor_cov=1e-2)
spoof.write(path_to_database + "train/feat/ubm/spoof.h5")

# Development protocol.
protocol = numpy.genfromtxt(dev_protocol_file, delimiter=" ", dtype=str)

# Get file list.
file_list = []
for i in range(0, len(protocol)):
    buffer = protocol[i, 0]
    file_list.append(buffer[0:9])

# Get label list.
labels = []
for i in range(0, len(protocol)):
    labels.append(protocol[i, 1])

# Feature server, extractor  attributes for training data.
extractor.audio_filename_structure = path_to_database + "dev/" + "{}.wav"
extractor.feature_filename_structure = path_to_database + "dev/feat/" + "{}.h5"
server.feature_filename_structure = path_to_database + "dev/feat/" + "{}.h5"

# Delete old files in directory.
helpers.delete_files(path_to_database + "dev/feat", ".h5")

# Process each development trial.
for i in range(0, len(file_list)):
    extractor.save(file_list[i])

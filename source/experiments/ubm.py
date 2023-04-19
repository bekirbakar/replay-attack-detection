import numpy
import sidekit


def delete_files():
    pass


path_to_database = ''
train_protocol_file = ''
dev_protocol_file = ''

protocol = numpy.genfromtxt(train_protocol_file, delimiter=' ', dtype=str)

file_list = []
for i in range(len(protocol)):
    buffer = protocol[i, 0]
    file_list.append(buffer[:9])

labels = [protocol[i, 1] for i in range(len(protocol))]

genuineIdx = []
spoofIdx = []
for i in range(len(labels)):
    if 'genuine' in labels[i]:
        genuineIdx.append(i)
    elif 'spoof' in labels[i]:
        spoofIdx.append(i)

extractor = sidekit.FeaturesExtractor(
    audio_filename_structure=f'{path_to_database}train/' + '{}.wav',
    feature_filename_structure=f'{path_to_database}train/feat/' + '{}.h5',
    sampling_frequency=16000,
    lower_frequency=0,
    higher_frequency=8000,
    filter_bank='lin',
    filter_bank_size=30,
    window_size=0.02,
    shift=0.01,
    ceps_number=20,
    vad=None,
    snr=40,
    pre_emphasis=0.97,
    save_param=['cep'],
    keep_all_features=True,
)

server = sidekit.FeaturesServer(
    features_extractor=None,
    feature_filename_structure=f'{path_to_database}train/feat/' + '{}.h5',
    sources=None,
    dataset_list=['cep'],
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
    keep_all_features=True,
)

delete_files(f'{path_to_database}train/feat', '.h5')

for item__ in genuineIdx:
    extractor.save(file_list[item__])

for item___ in spoofIdx:
    extractor.save(file_list[item___])

ubm_genuine_list = [file_list[item] for item in genuineIdx]
ubm_spoof_list = [file_list[item_] for item_ in spoofIdx]
ubm = sidekit.Mixture()
spoof = sidekit.Mixture()

delete_files(f'{path_to_database}train/feat/ubm', '.h5')

ubm.EM_split(features_server=server,
             feature_list=ubm_genuine_list,
             distrib_nb=512,
             iterations=(1, 2, 2, 4, 4, 4, 4),
             llk_gain=0.01,
             num_thread=1,
             save_partial=True,
             ceil_cov=10,
             floor_cov=1e-2)
ubm.write(f'{path_to_database}train/feat/ubm/genuine.h5')

spoof.EM_split(features_server=server,
               feature_list=ubm_spoof_list,
               distrib_nb=512,
               iterations=(1, 2, 2, 4, 4, 4, 4),
               llk_gain=0.01,
               num_thread=1,
               save_partial=True,
               ceil_cov=10,
               floor_cov=1e-2)
spoof.write(f'{path_to_database}train/feat/ubm/spoof.h5')

protocol = numpy.genfromtxt(dev_protocol_file, delimiter=' ', dtype=str)

file_list = []
for i in range(len(protocol)):
    buffer = protocol[i, 0]
    file_list.append(buffer[:9])

labels = [protocol[i, 1] for i in range(len(protocol))]

extractor.audio_filename_structure = f'{path_to_database}dev/' + '{}.wav'
extractor.feature_filename_structure = f'{path_to_database}dev/feat/' + '{}.h5'
server.feature_filename_structure = f'{path_to_database}dev/feat/' + '{}.h5'

delete_files(f'{path_to_database}dev/feat', '.h5')

for file in file_list:
    extractor.save(file)

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics

from src.helper_results import aucs_results_to_df

PLACES = ['street_traffic', 'metro_station', 'park', 'metro', 'street_pedestrian', 'shopping_mall', 'tram', 'bus',
          'public_square', 'airport']

CUDA = torch.device('cuda')

def compute_rec_errs(model, data_loader, device=CUDA):
    model.to(device)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for segments in data_loader:
            segments = segments.to(device)
            targets.append(segments)
            r_segments, _ = model(segments)
            preds.append(r_segments)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    _SE = (preds - targets) ** 2

    return _SE.mean(axis=-1).mean(axis=-1).detach().cpu()


def compute_labels(target_shape, e_idx, device=CUDA):
    labels = np.zeros(target_shape, dtype=np.int32)
    labels[e_idx] = 1
    return labels


def get_aucs(labels, errors):
    fpr, tpr, wn_thresholds = metrics.roc_curve(labels, errors)
    roc_auc_mse = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(labels, errors)
    pr_auc_score = metrics.auc(recall, precision)
    return roc_auc_mse, pr_auc_score


def compute_aucs(place='airport', model=None, model_name='conv-spad', segment_length=16384, event_type='gunshot',
                 mix_data_loader=None, e_idx=None, ebr=0, run_id=0, working_dir=None, df_name='cae_ci_results.csv',
                 device=CUDA):
    _err = compute_rec_errs(model, mix_data_loader, device=device)
    labels = compute_labels(_err.shape[0], e_idx)
    roc_auc_mse, pr_auc_score = get_aucs(labels, _err)
    df_path = working_dir + '/' + df_name
    aucs_results_to_df(place, event_type, model_name, segment_length, ebr, roc_auc_mse, pr_auc_score, run_id, df_path)

kwargs = {}

def test(data_path, model_path, model_name, segment_length, places=PLACES, event_types=['gunshot', 'glassbreak'],
         ebrs=[-6, 0, 6], n_runs=5, tr_bs=128, bs=1024, epochs=100, suffix='_44k_mono_spec.pt',
         model_name_csv='spad-100', fully=False):
    global kwargs
    bs = bs * 2 if segment_length == 16384 else bs
    bs = bs * 4 if segment_length == 4096 else bs

    for place in places:
        model_file_name = place + model_name + str(tr_bs) + '_' + str(epochs) + '.pt'
        model = torch.load(model_path + '/' + model_file_name)
        model.eval()
        for seed in range(n_runs):
            final_path = data_path + '/' + str(segment_length) + '/test/' + str(seed) + '/' + place
            for event_type in event_types:
                for ebr in ebrs:
                    print(f'{seed}) Place {place}, input {segment_length}, mixed with {event_type} at {ebr} testing ... ')
                    _suffix = 'mix_' + event_type + '_at_' + str(ebr) + '_on_' + place + suffix
                    test_data_path = final_path + '/' + _suffix
                    _data = torch.load(test_data_path)
                    mix_audio_features = _data['data']
                    mix_data_loader = DataLoader(dataset=mix_audio_features, batch_size=bs, shuffle=False, **kwargs)
                    e_idx = _data['index']
                    compute_aucs(place=place, model=model, model_name=model_name_csv,
                                 segment_length=segment_length, mix_data_loader=mix_data_loader, e_idx=e_idx,
                                 event_type=event_type, run_id=seed, ebr=ebr, working_dir=model_path)

if __name__ == '__main__':
    data_path = 'data/prepro'
    model_path = 'cae'

    # seg_lengths = [4096, 16384, 44100]
    seg_lengths = [4096]
    # places = ['metro', 'metro_station', 'park', 'tram', 'bus']
    places = PLACES #['street_traffic', 'street_pedestrian', 'shopping_mall', 'public_square', 'airport']
    if True:
        for sl in seg_lengths:
            epochs = 100
            model_name = '_cae_' + str(sl) + '_256_80_64_'
            suffix = '_44k_mono_spec_256_80_64.pt'
            model_name_csv = 'conv-spad-256-64-'+str(epochs)

            test(data_path=data_path, model_path=model_path, model_name=model_name,  places=places, ebrs=[-6, 0, 6],
                 segment_length=sl, epochs=epochs, n_runs=5, suffix=suffix, model_name_csv=model_name_csv)
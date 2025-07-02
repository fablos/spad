import os
import pandas as pd
import numpy as np
from sklearn import metrics




PLACES = ['street_traffic', 'metro_station', 'park', 'metro', 'street_pedestrian',
          'shopping_mall', 'tram', 'bus', 'public_square', 'airport']



def aucs_results_to_df(place, event_type, model_name, segment_length, ebr, roc_auc_mse, pr_auc_score, run_id, df_path):
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame(
            columns=['place', 'event_type', 'model', 'segment_length', 'ebr', 'roc_auc_mse', 'pr_auc_score', 'run_id'])
    df = df.append(
        pd.Series(
            {'place': place, 'event_type': event_type, 'model': model_name, 'segment_length': segment_length, 'ebr': ebr,
             'roc_auc_mse': roc_auc_mse, 'pr_auc_score': pr_auc_score, 'run_id': run_id}),
        ignore_index=True)
    df = df.drop_duplicates(subset=['place', 'event_type', 'model', 'segment_length', 'ebr', 'run_id'], keep='last')
    df.to_csv(df_path, index=False)
    return df


def load_results(scene_name, event_type, ebr, preds_dir, targets_dir, labels_dir):
    preds = np.load("{}/{}_{}_{}_{}.npy".format(preds_dir,'preds_test_WaveNet',scene_name, event_type, ebr))
    preds = np.squeeze(preds)
    targets =np.load("{}/{}_{}_{}_{}.npy".format(targets_dir, 'targets_test_WaveNet', scene_name, event_type, ebr))
    targets = np.squeeze(targets)
    labels =np.load("{}/{}_{}_{}_{}.npy".format(labels_dir,'label_list_WaveNet', scene_name, event_type, ebr))
    labels = np.squeeze(labels)
    return preds, targets, labels


def compute_metrics(wn_preds, wn_targets, wn_labels):
    wn_SE = (wn_preds - wn_targets) ** 2
    wn_mse = wn_SE.mean(axis=1)
    wn_fpr, wn_tpr, wn_thresholds = metrics.roc_curve(wn_labels, wn_mse)
    wn_roc_auc_mse = metrics.auc(wn_fpr, wn_tpr)
    precision, recall, _ = metrics.precision_recall_curve(wn_labels, wn_mse)
    pr_auc_score = metrics.auc(recall, precision)
    return wn_roc_auc_mse, pr_auc_score


# segment_length = 4096
# working_dir = '/home/loscudo/wok/tau_data/wavenet/' + str(segment_length)
# df_path = working_dir + '/cae_ci_results.csv'
# print('WaveNet Results:')
#
# run_id = 0
# for place in PLACES:
#     wavenet_preds_dir   = "/home/loscudo/wok/tau_data/wavenet/" + str(segment_length) + "/test_preds/" + str(run_id) + "/preds_test_WaveNet_"
#     wavenet_targets_dir = "/home/loscudo/wok/tau_data/wavenet/" + str(segment_length) + "/targets/ " + str(run_id) + "/targets_test_WaveNet_"
#     wavenet_labels_dir  = "/home/loscudo/wok/tau_data/wavenet/" + str(segment_length) + "/labels/" + str(run_id) + "/label_list_WaveNet_"
#
#     for event_type in ['glassbreak', 'gunshot']:
#         for ebr in [-6, 0, 6]:
#             wn_preds, wn_targets, wn_labels = load_results(place,wavenet_preds_dir, wavenet_targets_dir, wavenet_labels_dir )
#             wn_roc_auc_mse, pr_auc_score = compute_metrics(wn_preds, wn_targets, wn_labels)
#             aucs_results_to_df(place, event_type, ebr, 'wavenet', segment_length, wn_roc_auc_mse, pr_auc_score, run_id,
#                                df_path)
#             break
#         break
#     break


# df_path = '/mnt/nas/loscudo/workspace/tau_data/wavenet/' + str(4096) + '/cae_ci_results.csv'
# aucs_results_to_df('Nowhere', 'gun', 'wavenet', 4096, 0, 0.8, 0.75, 0, df_path)
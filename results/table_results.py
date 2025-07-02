import pandas as pd
import numpy as np
import sys

pd.options.display.latex.repr = True

PLACES = ['street_traffic', 'metro_station', 'park', 'metro', 'street_pedestrian', 'shopping_mall', 'tram', 'bus',
          'public_square', 'airport']


def compute_ci(std, n_it=10, z=1.96):
    return z * (std / np.sqrt(n_it))


def compute_stat(df, model='cae', place='street_traffic', event_type='gunshot', segment_length=4096, ebr=0, means=True):
    tmp_df = df.loc[
            (df['model'] == model) &
            (df['place'] == place) &
            (df['segment_length'] == segment_length) &
            (df['event_type'] == event_type) &
            (df['ebr'] == ebr)
            ]
    return tmp_df['roc_auc_mse'].mean(), tmp_df['roc_auc_mse'].std(), tmp_df['pr_auc_score'].mean(), tmp_df[
        'pr_auc_score'].std(), tmp_df['place'].shape[0]


def compute_mean_df(input_df, model, place, segment_length, event_type, ebr, means=True):
    _df = pd.DataFrame(columns=['model', 'place', 'name', 'mean', 'std', 'segment_length', 'event_type', 'ebr'])
    roc_m, roc_s, pr_m, pr_s, n_it = compute_stat(input_df, model=model, place=place, segment_length=segment_length,
                                            event_type=event_type, ebr=ebr, means=means)
    var = compute_ci(roc_s, n_it=n_it)
    _df = _df.append(pd.Series({'model': model, 'place': place, 'segment_length': segment_length, 'name': 'AUC', 'mean': roc_m,
                                'std': var, 'event_type': event_type, 'ebr': ebr}), ignore_index=True)
    var = compute_ci(pr_s, n_it=n_it)
    _df = _df.append(pd.Series({'model': model, 'place': place, 'segment_length': segment_length, 'name': 'AUPRC', 'mean': pr_m,
                                'std': var, 'event_type': event_type, 'ebr': ebr}), ignore_index=True)
    return _df


def build_ci_df(input_df, model='cae', place='street_traffic', segment_length=16384, event_type='gunshot', means=True):
    _df1_sx = compute_mean_df(input_df, model, place, segment_length, event_type, ebr=-6)
    _df1_sx = _df1_sx.drop(['event_type', 'segment_length'], axis=1)
    _df1_c = compute_mean_df(input_df, model, place, segment_length, event_type, ebr=0)
    _df1_c = _df1_c.drop(['event_type', 'segment_length'], axis=1)
    _df1_dx = compute_mean_df(input_df, model, place, segment_length, event_type, ebr=6)
    _df1_dx = _df1_dx.drop(['event_type'], axis=1)
    merged_inner = pd.merge(left=_df1_sx, right=_df1_c, left_on=['place', 'model', 'name'], right_on=['place', 'model', 'name'])
    merged_inner = merged_inner.drop(['ebr_x', 'ebr_y'], axis=1)
    merged_inner = merged_inner.rename(columns={'mean_x': 'mean_m6', 'std_x': 'std_m6', 'mean_y': 'mean_0', 'std_y': 'std_0'})
    merged_inner = pd.merge(left=merged_inner, right=_df1_dx, left_on=['model', 'place', 'name'], right_on=['model', 'place', 'name'])
    merged_inner = merged_inner.drop(['ebr'], axis=1)
    merged_inner = merged_inner.rename(columns={'mean': 'mean_6', 'std': 'std_6'})
    # _df = _df.drop_duplicates(subset=['name', 'event_type', 'ebr'], keep='last')
    return merged_inner.round(4)


def generate_table_results(df, models=['conv-spad-256-64-100', 'wavenet'],
                places=PLACES, segment_lengths=[4096, 16384, 44100], event_types=['gunshot', 'glassbreak']):
        for event_type in event_types:
            results = []
            for place in sorted(places):
                for segment_length in segment_lengths:
                    for model in models:
                        input_df = df.loc[(df['model'] == model) & (df['segment_length'] == segment_length)]
                        results.append(build_ci_df(input_df, model=model,
                                                   place=place, segment_length=segment_length,
                                                   event_type=event_type))
            r_df = pd.concat(results)

            _m6, _0, _6 = [], [], []
            for i, row in r_df.iterrows():
                # print(row['var_m6'], row['var_0'], row['var_6'])
                _m6.append('$pm ' + str(row['std_m6']) + '$')
                _0.append('$pm ' + str(row['std_0']) + '$')
                _6.append('$pm ' + str(row['std_6']) + '$')

            r_df['std_m6_pm'] = _m6
            r_df['std_0_pm'] = _0
            r_df['std_6_pm'] = _6
            # print(r_df.head(5))
            r_df = r_df[['place', 'name', 'segment_length', 'model', 'mean_m6', 'std_m6_pm', 'mean_0', 'std_0_pm', 'mean_6', 'std_6_pm']]
            original_stdout = sys.stdout
            with open(event_type+'_table.txt', 'w') as f:
                sys.stdout = f
                print(r_df.sort_values(by=['place', 'name', 'segment_length', 'model']).to_latex(index=False))
            sys.stdout = original_stdout


df_4096 = pd.read_csv('wavenet_ci_results_4096.csv')
# df_16384 = pd.read_csv('wavenet_ci_results_16384.csv')
df_cae = pd.read_csv('cae_ci_results.csv')

# df = df_cae
# df = pd.concat([df_4096, df_16384, df_cae])
df = pd.concat([df_4096, df_cae])

places = ['metro_station', 'park', 'tram', 'bus', 'metro']
places = PLACES
generate_table_results(df, places=places, segment_lengths=[4096])
# print_ci_df(df, segment_lengths=[4096])
# print_ci_df(df, models=['conv-spad'])

# print_ci_df(df, segment_lengths=[4096, 16384], event_types=['gunshot'])
# print_ci_df(df, segment_lengths=[4096, 16384], event_types=['glassbreak'])

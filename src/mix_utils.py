import numpy as np
import torch
from tqdm import tqdm


def get_event_amplitude_scaling_factor(s, n, target_ebr_db, method='rmse'):
    """
    Different lengths for signal and noise allowed: longer noise assumed to be stationary enough,
    and rmse is calculated over the whole signal
    """
    rmse = lambda x: torch.sqrt(torch.mean(torch.abs(x) ** 2, dim=0))
    original_sn_rmse_ratio = rmse(s) / rmse(n)
    target_sn_rmse_ratio = 10 ** (target_ebr_db / float(20))
    signal_scaling_factor = target_sn_rmse_ratio / original_sn_rmse_ratio
    return signal_scaling_factor


def simple_mix_procedure(bg_audio, anomaly_samples, magic_anticlipping_factor=0.8, anti_clipping=True):
    bg_audio += anomaly_samples
    mixture = magic_anticlipping_factor * bg_audio
    if anti_clipping and torch.max(torch.abs(mixture)) >= 1:  # make sure that we did not introduce clipping
        mixture /= torch.max(torch.abs(mixture))
    return mixture


def do_mix_on_segments(place_audios: torch.tensor, event_audios: list, magic_anticlipping_factor=0.8, ebr=0,
                       segment_length=4096, seed=42, p=0.6, permute=False, seed_np=False, anti_clipping=True):
    """
    for each audio in input we randomly pick up a anomaly event from the passed list,
    then we sample a random offset to inject the event, apply a anti-clipping factor
    and check is there is any clipping left that we correct via normalization.
    """
    l = []
    e_idx = []
    if seed_np:
        np.random.seed(seed)

    bg_audios = place_audios.clone()
    if permute:
        bg_audios = bg_audios[torch.randperm(bg_audios.shape[0])].view(bg_audios.size())

    samples, audio_samples = bg_audios.shape
    over = audio_samples % segment_length
    bg_audios = torch.narrow(bg_audios, 1, 0, audio_samples - over)
    bg_audios = bg_audios.contiguous().view(-1, segment_length)
    r = np.random.choice(2, bg_audios.shape[0], p=[p, (1 - p)])

    for i in tqdm(range(bg_audios.shape[0]), desc=f'Mixing anomaly events at {ebr}db with background', ascii=True):
        bg_audio = bg_audios[i]
        if r[i] > 0:
            mixture = bg_audio
        else:
            event_audio = event_audios[np.random.choice(len(event_audios), 1)[0]]
            longest_possible_offset = event_audio.shape[0] - segment_length
            event_offset_samples = np.random.choice(longest_possible_offset, 1)[0] if longest_possible_offset > 0 else 0
            anomaly_samples = event_audio[event_offset_samples:(event_offset_samples + segment_length)]
            scaling_factor = get_event_amplitude_scaling_factor(anomaly_samples, bg_audio, target_ebr_db=ebr)
            anomaly_samples = scaling_factor * anomaly_samples
            if bg_audio.shape != anomaly_samples.shape:
                _t = torch.zeros_like(bg_audio)
                _t[:anomaly_samples.shape[0]] = anomaly_samples
                anomaly_samples = _t
            mixture = simple_mix_procedure(bg_audio, anomaly_samples, anti_clipping=anti_clipping,
                                           magic_anticlipping_factor=magic_anticlipping_factor)
            e_idx.append(i)  # index of the events
        l.append(mixture)
    return torch.stack(l, dim=0), e_idx

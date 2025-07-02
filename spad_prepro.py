import os
from pathlib import Path
import librosa
import glob

import numpy as np
import torch

from tqdm import tqdm
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import dcase_util

from src.mix_utils import do_mix_on_segments

CITIES = ['milan', 'lyon', 'vienna', 'barcelona', 'helsinki', 'paris', 'lisbon', 'london', 'stockholm', 'prague']
PLACES = ['street_traffic', 'metro_station', 'park', 'metro', 'street_pedestrian', 'shopping_mall', 'tram', 'bus',
          'public_square', 'airport']


def seed_torch(seed=42):
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def delete_aux_audio_files(folder):
    for dirpath, d, f in os.walk(folder):
        for file in f:
            if file[-5:].lower() != 'a.wav':
                os.remove(os.path.join(dirpath, file))


def get_audio_files(folder):
    files = {}
    for dirpath, d, f in os.walk(folder):
        for file in f:
            if file[-5:].lower() == 'a.wav':
                parts = file.split('-')
                if parts[0] not in files:
                    files[parts[0]] = {}
                if parts[1] not in files[parts[0]]:
                    files[parts[0]][parts[1]] = []
                files[parts[0]][parts[1]].append(os.path.join(dirpath, file))
    return files


def download_data(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    tau_folder = data_dir + 'TAU-urban-acoustic-scenes-2020-mobile-development/audio'
    if not Path(tau_folder).is_dir():
        db = dcase_util.datasets.TAUUrbanAcousticScenes_2020_Mobile_DevelopmentSet(data_path=data_dir)
        # db.initialize()
        # db.show()
        delete_aux_audio_files(tau_folder)
    else:
        print('Background audio files exist, skip init')
    if not Path('data/TUT-rare-sound-events-2017-development').is_dir():
        events_link = 'https://zenodo.org/record/401395/files/TUT-rare-sound-events-2017-development.source_data_events.zip'
        file_name = 'TUT-rare-sound-events-2017-development.source_data_events.zip'
        import requests, zipfile
        response = requests.get(events_link, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True,
                            desc=' Downloading event audio files', ascii=True)
        with open(file_name, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        else:
            print(f'Unpacking zipped file into {data_dir}')
            zip_ref = zipfile.ZipFile(file_name)
            zip_ref.extractall(data_dir)
            zip_ref.close()
            os.remove(file_name)
    else:
        print('Event audio files exist, skip init')


def load_audio(files, place, city):
    pbar = tqdm(total=len(files[place]), desc=place + '-' + city + ' - loading audio files', ascii=True)
    tensor_audio = []
    for file in files[place][city]:
        (audio, _) = librosa.core.load(file, sr=44100, mono=True)
        # little hack since some files has little few more than that
        tensor_audio.append(torch.tensor(audio[:441000]))
        pbar.update(1)
    pbar.close()
    return torch.stack(tensor_audio, dim=0)


def load_events(data_path, dest_path, event_type):
    files = []
    for dirpath, d, f in os.walk(data_path):
        for file in f:
            if file[-4:].lower() == '.wav':
                files.append(os.path.join(dirpath, file))
    pbar = tqdm(total=len(files), desc=event_type + ' - loading audio files', ascii=True)
    audio_list = []
    for file in files:
        (audio, _) = librosa.core.load(file, sr=44100, mono=True)
        audio_list.append(torch.tensor(audio))
        pbar.update(1)
    pbar.close()
    dest_file_path = dest_path + '/' + event_type + '_44k_mono.pt'
    print(f'\tSaving at {dest_file_path}')
    torch.save(audio_list, dest_file_path)


def get_spect_tools(sample_rate=44100, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000):
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None

    # Spectrogram extractor
    spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                        win_length=window_size, window=window, center=center,
                                        pad_mode=pad_mode,
                                        freeze_parameters=True)

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                        top_db=top_db,
                                        freeze_parameters=True)
    return spectrogram_extractor, logmel_extractor


def segment_by_place_n_city(data_path, dest_path, spectrogram_extractor, logmel_extractor, places=PLACES, cities=CITIES,
                            segment_length=4096, suffix="wavenet_train_ds_spec.pt"):
    files = get_audio_files(data_path)
    for place in places:
        print(f'Working on {place}')
        dataset = []
        for city in cities:
            print(f'\tLoading audio files for ' + city)
            final_path = dest_path + '/' + str(segment_length) + '/train/' + place + '/'
            Path(final_path).mkdir(parents=True, exist_ok=True)
            audio_tensor = load_audio(files, place, city)
            samples, audio_samples = audio_tensor.shape
            over = audio_samples % segment_length
            audio_tensor = torch.narrow(audio_tensor, 1, 0, audio_samples - over)
            audio_segments = audio_tensor.contiguous().view(-1, segment_length)
            x = spectrogram_extractor(audio_segments)  # (batch_size, 1, time_steps, freq_bins)
            x = logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            dataset.append(x)
        dataset = torch.cat(dataset, dim=0)
        torch.save(dataset, final_path + suffix)


def generate_test_dataset(data_path, spectrogram_extractor, logmel_extractor, places=PLACES, segment_length=4096,
                          magic_anticlipping_factor=0.8, permute=False, ebrs=[-6, 0, 6], seed=0,
                          suffix='_44k_mono_spec.pt'):
    seed_torch(seed)
    for event_type in ['gunshot', 'glassbreak']:
        event_list = torch.load(data_path + '/events/' + event_type + '_44k_mono.pt')
        print(f'{event_type} - {len(event_list)}')
        for place in places:
            place_data = torch.load(data_path + '/raw_test/' + place + '/test_data_44k_mono.pt')
            final_path = data_path + '/' + str(segment_length) + '/test/' + str(seed) + '/' + place
            Path(final_path).mkdir(parents=True, exist_ok=True)
            for ebr in ebrs:
                print(f'{seed} : Working on {place}, with input {segment_length}, for {event_type} at {ebr}db')
                mixture, e_idx = do_mix_on_segments(place_data, event_list,
                                                    magic_anticlipping_factor=magic_anticlipping_factor,
                                                    ebr=ebr, seed=seed, seed_np=True, segment_length=segment_length,
                                                    permute=permute)
                x = spectrogram_extractor(mixture)  # (batch_size, 1, time_steps, freq_bins)
                if torch.isnan(x).any():
                    x = torch.tensor(np.nan_to_num(x.cpu().numpy()))
                mix_audio_features = logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
                _suffix = event_type + '_at_' + str(ebr) + '_on_' + place + suffix
                dest_file_path = final_path + '/mix_' + _suffix
                print(dest_file_path)
                torch.save({'data': mix_audio_features, 'index': e_idx}, dest_file_path)


if __name__ == '__main__':
    train_cities = CITIES[5:]
    test_cities = CITIES[:5]

    # seg_lengths = [4096, 16384]
    seg_lengths = [4096]

    spectrogram_extractor, logmel_extractor = get_spect_tools(window_size=256, hop_size=80)

    places = PLACES #['street_traffic', 'street_pedestrian', 'shopping_mall', 'public_square', 'airport']

    if False:
        download_data(data_dir='data/')

    data_path = 'data/TAU-urban-acoustic-scenes-2020-mobile-development/audio'
    events_base_path = 'data/TUT-rare-sound-events-2017-development/data/source_data/events/'
    dest_path = 'data/prepro'

    if False:
        _tmp = dest_path
        dest_path = dest_path + '/events'
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        for event in ['babycry', 'glassbreak', 'gunshot']:
            events_path = events_base_path + event
            load_events(events_path, dest_path=dest_path, event_type=event)
        dest_path = _tmp

    if False:
        suffix = "spad_train_ds_spec_256_80_64.pt"
        for sl in seg_lengths:
            segment_by_place_n_city(data_path, dest_path, spectrogram_extractor, logmel_extractor,
                                    places=places, cities=train_cities,
                                    segment_length=sl, suffix=suffix)

    if False:
        suffix = '_44k_mono_spec_256_80_64.pt'
        for sl in seg_lengths:
            for i in range(5):
                generate_test_dataset(dest_path, spectrogram_extractor, logmel_extractor, ebrs=[-6, 0, 6],
                                      places=places, segment_length=sl, seed=i, suffix=suffix)

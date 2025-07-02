import os
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.cae import ConvAutoencoder as CAE
from src.cae_mini import MiniConvAutoencoder as miniCAE
from src.fc_ae import Autoencoder as FC_AE

TEST_CITIES = ['milan', 'lyon', 'vienna', 'barcelona', 'helsinki']
TRAIN_CITIES = ['paris', 'lisbon', 'london', 'stockholm', 'prague']
PLACES = ['street_traffic', 'metro_station', 'park', 'metro', 'street_pedestrian', 'shopping_mall', 'tram', 'bus',
          'public_square', 'airport']

CUDA = torch.device('cuda')

kwargs = {}  # {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

def seed_torch(seed=42):
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def init_tb_logger(model, model_name, place, segment_length, single=True):
    from datetime import datetime
    date_time = datetime.now()
    model.eval()
    tb = SummaryWriter('cae/runs/' + place + '/' + model_name + '/' + str(segment_length) + '/' + date_time.strftime(
        "%d_%b_%H-%M-%S") + '/')
    return tb


def _train(model, train_loader, place, tb, epochs=250, device=CUDA):
    model.to(device);
    t_loss_history = []
    for epoch in tqdm(range(epochs), ascii=True):
        total_loss = 0
        for i, segments in enumerate(train_loader):
            segments = segments.to(device)
            # print(segments.shape)
            # exit(0)
            recon_loss = model.training_step(segments)
            total_loss += recon_loss
        t_loss_history.append(total_loss / (i + 1))
        tb.add_scalar(place + "/01_TrainLoss", total_loss / (i + 1), epoch)
    print('Epoch:{}, TrainLoss:{:.4f}'.format(epochs, float(t_loss_history[-1])))


def train(t_steps, data_path, places=PLACES, batch_size=128, epochs=100, segment_length=4096,
          dest_path='cae', suffix="wavenet_train_ds_spec.pt", model_name=''):
    global kwargs
    for place in places:
        print(f'Working on {place} with data {suffix}')
        # place_dir = data_path + '/' + str(segment_length) + '/train/' + place
        # dataset = torch.load(os.path.join(place_dir, suffix))
        place_dir = data_path / str(segment_length) / 'train' / place
        dataset = torch.load( place_dir / suffix)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True, **kwargs)
        model = CAE(t_steps=t_steps)
        tb = init_tb_logger(model, 'conv-spad', place, segment_length)
        _train(model, train_loader, place, tb, epochs)
        model.to('cpu')
        file_name = place + model_name + str(batch_size) + '_' + str(epochs) + '.pt'
        torch.save(model, dest_path + '/' + file_name)
        print(file_name)


if __name__ == '__main__':
    seed_torch(27)
    # data_path = os.path.join(os.path.dirname(__file__), 'data', 'prepro')
    data_path = Path('data/prepro')
    # seg_lengths = [4096, 16384, 44100]
    seg_lengths = [4096]

    t_steps = {'44100': 138, '88200': 276}
    t_steps_256_80 = {'4096': 52, '16384': 52}

    # places = ['metro', 'metro_station', 'park', 'tram', 'bus']
    places = ['street_traffic', 'street_pedestrian', 'shopping_mall', 'public_square', 'airport']
    if True:
        suffix = "wavenet_train_ds_spec_256_80_64.pt"   # window_size=256, hop_size=80, mel_bins=64
        for sl in seg_lengths:
            t_steps = t_steps_256_80[str(sl)]
            mel_bins = 64  # 32
            model_name = '_cae_' + str(sl) + '_256_80_64_'
            train(t_steps=t_steps, data_path=data_path, places=places, segment_length=sl,
                  epochs=100, model_name=model_name, suffix=suffix)

import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-15

class AE_Encoder(nn.Module):
    def __init__(self, input_size=4096, enc_input_size=2048, z_dim=64):
        super(AE_Encoder, self).__init__()
        self.pre_encoder = nn.Linear(input_size, enc_input_size)
        self.encoder = nn.Sequential(
            nn.Linear(enc_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim)
        )

    def forward(self, x):
        z = self.encoder(self.pre_encoder(x))
        return z


class AE_Decoder(nn.Module):
    def __init__(self, output_size=4096, dec_output_size=2048, z_dim=64):
        super(AE_Decoder, self).__init__()
        self.num_features = dec_output_size
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, dec_output_size)
        )
        self.post_decoder = nn.Linear(dec_output_size, output_size)

    def forward(self, z):
        x_hat = self.decoder(z)
        return self.post_decoder(x_hat)

class Autoencoder(nn.Module):
    def __init__(self, input_size=2048, z_dim=64, gen_lr=0.0001):
        super(Autoencoder, self).__init__()
        self.z_dim = z_dim
        self.encoder = AE_Encoder(input_size=input_size, z_dim=z_dim)
        self.decoder = AE_Decoder(output_size=input_size, z_dim=z_dim)
        self.init_optim(gen_lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # AE
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def init_optim(self, gen_lr):
        # encode/decode optimizers
        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=gen_lr)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=gen_lr)


    def training_step(self, x):
        x = torch.flatten(x, start_dim=1)
        x_hat, _ = self(x)
        assert not torch.isnan(x_hat).any()
        recon_loss = self.criterion(x_hat, x)  # [b, 2048]
        recon_loss.backward()
        self.optim_decoder.step()
        self.optim_encoder.step()
        # Zero Grad
        self.optim_decoder.zero_grad()
        self.optim_encoder.zero_grad()
        return recon_loss.item()

    # def test_step(self, x):
    #     self.eval()
    #     with torch.no_grad():
    #         x = torch.flatten(x, start_dim=1)
    #         x_hat, _ = self(x)
    #         assert not torch.isnan(x_hat).any()
    #         loss = self.criterion(x_hat, x)
    #     return loss.item()

    def test_step(self, x):
        self.eval()
        with torch.no_grad():
            b, c, t, m = x.shape
            x = torch.flatten(x, start_dim=1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
        return x_hat.view(b, c, t, m), z


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_memory(model):
    from hurry.filesize import size
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    return size(mem)


if __name__ == '__main__':
    # b = torch.rand(2, 2048)
    # encoder = AE_Encoder(input_size=2048, z_dim=64)
    # z = encoder(b)
    # decoder = AE_Decoder(output_size=2048, z_dim=64)
    # x_hat = decoder(z)

    # input_lengths = {'4096': 832, '16384': 3328, '44100': 8832, '88200': 17664}
    # b_4096 = {'x': torch.rand(1, 1, 13, 64), 'ts': 13}
    # b_16384 = {'x': torch.rand(1, 1, 52, 64), 'ts': 52}
    # b_44100 = {'x': torch.rand(1, 1, 138, 64), 'ts': 138}
    # b_88200 = {'x': torch.rand(1, 1, 276, 64), 'ts': 276}
    # inputs = {'4096': b_4096, '16384': b_16384, '44100': b_44100, '88200': b_88200}
    input_lengths = {'4096': 1664}
    b_4096 = {'x': torch.rand(1, 1, 52, 32), 'ts': 52}
    # b_16384 = {'x': torch.rand(1, 1, 52, 32), 'ts': 52}
    inputs = {'4096': b_4096} #, '16384': b_16384}
    from pthflops import count_ops
    l = []
    s = []
    mel_bins = 32
    for k in input_lengths:
        _aae = Autoencoder(input_size=input_lengths[k], z_dim=64)
        s.append(f'for input {k} we have {count_parameters(_aae)} params and memory {count_memory(_aae)}')
        enc_ops, _ = count_ops(_aae.encoder, torch.flatten(inputs[k]['x'], start_dim=1))
        # z = _aae.encoder(torch.flatten(inputs[k]['x'], start_dim=1))
        x_hat, z = _aae(torch.flatten(inputs[k]['x'], start_dim=1))
        dec_ops, _ = count_ops(_aae.decoder, z)
        l.append(enc_ops + dec_ops)

    for flops in l:
        print(f'Tot FLOPs {flops}')

    for ss in s:
        print(ss)


    # aae = Autoencoder(input_size=2048, z_dim=64)
    # print(count_parameters(aae))

    # print(f'{aae.training_step(b)}')
    # print(f'{aae.test_step(b)}')
    # #torch.onnx.export(ae, b, "aae.onnx")

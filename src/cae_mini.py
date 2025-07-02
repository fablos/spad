import torch
from torch import nn


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_seq(seq):
    for l in seq:
        if hasattr(l, 'weight'):
            init_layer(l)


class CAE_Encoder(nn.Module):
    def __init__(self, t_steps=13, z_dim=64, mel_bins=64):
        super(CAE_Encoder, self).__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=(1, 2), padding=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, 5),
        )
        self.lin_enc = nn.Sequential(
            nn.Linear(t_steps * 64, z_dim)
            # nn.Linear(252, 128),
            # nn.Linear(128, z_dim)
        )
        init_bn(self.bn0)
        init_seq(self.lin_enc)
        init_seq(self.encoder)

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        _x = self.encoder(x)
        (x1, _) = torch.max(_x, dim=3)
        x2 = torch.mean(_x, dim=3)
        _x = x1 + x2
        _x = torch.flatten(_x, start_dim=1)
        z = self.lin_enc(_x)
        return z


class CAE_Decoder(nn.Module):
    def __init__(self, t_steps=13, z_dim=64, mel_bins=64):
        super(CAE_Decoder, self).__init__()
        self.t_steps = t_steps
        self.mel_bins = mel_bins
        self.lin_dec = nn.Sequential(
            nn.Linear(z_dim, t_steps * mel_bins)
            # nn.Linear(z_dim, 128),
            # nn.Linear(128, 252)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=(1, 2), padding=1, output_padding=(0, 1)),
        )
        init_seq(self.lin_dec)
        init_seq(self.decoder)

    def forward(self, z):
        x_hat = self.lin_dec(z)
        x_hat = x_hat.view(z.shape[0], self.mel_bins, self.t_steps, 1)
        return self.decoder(x_hat)


class MiniConvAutoencoder(nn.Module):
    def __init__(self, t_steps=13, z_dim=64, mel_bins=64, gen_lr=0.0001):
        super(MiniConvAutoencoder, self).__init__()
        self.t_steps = t_steps
        self.z_dim = z_dim
        self.encoder = CAE_Encoder(t_steps=t_steps, mel_bins=mel_bins)
        self.decoder = CAE_Decoder(t_steps=t_steps, mel_bins=mel_bins)
        self.init_optim(gen_lr)
        self.criterion = nn.MSELoss()

    def init_optim(self, gen_lr):
        # encode/decode optimizers
        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=gen_lr)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=gen_lr)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def training_step(self, x):
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

    def test_step(self, x):
        self.eval()
        with torch.no_grad():
            x_hat, _ = self(x)
            assert not torch.isnan(x_hat).any()
            loss = self.criterion(x_hat, x)
        return loss.item()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_memory(model):
    from hurry.filesize import size
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    return size(mem)

if __name__ == '__main__':
    b_4096 = {'x': torch.rand(1, 1, 65, 32), 'ts': 65}
    # b_16384 = {'x': torch.rand(1, 1, 52, 64), 'ts': 52}
    # b_44100 = {'x': torch.rand(1, 1, 138, 64), 'ts': 138}
    # b_88200 = {'x': torch.rand(1, 1, 276, 64), 'ts': 276}

    for b in [b_4096]: #, b_16384, b_44100, b_88200]:
        cae = MiniConvAutoencoder(t_steps=b['ts'], mel_bins=32)
        print(count_parameters(cae))
        print(count_memory(cae))
        x_hat, z = cae(b['x'])
        # print(f'{cae.training_step(b["x"])}')
        # print(f'{cae.test_step(b["x"])}')
        # print(x.shape, torch.flatten(x, start_dim=1).shape)
        # print(x.shape, z.shape, x_hat.shape)


    # torch.onnx.export(ae, b, "ae.onnx")

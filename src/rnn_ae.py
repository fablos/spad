import torch
import torch.nn as nn


class SignalEncoder(nn.Module):
    def __init__(self, n_features=64, hidden_dim=32, rnn_num_layers=2, bidir=True, dropout=0.2):
        super(SignalEncoder, self).__init__()
        self.net = nn.LSTM(input_size=n_features, hidden_size=hidden_dim,
                           num_layers=rnn_num_layers, bidirectional=bidir,
                           dropout=dropout, batch_first=True)
        self.hidden_dim = hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.D = 2 if bidir else 1

    def forward(self, x):
        h0 = torch.randn(self.D * self.rnn_num_layers, x.shape[0], self.hidden_dim, device=x.device)
        c0 = torch.randn(self.D * self.rnn_num_layers, x.shape[0], self.hidden_dim, device=x.device)
        _, (hn, cn) = self.net(x, (h0, c0))
        hn = torch.permute(hn, (1, 0, 2)).contiguous()
        cn = torch.permute(cn, (1, 0, 2)).contiguous()
        hc = torch.cat([hn, cn], dim=1)
        return torch.flatten(hc, start_dim=1)


class SignalDecoder(nn.Module):
    def __init__(self, num_steps=52, n_features=64, hidden_dim=32, rnn_num_layers=2, bidir=True, dropout=0.2):
        super(SignalDecoder, self).__init__()
        self.num_steps = num_steps
        self.n_features = n_features
        self.D = 2 if bidir else 1
        self.net = nn.LSTM(input_size=n_features, hidden_size=hidden_dim,
                           num_layers=rnn_num_layers, bidirectional=True,
                           dropout=dropout, batch_first=True)
        self.hidden_dim = hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.teacher_forcing_prob = 0.5

    def forward(self, code, x):
        use_teacher_forcing = False  # random.random() < self.teacher_forcing_prob
        output = torch.zeros((x.size(0), 1, self.n_features), device=x.device)
        outputs = torch.zeros((x.size(0), self.num_steps, self.n_features), device=x.device)
        _code = code.view(code.size(0), 2, self.D * self.rnn_num_layers, -1)
        hidden = (torch.permute(_code[:, 0, :, :], (1, 0, 2)).contiguous(),
                  torch.permute(_code[:, 1, :, :], (1, 0, 2)).contiguous())  # h0, c0
        # loss = 0.0
        for i in range(self.num_steps):
            output, hidden = self.net(output, hidden)
            # loss += self.criterion(output[:, 0, :], x[:, i, :])
            outputs[:, i, :] = output[:, 0, :]  # .detach()
            output = x[:, i, :].detach().unsqueeze(dim=1) if use_teacher_forcing else output.detach()
        return outputs


class RnnAutoencoder(nn.Module):
    def __init__(self, gen_lr=0.001):
        super(RnnAutoencoder, self).__init__()
        self.encoder = SignalEncoder()
        self.decoder = SignalDecoder()
        self.init_optim(gen_lr)
        self.criterion = nn.L1Loss()

    def init_optim(self, gen_lr):
        # encode/decode optimizers
        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=gen_lr)
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=gen_lr)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z, x)
        return x_hat, z

    def training_step(self, x):
        self.train()
        x = x.squeeze(dim=1)
        x_hat, _ = self(x)
        assert not torch.isnan(x_hat).any()
        recon_loss = self.criterion(x_hat, x)  # [b, 52, 64]
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

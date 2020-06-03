
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, n_classes, n_dim=128, latent_dim=32, win_size=3):
        super().__init__()

        self.n_dim = n_dim
        self.latent_dim = latent_dim
        self.win_size = win_size
        
        if win_size == 1:
            shape = [1+n_classes, latent_dim]
        else:
            shape = [1+n_classes, win_size, latent_dim]
        
        # Mean and variance are trained parameters
        self.mean = torch.nn.Parameter(torch.randn(*shape))
        self.logvar = torch.nn.Parameter(torch.randn(*shape))

    def forward(self, i, z=None):
        batch_size = i.shape[0]

        mu = stack_indices(self.mean, i)
        logvar = stack_indices(self.logvar, i)

        # Generate some value in N(mu, std)
        std = torch.exp(0.5 * logvar)

        if z is None:
            z = torch.randn_like(std)

        v = mu + std * z

        return v, mu, logvar

class Decoder(nn.Module):
    def __init__(self, n_dim=128, latent_dim=32, win_size=3):
        super().__init__()

        self.n_dim = n_dim
        self.latent_dim = latent_dim
        self.win_size = win_size

        layers = []
        while latent_dim < n_dim:
            layers += [
                nn.Linear(latent_dim, 2*latent_dim),
                nn.ReLU()
            ]
            latent_dim *= 2

        assert latent_dim == n_dim

        layers.append(nn.Linear(n_dim, n_dim))

        self.dense = nn.Sequential(*layers)

    def forward(self, x, bound=None):
        y = self.dense(x)
        if bound is not None:
            assert len(bound) == 2
            y = torch.clamp(y, *bound)

        return y

class Model(nn.Module):
    def __init__(self, n_classes, phonemes, n_dim=128, latent_dim=32, win_size=3):
        super().__init__()

        assert n_classes is not None

        self.n_dim = n_dim
        self.latent_dim = latent_dim
        self.win_size = win_size

        self.phonemes = phonemes

        encoder = Encoder(
                n_classes=n_classes,
                n_dim=n_dim,
                latent_dim=latent_dim,
                win_size=win_size)

        decoder = Decoder(n_dim=n_dim, latent_dim=latent_dim, win_size=win_size)

        self.encoder = encoder
        self.decoder = decoder

        self.loss_fn = nn.SmoothL1Loss(reduction='sum')

    def forward(self, x, use_mean=False, z=None):
        if len(x.shape) == 1:
            y, mu, logvar = self.encoder(x, z=None)
            x = mu if use_mean else y

            return self.decoder(x), mu, logvar
        else:
            return self.decoder(x)

    def fabricate(self, x, **args):
        """ Constructs a raw spectrogram of human speech by
        directly calling self.forward()
        """
        y, _, _ = self.forward(x, **args)
        return torch.transpose(torch.reshape(-1, self.n_dim))

    def training_loss(self, x, i):
        # Generate a reconstruction
        y, mu, logvar = self(i.to(device))

        recon_loss = self.loss_fn(x.to(device), y)
        
        # Based on https://github.com/pytorch/examples/blob/master/vae/main.py
        # https://arxiv.org/abs/1312.6114
        kl_loss = -0.5 * torch.sum(logvar - mu ** 2 - logvar.exp())

        return recon_loss + kl_loss

    def training_step(self, x, i, opt):
        loss = self.training_loss(x, i)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.cpu().item()

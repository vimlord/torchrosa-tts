import torch
from torch import nn

device = torch.device('cpu')#torch.device('cuda' if torch.has_cuda else 'cpu')

def stack_indices(arr, idxs):
    return torch.stack([
        arr[i] for i in idxs
    ], axis=0)

class Embedding(nn.Module):
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

class Encoder(nn.Module):
    def __init__(self, n_dim=128, latent_dim=32, win_size=3):
        super().__init__()

        self.n_dim = n_dim
        self.latent_dim = latent_dim
        self.win_size = win_size

        if n_dim > latent_dim:
            self.dropout = nn.Dropout(p=0.2)

            layers_train = []
            layers_test = []

            while latent_dim < n_dim:
                lyr = [
                    nn.Linear(n_dim, n_dim // 2),
                    nn.ReLU(),
                ]
                layers_train += lyr
                layers_test += lyr
                layers_train += [self.dropout]

                n_dim //= 2

            assert latent_dim == n_dim
            
            self.dense_train = nn.Sequential(*layers_train)
            self.dense_test = nn.Sequential(*layers_test)
        
        self.mean = nn.Linear(n_dim, n_dim)
        self.logvar = nn.Linear(n_dim, n_dim)

    def forward(self, x, z=None, train=True):
        if self.n_dim == self.latent_dim:
            y = x
        elif train:
            y = self.dense_train(x)
        else:
            y = self.dense_test(x)

        mu = self.mean(y)
        logvar = self.logvar(y)

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

        if n_dim > latent_dim:
            self.dropout = nn.Dropout(p=0.2)

            layers_train = []
            layers_test = []
            while latent_dim < n_dim:
                lyr = [
                    nn.ReLU(),
                    nn.Linear(latent_dim, 2*latent_dim),
                ]
                layers_train += lyr
                layers_test += lyr
                layers_train += [self.dropout]

                latent_dim *= 2

            assert latent_dim == n_dim
            
            self.dense_train = nn.Sequential(*layers_train)
            self.dense_test = nn.Sequential(*layers_test)

    def forward(self, x, bound=None, train=True):
        if self.n_dim == self.latent_dim:
            y = x
        elif train:
            y = self.dense_train(x)
        else:
            y = self.dense_test(x)

        if bound is not None:
            assert len(bound) == 2
            y = torch.clamp(y, *bound)

        return y

class Model(nn.Module):
    def __init__(self, phonemes, n_dim=128, latent_dim=32, win_size=3):
        super().__init__()

        if '_' in phonemes:
            assert phonemes[0] == '_'

        self.n_dim = n_dim
        self.latent_dim = latent_dim
        self.win_size = win_size

        self.phonemes = phonemes

        embedding = Embedding(
                n_classes=len([p for p in phonemes if p != '_']),
                n_dim=n_dim,
                latent_dim=latent_dim,
                win_size=win_size)

        encoder = Encoder(
                n_dim=n_dim,
                latent_dim=latent_dim,
                win_size=win_size)

        decoder = Decoder(
                n_dim=n_dim,
                latent_dim=latent_dim,
                win_size=win_size)

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

        self.loss_fn = nn.SmoothL1Loss(reduction='sum')

    def forward(self, x, use_mean=False, z=None, train=True):
        if len(x.shape) == 1:
            y, mu, logvar = self.embedding(x, z=z)
        else:
            y, mu, logvar = self.encoder(x, train=train)

        x = mu if use_mean else y

        return self.decoder(x, train=train), mu, logvar

    def fabricate(self, idxs, lens=None, **args):
        """ Constructs a raw spectrogram of human speech by
        directly calling self.forward()
        """
        y, _, _ = self(idxs, train=False, use_mean=True, **args)
        
        if lens is not None:
            y = torch.cat([v[:l] for v, l in zip(y, lens)], dim=0)

        return y.reshape((-1, self.n_dim)).permute(1, 0)

    def training_loss(self, x, i, train=True):
        x = x.to(device)
        i = i.to(device)

        # Generate a reconstruction
        y, mu1, logvar1 = self(x, train=train)

        # Generate a fabrication
        _, mu2, logvar2 = self.embedding(i)
        
        recon_loss = self.loss_fn(x, y) + self.loss_fn(mu1, mu2) + self.loss_fn(logvar1, logvar2)
        
        if self.latent_dim < self.n_dim:
            kl_loss = -0.5 * torch.sum(logvar1 - mu1 ** 2 - logvar1.exp())
            return recon_loss + kl_loss
        else:
            return recon_loss

    def training_step(self, x, i, opt):
        loss = self.training_loss(x, i)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.cpu().item()



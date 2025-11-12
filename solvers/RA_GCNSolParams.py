import copy
import numpy as np
import torch
import torch.utils.data as tud
import torch.optim.optimizer as too
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as to
import typing as t
import tqdm
import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as sol
from utils.legacy_code import plot_loss_history


class RA_GCNWholeSolver(sol.WholeSolver):
    def __init__(self, params: sol.SolverBaseParams):
        super().__init__(params)

    def _train(self, classifier: nn.Module, cls_optimizers: [too.Optimizer], generator: nn.Module, gen_optimizers: [too.Optimizer],
               epoch_No: int) -> (t.Optional[float], t.Optional[float]):
        classifier.train()
        for cls_optimizer in cls_optimizers:
            cls_optimizer.zero_grad()

        if epoch_No % self.args.cls_gen_train_ratio == 0 and isinstance(classifier, nw.AdversarialNetwork) and isinstance(generator, nw.AdversarialNetwork):
            for gen_optimizer in gen_optimizers:
                gen_optimizer.zero_grad()

            cls_scores = classifier.forward(self.all_data)
            gen_scores = generator.forward(self.all_data)

            cls_loss = classifier.adversarial_loss(cls_scores, self.train_labels, gen_scores.detach())
            cls_loss_value = cls_loss.item()

            if np.isnan(cls_loss_value):
                self.logger.warning(f"cls_loss_value is {cls_loss_value}.")
                return None, None

            cls_loss.backward()
            for cls_optimizer in cls_optimizers:
                cls_optimizer.step()
        else:
            cls_loss_value = None

        cls_scores = classifier.forward(self.all_data)
        gen_scores = generator.forward(self.all_data)

        gen_loss = generator.adversarial_loss(gen_scores, self.train_labels, cls_scores.detach())
        gen_loss_value = gen_loss.item()

        if np.isnan(gen_loss_value):
            self.logger.warning(f"gen_loss_value is {gen_loss_value}.")
            return None, None

        gen_loss.backward()

        with torch.no_grad():
            for param in generator.parameters():
                param.grad = -param.grad
        for gen_optimizer in gen_optimizers:
            gen_optimizer.step()

        return cls_loss_value, gen_loss_value


class RA_GCNBatchSolver(sol.BatchSolver):
    def _train(self, classifier: nw.AdversarialNetwork, cls_optimizers: [too.Optimizer], generator: nw.AdversarialNetwork, gen_optimizers: [too.Optimizer],
               epoch_No: int) -> (t.Optional[float], t.Optional[float]):
        subset = tud.Subset(self.dataset, classifier.current_indices)
        data_loader = tud.DataLoader(subset, batch_size=self.args.batch_size, shuffle=False)
        classifier.train()

        cls_loss_value = 0.0
        gen_loss_value = 0.0
        i = -1
        for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(data_loader)):
            assert isinstance(batch_y, torch.LongTensor)

            if batch_y.size(0) == 1:
                i -= 1
                continue

            for cls_optimizer in cls_optimizers:
                cls_optimizer.zero_grad()

            if isinstance(classifier, nw.NetworkBase):
                classifier.batch_index = i
            if isinstance(generator, nw.NetworkBase):
                generator.batch_index = i

            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.to(ag.Arguments.device)
            labels = batch_y.to(ag.Arguments.device)

            for cls_optimizer in cls_optimizers:
                cls_optimizer.zero_grad()

            cls_scores = classifier.forward(batch_x)
            gen_scores = generator.forward(batch_x)

            cls_loss = classifier.adversarial_loss(cls_scores, labels, gen_scores.detach())
            cls_loss_value2 = cls_loss.item()

            if np.isnan(cls_loss_value2):
                self.logger.warning(f"Epoch {epoch_No}, Iteration {i+1}: cls_loss_value is {cls_loss_value2}.")
                return None, None

            cls_loss.backward()
            for cls_optimizer in cls_optimizers:
                cls_optimizer.step()

            cls_scores = classifier.forward(batch_x).detach()
            gen_loss = generator.adversarial_loss(gen_scores, labels, cls_scores)

            gen_loss_value2 = gen_loss.item()
            if np.isnan(gen_loss_value2):
                self.logger.warning(f"Epoch {epoch_No}, Iteration {i+1}: gen_loss_value is {gen_loss_value2}.")
                return None, None

            gen_loss_value += gen_loss_value2

            gen_loss.backward()

            with torch.no_grad():
                for param in generator.parameters():
                    param.grad = -param.grad
            for gen_optimizer in gen_optimizers:
                gen_optimizer.step()

            if self.args.print_every_iterations > 0 and (i+1) % self.args.print_every_iterations == 0:
                msg = f"Epoch {epoch_No}, Iteration {i+1}: gen_loss_value={gen_loss_value:.5f}"
                self.logger.info(msg)

        cls_loss_value /= (i + 1)
        gen_loss_value /= (i + 1)
        return cls_loss_value, gen_loss_value
    

class RA_GCNSolParams(sol.MLPSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.RA_GCNArgs):
        super().__init__(all_samples, logger, args)
        self.args = args
        self.solver_name = "RA_GCNBatchSolver" if not args.learning_mode else "RA_GCNWholeSolver"

    def _preprocess_data(self) -> t.Union[np.ndarray, torch.Tensor]:
        """
        obsolete, won't be invoked.
        """
        dataset_params = ds.BrainDatasetParams(self.args.dataset_dir, self.all_samples, self.args.txt_indices)
        dataset = ds.BrainDataset(dataset_params)
        # `shuffle` of DataLoader must be False when use data_loader.dataset.init_indices().
        data_loader = tud.DataLoader(dataset, batch_size=self.args.VAE_batch_size, shuffle=False)

        print(f"Computing features of {self.args.txt_indices}...")
        latent_dim = int((self.args.hidden_dim * 4 * 11 * 13 * 11) / 1000)
        all_data = reduce_dimensionality(data_loader, 1, latent_dim, self.args.hidden_dim, self.args.VAE_num_epochs,
                                         self.args.optimizer.learning_rate, self.args.log_dir_path)
        return all_data

    def _prepare_network(self):
        create_cls_params = nw.GCNParams(self.args, self.all_samples)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)

        args = copy.copy(self.args)
        args.optimizer.learning_rate = args.gen_lr_rate
        create_gen_params = nw.GCNParams(args, self.all_samples)
        self.create_gen_factory = nw.NetworkFactory(create_gen_params, self.args.net_name + "WeightGenerator")


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.fc_avg = nn.Linear(hidden_dim * 4 * 11 * 13 * 11, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 4 * 11 * 13 * 11, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        avg_logits = self.fc_avg(x)
        var_logits = self.fc_var(x)
        return avg_logits, var_logits


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(latent_dim, self.hidden_dim * 4 * 11 * 13 * 11)
        self.de_conv1 = nn.ConvTranspose3d(self.hidden_dim * 4, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.de_conv2 = nn.ConvTranspose3d(self.hidden_dim * 2, self.hidden_dim, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.de_conv3 = nn.ConvTranspose3d(self.hidden_dim, output_dim, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.pad = nn.ReflectionPad3d((0, 0, 1, 1, 0, 0))

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.shape[0], self.hidden_dim * 4, 11, 13, 11)
        x = F.relu(self.de_conv1(x))
        x = F.relu(self.de_conv2(x))
        x = self.de_conv3(x)
        x = self.pad.forward(x)
        x = torch.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)

    @staticmethod
    def __re_parameterize(avg_logits, var_logits):
        std = torch.exp(0.5 * var_logits)
        eps = torch.randn_like(std)
        return avg_logits + eps * std

    def forward(self, x):
        avg_logits, var_logits = self.encoder(x)
        z = VAE.__re_parameterize(avg_logits, var_logits)
        recon_x = self.decoder(z)
        return recon_x, avg_logits, var_logits


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.hidden_dim = 16
        self.conv1 = nn.Conv3d(input_channels, self.hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(self.hidden_dim, self.hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(self.hidden_dim * 2, self.hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(self.hidden_dim * 4 * 11 * 13 * 11, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.shape[0], -1)
        x = torch.sigmoid(self.fc(x))
        return x


class VAE_GAN(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE_GAN, self).__init__()
        self.vae = VAE(input_dim, latent_dim, hidden_dim)
        self.discriminator = Discriminator(input_dim)

    def forward(self, x):
        recon_x, avg_logits, var_logits = self.vae(x)
        disc_real = self.discriminator(x)
        disc_fake = self.discriminator(recon_x)
        return recon_x, avg_logits, var_logits, disc_real, disc_fake


def vae_loss(recon_x, x, avg_logits, var_logits):
    MSE = F.mse_loss(recon_x, x, reduction="mean")
    KLD = -0.5 * torch.sum(1 + var_logits - avg_logits.pow(2) - var_logits.exp())
    return MSE + KLD


def gan_loss(disc_real, disc_fake):
    real_labels = torch.ones_like(disc_real)
    fake_labels = torch.zeros_like(disc_fake)
    real_loss = F.binary_cross_entropy(disc_real, real_labels)
    fake_loss = F.binary_cross_entropy(disc_fake, fake_labels)
    return real_loss + fake_loss


def reduce_dimensionality(data_loader: tud.DataLoader, input_dim: int, latent_dim: int, hidden_dim: int, epochs: int, learning_rate: float, log_dir_path: str)\
        -> [[float]]:

    model = VAE_GAN(input_dim, latent_dim, hidden_dim)
    optimizer = to.Adam(model.parameters(), learning_rate)

    model = model.to(ag.Arguments.device)
    loss_history = []
    net_state = None
    min_loss = np.inf

    for epoch in range(epochs):
        model.train()
        for i, (data, _) in enumerate(data_loader):
            assert isinstance(data, torch.Tensor), "Dismiss a warning."
            matrices = data.to(ag.Arguments.device)

            recon_matrices, avg_logits, var_logits, disc_real, disc_fake = model.forward(matrices)

            vae_loss_value = vae_loss(recon_matrices, matrices, avg_logits, var_logits)
            gan_loss_value = gan_loss(disc_real, disc_fake)
            loss = vae_loss_value + gan_loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_history.append(loss_value)
            print(f"Iteration {i+1}/{len(data_loader)}: loss={loss_value:.4f}")

            if loss_value < min_loss:
                min_loss = loss_value
                net_state = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch+1}/{epochs}]: loss={loss_history[-1]:.4f}")

    plot_loss_history(loss_history, log_dir_path, "VAE_GAN_loss_history")

    features = []
    model.load_state_dict(net_state)
    model.eval()
    with torch.no_grad():
        for matrices, _ in data_loader:
            matrices = matrices.to(ag.Arguments.device)
            avg_logits, var_logits = model.vae.encoder(matrices)
            features.append(avg_logits.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features

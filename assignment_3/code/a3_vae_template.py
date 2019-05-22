import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
#import numpy as np
import math
from scipy.stats import norm
import itertools
import numpy as np

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20,input_dim=784):
        super().__init__()

        #Normalize the input between -1 and 1 as in Max Welling's paper
        #self.tanh = nn.Tanh()

        #Use a ReLU layer and Adam to speed up convergence as stated in Carl Doersh's Tutorial
        self.relu = nn.ReLU()

        #Then network has 2 outputs, parameters for the Gaussian, but weights are seperate for each parameter
        self.first_mu_layer = nn.Linear(input_dim, hidden_dim)
        self.second_mu_layer = nn.Linear(hidden_dim, z_dim)

        self.first_std_layer = nn.Linear(input_dim, hidden_dim)
        self.second_std_layer = nn.Linear(hidden_dim, z_dim)


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        first_mu_output = self.first_mu_layer(input)
        after_relu = self.relu(first_mu_output )
        mean = self.second_mu_layer(after_relu)

        first_std_output = self.first_std_layer(input)
        after_relu = self.relu(first_std_output )
        std = self.second_std_layer(after_relu)

        #Enforce that std should be positive
        std = nn.functional.relu(std)


        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20,input_dim=784):
        super().__init__()


        #Normalize the input between -1 and 1 as in Max Welling's paper
        #self.tanh = nn.Tanh()  # torch.tanh

        #Use a ReLU layer and Adam to speed up convergence as stated in Carl Doersh's Tutorial paper
        self.relu = nn.ReLU()

        #Then network has 1 output, the parameter of probability for the Bernoulli
        #And assume that output is the same dimension as input
        self.first_mu_layer = nn.Linear(z_dim, hidden_dim)
        self.second_mu_layer = nn.Linear(hidden_dim, input_dim)


    def forward(self, input):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """

        first_mu_output = self.first_mu_layer(input)
        after_relu = self.relu(first_mu_output )
        mean = self.second_mu_layer(after_relu)

        #Constrain values to 0 and 1
        mean = torch.sigmoid(mean)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20,device=torch.device('cpu'),input_dim=784):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim,input_dim)
        self.decoder = Decoder(hidden_dim, z_dim,input_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        mean_encoder,std_encoder = self.encoder.forward(input)
        epsilon = torch.randn(1,self.z_dim )#device=self.device)
        #epsilon = torch.randn(mean_encoder.shape)  # device=self.device)
        z = mean_encoder + epsilon * std_encoder

        output_decoder = self.decoder.forward(z)
        batch_size = mean_encoder.shape[0]

        #print("VAE forward")
        #In the case of Bernoulli distribution, the reconstriction error is the cross entropy and we have binary MNIST
        #We can also use the built in BCEwithlogits loss that will apply sigmoid for us and log-sum-exp trick to make it stable
        #However, results can be different/slightly worse as first take mean over this reconstruction instead over the sum of reconstruction and reg, can still use the formula
        #criterion = nn.BCEWithLogitsLoss()
        #recon = criterion(output_decoder,input)

        #Add some small value to denominator of log to avoid instability of inf
        recon = -1 * (input * torch.log(output_decoder) + (1 - input) * torch.log(1-output_decoder))
        recon = torch.sum(recon,dim=1)

        #KL divergence between q(z|x) and p(z). Add some small value to denominator of log to avoid instability of inf
        D_qp_univariate = torch.log(1/ (std_encoder + 0.0000001) + (std_encoder**2 + mean_encoder**2) / 2 - 0.5)
        reg = torch.sum(D_qp_univariate, dim=1)

        #average_negative_elbo =
        average_negative_elbo = (torch.sum(recon,dim=0) + torch.sum(reg,dim=0)) / batch_size

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        #Get z from standard normal
        z = torch.randn(n_samples, self.z_dim)
        z = z.to(device)

        #Decode the sampled z
        sampled_ims = self.decoder.forward(z)
        im_means = torch.sum(sampled_ims,dim=0)/ len(sampled_ims)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    model.zero_grad()
    average_epoch_elbo = 0

    for step, batch_inputs in enumerate(data):

        #print(batch_inputs.shape) #Batch_inputs: [128,1,28,28] shape
        #Convert to ]128,784]
        batch_inputs = batch_inputs.reshape(batch_inputs.shape[0],-1)
        batch_inputs = batch_inputs.to(device)

        average_negative_elbo = model.forward(batch_inputs)
        average_epoch_elbo += average_negative_elbo.item()

        # train the model
        if model.training:
            model.zero_grad()
            average_negative_elbo.backward()
            optimizer.step()

    average_epoch_elbo /= len(data)

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """

    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')

    plt.title("ELBO values while training", fontsize=18, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename)

def plot_samples(sampled,x_y_image_dim,grid_size,epoch):

    sampled = sampled.view(-1,1,x_y_image_dim,x_y_image_dim)
    grid = make_grid(sampled,grid_size)
    save_image(grid,filename ="samples_epoch_" + str(epoch) + ".png")

def plot_manifold(x_y_image_dim, manifold_size,model):

    #grid = torch.linspace(0, 1, manifold_size)
    #samples = [norm.ppf(x) for x in grid for y in grid]
    #grid = torch.linspace(norm.ppf(0),norm.ppf())

    grid = np.linspace(0.05, 0.95, manifold_size)
    grid = list(map(norm.ppf,grid))
    z = torch.tensor(list(itertools.product(grid,grid)) )
    print(z.shape)
    #z = torch.stack(z)
    print(z.shape)
    z = z.view(-1,1,20,20)
    z = z.to(model.device)
    output = model.decoder.forward(z)

    #output = output.view(-1,1,x_y_image_dim,x_y_image_dim)
    #print(output)
    #print(output.shape)
    manifold = make_grid(output,manifold_size)
    save_image(manifold.t(1,2, 0),filename ="Manifold.png")
    f

def main(device):

    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim,device=device,input_dim = ARGS.input_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # size by size for grid or manifold
    grid_size = 4
    manifold_size = 20

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        x_and_y_dim = int(math.sqrt(ARGS.input_dim))

        sampled, im_means = model.sample(grid_size*grid_size)
        plot_samples(sampled,x_and_y_dim,grid_size,epoch)

        plot_manifold(x_and_y_dim, manifold_size, model)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    plot_manifold(x_and_y_dim,mani_size,model)


    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--input_dim', default=784, type=int,
                        help='dimensionality of input')
    parser.add_argument('--device', default='cpu', type=str,
                        help='cpu or cuda')

    ARGS = parser.parse_args()

    torch.manual_seed(42)
    #np.random.seed(42)
    if 'cuda' in ARGS.device.lower() and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    main(device)

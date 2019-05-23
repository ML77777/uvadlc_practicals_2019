import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.first_layer = nn.Linear( args.latent_dim, 128)
        self.first_activation = nn.LeakyReLU(0.2)

        self.second_layer = nn.Linear( 128, 256)
        self.second_norm =  nn.BatchNorm1d(256)
        self.second_activation = nn.LeakyReLU(0.2)

        self.third_layer = nn.Linear(256, 512)
        self.third_norm =  nn.BatchNorm1d(512)
        self.third_activation = nn.LeakyReLU(0.2)

        self.fourth_layer = nn.Linear(512, 1024)
        self.fourth_norm =  nn.BatchNorm1d(1024)
        self.fourth_activation = nn.LeakyReLU(0.2)

        self.output_layer = nn.Linear(1024, 784)
        self.output_activation = nn.Tanh()   #Use tanh as for better training as stated by the tips on piazza https://github.com/soumith/ganhacks

        self.layers = [self.first_layer,self.first_activation, self.second_layer, self.second_norm,self.second_activation, self.third_layer, self.third_norm,self.third_activation,
                       self.fourth_layer, self.fourth_norm, self.fourth_activation, self.output_layer,self.output_activation]

        #With dropout tip
        #self.layers = [self.first_layer,self.first_activation,nn.Dropout(0.5),
        # self.second_layer, self.second_norm,self.second_activation,nn.Dropout(0.5),
        #  self.third_layer, self.third_norm,self.third_activation,nn.Dropout(0.5),
        #  self.fourth_layer, self.fourth_norm, self.fourth_activation,nn.Dropout(0.5),
        #  self.output_layer,self.output_activation]


    def forward(self, z):
        # Generate images from z
        #pass

        input = z
        for layer in self.layers:
            input = layer(input)

        generated_output = input

        return generated_output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.first_layer = nn.Linear(784, 512)
        self.first_activation = nn.LeakyReLU(0.2)

        self.second_layer = nn.Linear( 512, 256)
        self.second_activation = nn.LeakyReLU(0.2)

        self.output_layer = nn.Linear(256, 1)
        self.output_activation = nn.Sigmoid()   #Use tanh as for better training as stated by the tips on piazza https://github.com/soumith/ganhacks

        self.layers = [self.first_layer,self.first_activation, self.second_layer,self.second_activation,self.output_layer,self.output_activation]


    def forward(self, img):
        # return discriminator score for img


        input = img
        for layer in self.layers:
            input = layer(input)

        img_score = input

        return img_score


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    device = args.device

    args.n_epochs = 5
    tanh = nn.Tanh()
    prev_acc = 0
    freeze_acc = 0.75

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            #imgs.cuda()

            # Train Generator
            # ---------------
            z = torch.randn( batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            prediction_d = discriminator(gen_imgs)
            gen_loss = -torch.log(prediction_d)
            gen_loss = gen_loss.sum()

            gen_loss.clamp(min=0.00000001, max=100)
            #print(" Gen loss" ,gen_loss)

            optimizer_G.zero_grad()
            gen_loss.backward()
            optimizer_G.step()


            # Train Discriminator
            # -------------------
            #optimizer_D.zero_grad()


            actual_batch_size = imgs.shape[0]
            imgs = imgs.view(actual_batch_size,784)

            z = torch.randn( actual_batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            gen_imgs.detach()

            pred_fake = discriminator(gen_imgs)

            norm_imgs = tanh(imgs) #Normalize between -1 and 1.
            pred_real = discriminator(norm_imgs)

            if prev_acc < freeze_acc:
                disc_loss = - (torch.log(pred_real) + torch.log(1 - pred_fake))
                disc_loss = disc_loss.sum()
                disc_loss.clamp(min=0.00000001, max=100)
                #print(" Disc loss", disc_loss)

                optimizer_D.zero_grad()
                disc_loss.backward()
                optimizer_D.step()

            #Accuracy
            TP = (pred_real >= .5).sum().item()
            TN = (pred_fake < .5).sum().item()
            total = (2*actual_batch_size)
            prev_acc = (TP + TN) / total

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True)
                save_image(gen_imgs[:25].view(-1, 1, 28, 28),
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
        print("Epoch " + str(epoch) + " Done")


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(args.device)
    discriminator = Discriminator().to(args.device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--device', default='cpu', type=str,
                        help='cpu or cuda')
    args = parser.parse_args()

    #model = Generator()
    #model.load_state_dict(torch.load("mnist_generator.pt"))
    #z = torch.randn(args.batch_size, args.latent_dim)
    #gen_imgs = model(z)
    #save_image(gen_imgs[:25].view(-1, 1, 28, 28),
    #           'test.png',
    #           nrow=5, normalize=True)

    main()

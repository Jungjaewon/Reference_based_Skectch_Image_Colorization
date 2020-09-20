
import os
import time
import operator
import datetime
import torch
import torch.nn as nn

from torchvision import models
from functools import reduce
from model import Generator
from model import Discriminator

from res_model import Generator as rGenerator
from res_model import Discriminator as rDiscriminator

from unet import UNet

from torchvision.models import vgg19
from torchvision.utils import save_image


vgg_activation = dict()

def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output.detach()

    return hook

class Solver(object):

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.data_loader = data_loader
        self.img_size    = config['MODEL_CONFIG']['IMG_SIZE']
        assert self.img_size in [256]

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_recon = config['TRAINING_CONFIG']['LAMBDA_G_RECON']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_gp     = config['TRAINING_CONFIG']['LAMBDA_GP']
        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']
        self.mse_loss = torch.nn.MSELoss()

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.model_type = config['TRAINING_CONFIG']['MODEL_TYPE']

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'true'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'true'

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # vgg activation
        self.target_layer = ['relu_3', 'relu_8', 'relu_17', 'relu_26', 'relu_35']
        self.gt_activation = dict()
        self.model_activation = dict()

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()

    def build_model(self):

        if self.model_type == 'unet':
            self.G = UNet(n_channels=3, n_classes=3)
            self.D = rDiscriminator(spec_norm=self.d_spec).to(self.gpu)
        elif self.model_type == 'res':
            self.G = rGenerator(spec_norm=self.g_spec).to(self.gpu)
            self.D = rDiscriminator(spec_norm=self.d_spec).to(self.gpu)
        else:
            self.G = Generator(spec_norm=self.g_spec).to(self.gpu)
            self.D = Discriminator(spec_norm=self.d_spec).to(self.gpu)

        self.vgg = vgg19(pretrained=True)
        for layer in self.target_layer:
            self.vgg.features[int(layer[-1])].register_forward_hook(get_activation(layer))

        """
        self.vgg.features[3].register_forward_hook(get_activation('relu_3'))
        self.vgg.features[8].register_forward_hook(get_activation('relu_8'))
        self.vgg.features[17].register_forward_hook(get_activation('relu_17'))
        self.vgg.features[26].register_forward_hook(get_activation('relu_26'))
        self.vgg.features[35].register_forward_hook(get_activation('relu_35'))
        """

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def train(self):

        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        _, fixed_color, fixed_sketch = next(data_iter)

        fixed_sketch = fixed_sketch.to(self.gpu)
        fixed_color = fixed_color.to(self.gpu)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_time = time.time()
        print('Start training...')
        for e in range(self.epoch):

            for i in range(iterations):
                try:
                    _, color, sketch = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, color, sketch = next(data_iter)

                color = color.to(self.gpu)
                sketch = sketch.to(self.gpu)

                loss_dict = dict()
                #print(target_images[:, 0].size()) # torch.Size([batch_size, ch, H, W])
                if (i + 1) % self.d_critic == 0:
                    out_score = self.D(color)

                    d_loss_real = -torch.mean(out_score)

                    x_fake = self.G(sketch)
                    out_score = torch.mean(x_fake.detach())
                    d_loss_fake = torch.mean(out_score)

                    # Compute loss for gradient penalty.
                    alpha = torch.rand(color.size(0), 1, 1, 1).to(self.gpu)
                    x_hat = (alpha * color.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                    out_src = self.D(x_hat)
                    d_loss_gp = self.gradient_penalty(out_src, x_hat)

                    # Backward and optimize.
                    d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_gp * d_loss_gp
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Logging.
                    loss_dict['D/loss_real'] = d_loss_real.item()
                    loss_dict['D/loss_fake'] = d_loss_fake.item()
                    loss_dict['D/loss_gp'] = d_loss_gp.item()

                if (i + 1) % self.g_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(sketch)
                    out_src = self.D(x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_recon = self.mse_loss(x_fake, color)

                    # Backward and optimize.
                    g_loss = self.lambda_g_fake * g_loss_fake + self.lambda_g_recon * g_loss_recon
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss_dict['G/loss_fake'] = g_loss_fake.item()
                    loss_dict['G/loss_recon'] = g_loss_recon.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    image_report = list()
                    image_report.append(fixed_sketch)
                    image_report.append(fixed_color)
                    image_report.append(self.G(fixed_sketch))
                    x_concat = torch.cat(image_report, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(e + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))
            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(e + 1))
                D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(e + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

        print('Training is finished')

    def test(self):
        pass


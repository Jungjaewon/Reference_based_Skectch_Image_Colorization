
import os
import time
import datetime
import torch
import torch.nn as nn
import glob
import os.path as osp

from model import Generator
from model import Discriminator
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
        self.lambda_g_style = config['TRAINING_CONFIG']['LAMBDA_G_SYTLE']
        self.lambda_g_percep = config['TRAINING_CONFIG']['LAMBDA_G_PERCEP']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gp     = config['TRAINING_CONFIG']['LAMBDA_GP']
        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']
        self.mse_loss = nn.MSELoss()

        self.triplet = config['TRAINING_CONFIG']['TRIPLE_LOSS'] == 'True'
        self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        assert self.gan_loss in ['lsgan', 'wgan', 'vanilla']

        if self.triplet:
            self.triplet_loss = nn.TripletMarginLoss(margin=config['TRAINING_CONFIG']['LAMBDA_TR'])
            self.triple_margin = config['TRAINING_CONFIG']['LAMBDA_TR']
            #self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.scaled_dot_product(),
            # margin=config['TRAINING_CONFIG']['LAMBDA_TR'])
            # triplet_loss(anchor, positive, negative)

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']
        if self.gan_loss == 'lsgan':
            self.adversarial_loss = torch.nn.MSELoss()
        elif self.gan_loss =='vanilla':
            self.adversarial_loss = torch.nn.BCELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'True'

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # vgg activation
        self.target_layer = ['relu_3', 'relu_8', 'relu_13', 'relu_17']

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
        self.G = Generator(spec_norm=self.g_spec).to(self.gpu)
        self.D = Discriminator(spec_norm=self.d_spec, LR=0.2).to(self.gpu)
        self.vgg = vgg19(pretrained=True)
        for layer in self.target_layer:
            self.vgg.features[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))
        self.vgg.to(self.gpu)

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

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def scaled_dot_product(self, a, b):
        #https://github.com/pytorch/pytorch/issues/18027
        channel = a.size(1)
        scale_factor = torch.sqrt(torch.cuda.FloatTensor([channel]))
        out = torch.bmm(a.view(self.batch_size, 1, channel), b.view(self.batch_size, channel , 1)).reshape(-1)
        #out = torch.div(out, scale_factor)
        return out

    def triple_loss_custom(self, anchor, positive, negative, margin=12):
        distance = self.scaled_dot_product(anchor, positive) - self.scaled_dot_product(anchor, negative) + margin
        #print('distance : ',distance)
        #print('torch.max(distance, torch.zeros_like(distance)) : ', torch.max(distance, torch.zeros_like(distance)))
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*-G.ckpt'))

        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(epoch))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(epoch))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.G.to(self.gpu)
        self.D.to(self.gpu)
        return epoch

    def image_reporting(self, fixed_sketch, fixed_reference, fixed_elastic_reference, epoch, postfix=''):
        image_report = list()
        image_report.append(fixed_sketch.expand_as(fixed_reference))
        image_report.append(fixed_elastic_reference)
        image_report.append(fixed_reference)
        fake_result, _ = self.G(fixed_elastic_reference, fixed_sketch)
        image_report.append(fake_result)
        x_concat = torch.cat(image_report, dim=3)
        sample_path = os.path.join(self.sample_dir, '{}-images{}.jpg'.format(epoch, postfix))
        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

    def train(self):

        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        _, fixed_elastic_reference, fixed_reference, fixed_sketch = next(data_iter)

        splited_fixed_sketch = list(torch.chunk(fixed_sketch, self.batch_size, dim=0))
        first_fixed_sketch = splited_fixed_sketch[0]
        del splited_fixed_sketch[0]
        splited_fixed_sketch.append(first_fixed_sketch)
        shifted_fixed_sketch = torch.cat(splited_fixed_sketch, dim=0)

        fixed_sketch = fixed_sketch.to(self.gpu)
        fixed_reference = fixed_reference.to(self.gpu)
        fixed_elastic_reference = fixed_elastic_reference.to(self.gpu)
        shifted_fixed_sketch = shifted_fixed_sketch.to(self.gpu)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_epoch = self.restore_model()
        start_time = time.time()
        print('Start training...')
        for e in range(start_epoch, self.epoch):

            for i in range(iterations):
                try:
                    _, elastic_reference, reference, sketch = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, elastic_reference, reference, sketch = next(data_iter)

                elastic_reference = elastic_reference.to(self.gpu)
                reference = reference.to(self.gpu)
                sketch = sketch.to(self.gpu)

                loss_dict = dict()
                if (i + 1) % self.d_critic == 0:

                    fake_images, _ = self.G(elastic_reference, sketch)
                    d_loss = None

                    if self.gan_loss in ['lsgan', 'vanilla']:
                        real_score = self.D(torch.cat([reference, sketch], dim=1))
                        fake_score = self.D(torch.cat([fake_images.detach(), sketch], dim=1))
                        d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
                        d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake
                    elif self.gan_loss == 'wgan':
                        real_score = self.D(torch.cat([reference, sketch], dim=1))
                        fake_score = self.D(torch.cat([fake_images.detach(), sketch], dim=1))
                        d_loss_real = -torch.mean(real_score)
                        d_loss_fake = torch.mean(fake_score)
                        alpha = torch.rand(reference.size(0), 1, 1, 1).to(self.gpu)
                        x_hat = (alpha * reference.data + (1 - alpha) * fake_images.data).requires_grad_(True)
                        out_src = self.D(x_hat)
                        d_loss_gp = self.gradient_penalty(out_src, x_hat)
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_d_gp * d_loss_gp

                    # Backward and optimize.
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Logging.
                    loss_dict['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
                    loss_dict['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()

                    if self.gan_loss == 'wgan':
                        loss_dict['D/loss_pg'] = self.lambda_d_gp * d_loss_gp.item()

                if (i + 1) % self.g_critic == 0:
                    fake_images, q_k_v_list = self.G(elastic_reference, sketch)
                    fake_score = self.D(torch.cat([fake_images, sketch], dim=1))
                    if self.gan_loss in ['lsgan', 'vanilla']:
                        g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
                    elif self.gan_loss == 'wgan':
                        g_loss_fake = - torch.mean(fake_score)
                    else:
                        pass
                    g_loss_recon = self.l1_loss(fake_images, reference)

                    fake_activation = dict()
                    real_activation = dict()

                    self.vgg(reference)
                    for layer in self.target_layer:
                        fake_activation[layer] = vgg_activation[layer]
                    vgg_activation.clear()

                    self.vgg(fake_images)
                    for layer in self.target_layer:
                        real_activation[layer] = vgg_activation[layer]
                    vgg_activation.clear()

                    g_loss_style = 0
                    g_loss_percep = 0

                    for layer in self.target_layer:
                        g_loss_percep += self.l1_loss(fake_activation[layer], real_activation[layer])
                        g_loss_style += self.l1_loss(self.gram_matrix(fake_activation[layer]), self.gram_matrix(real_activation[layer]))

                    if self.triplet:
                        anchor = q_k_v_list[0].view(self.batch_size, -1)
                        positive = q_k_v_list[1].contiguous().view(self.batch_size, -1)
                        negative = q_k_v_list[2].contiguous().view(self.batch_size, -1)
                        #g_loss_triple = self.triple_loss_custom(anchor=anchor, positive=positive, negative=negative, margin=self.triple_margin)
                        g_loss_triple = self.triplet_loss(anchor=anchor, positive=positive, negative=negative)

                    g_loss = self.lambda_g_fake * g_loss_fake + \
                    self.lambda_g_recon * g_loss_recon + \
                    self.lambda_g_percep * g_loss_percep + \
                    self.lambda_g_style * g_loss_style
                    if self.triplet:
                        g_loss += 1 * g_loss_triple

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss_dict['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                    loss_dict['G/loss_recon'] = self.lambda_g_recon * g_loss_recon.item()
                    loss_dict['G/loss_style'] = self.lambda_g_style * g_loss_style.item()
                    loss_dict['G/loss_percep'] = self.lambda_g_percep * g_loss_percep.item()
                    if self.triplet:
                        loss_dict['G/loss_triple'] = g_loss_triple.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    self.image_reporting(fixed_sketch, fixed_reference, fixed_elastic_reference, e + 1, postfix='')
                    self.image_reporting(shifted_fixed_sketch, fixed_reference, fixed_elastic_reference, e + 1, postfix='_shifted')
                    print('Saved real and fake images into {}...'.format(self.sample_dir))
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


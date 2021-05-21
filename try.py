import argparse
import time
import os


help_description = 'This script trains a FutureGAN model for video prediction according to the specified optional arguments.'

parser = argparse.ArgumentParser(description=help_description)

# general
parser.add_argument('--dgx', type=bool, default=False, help='set to True, if code is run on dgx, default=`False`')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus for (multi-)gpu training, default=1')
parser.add_argument('--random_seed', type=int, default=int(time.time()), help='seed for generating random numbers, default = `int(time.time())`')
parser.add_argument('--ext', action='append', default=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm'], help='list of strings of allowed file extensions, default=[`.jpg`, `.jpeg`, `.png`, `.ppm`, `.bmp`, `.pgm`]')
parser.add_argument('--use_ckpt', type=bool, default=False, help='continue training from checkpoint, default=`False`')

parser.add_argument('--ckpt_path', action='append', help='list of path(s) to training checkpoints to continue training or for testing, [0] Generator and [1] Discriminator, default=``')
parser.add_argument('--data_root', type=str, default='', help='path to root directory of training data (ex. -->path_to_dataset/train)')
parser.add_argument('--log_dir', type=str, default='./logs', help='path to directory of log files')
parser.add_argument('--experiment_name', type=str, default='mmnist', help='name of experiment (if empty, current date and time will be used), default=``')

parser.add_argument('--d_cond', type=bool, default=True, help='condition discriminator on input frames, default=`True`')
parser.add_argument('--nc', type=int, default=1, help='number of input image color channels, default=3')
parser.add_argument('--max_resl', type=int, default=64, help='max. frame resolution --> image size: max_resl x max_resl , default=128')
parser.add_argument('--nframes_in', type=int, default=10, help='number of input video frames in one sample, default=12')
parser.add_argument('--nframes_pred', type=int, default=10, help='number of video frames to predict in one sample, default=6')
# p100
parser.add_argument('--batch_size_table', type=dict, default={4:32, 8:16, 16:8, 32:4, 64:2, 128:1, 256:1, 512:1, 1024:1}, help='batch size table:{img_resl:batch_size, ...}, change according to available gpu memory')
## dgx
#parser.add_argument('--batch_size_table', type=dict, default={4:256, 8:128, 16:64, 32:32, 64:16, 128:8, 256:1, 512:1, 1024:1}, help='batch size table:{img_resl:batch_size, ...}, change according to available gpu memory')
parser.add_argument('--trns_tick', type=int, default=10, help='number of epochs for transition phase, default=10')
parser.add_argument('--stab_tick', type=int, default=10, help='number of epochs for stabilization phase, default=10')

# training
parser.add_argument('--nz', type=int, default=512, help='dimension of input noise vector z, default=512')
parser.add_argument('--ngf', type=int, default=512, help='feature dimension of final layer of generator, default=512')
parser.add_argument('--ndf', type=int, default=512, help='feature dimension of first layer of discriminator, default=512')

parser.add_argument('--loss', type=str, default='wgan_gp', help='which loss functions to use (choices: `gan`, `lsgan` or `wgan_gp`), default=`wgan_gp`')
parser.add_argument('--d_eps_penalty', type=bool, default=True, help='adding an epsilon penalty term to wgan_gp loss to prevent loss drift (eps=0.001), default=True')
parser.add_argument('--acgan', type=bool, default=False, help='adding a label penalty term to wgan_gp loss --> makes GAN conditioned on classification labels of dataset, default=False')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type, default=adam')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--lr_decay', type=float, default=0.87, help='learning rate decay at every resolution transition, default=0.87')

parser.add_argument('--lrelu', type=bool, default=True, help='use leaky relu instead of relu, default=True')
parser.add_argument('--padding', type=str, default='zero', help='which padding to use (choices: `zero`, `replication`), default=`zero`')
parser.add_argument('--w_norm', type=bool, default=True, help='use weight scaling, default=True')
parser.add_argument('--batch_norm', type=bool, default=False, help='use batch-normalization (not recommended), default=False')
parser.add_argument('--g_pixelwise_norm', type=bool, default=True, help='use pixelwise normalization for generator, default=True')
parser.add_argument('--d_gdrop', type=bool, default=False, help='use generalized dropout layer (inject multiplicative Gaussian noise) for discriminator when using LSGAN loss, default=False')
parser.add_argument('--g_tanh', type=bool, default=False, help='use tanh at the end of generator, default=False')
parser.add_argument('--d_sigmoid', type=bool, default=False, help='use sigmoid at the end of discriminator, default=False')
parser.add_argument('--x_add_noise', type=bool, default=False, help='add noise to the real image(x) when using LSGAN loss, default=False')
parser.add_argument('--z_pixelwise_norm', type=bool, default=False, help='if mode=`gen`: pixelwise normalization of latent vector (z), default=False')

# display and save
parser.add_argument('--tb_logging', type=bool, default=False, help='enable tensorboard visualization, default=True')
parser.add_argument('--update_tb_every', type=int, default=100, help='display progress every specified iteration, default=100')
parser.add_argument('--save_img_every', type=int, default=100, help='save images every specified iteration, default=100')
parser.add_argument('--save_ckpt_every', type=int, default=5, help='save checkpoints every specified epoch, default=5')


# parse and save training config
config = parser.parse_args()


log_dir = config.log_dir+'/'+ config.experiment_name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# save config settings to file
with open(log_dir+'/train_config.txt', 'w') as f:
    print('------------- training configuration -------------', file=f)
    for k, v in vars(config).items():
        print(('{}: {}').format(k, v), file=f)
    print(' ... loading training configuration ... ')
    print(' ... saving training configuration to {}'.format(f))
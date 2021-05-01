# -*- coding: utf-8 -*-

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
# import matplotlib.animation as animation
from . import model
import wandb
from .utils import general
from .utils.dataset import Dataset, load_imagelist_landmark



# Set random seed for reproducibility
# manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
# random.seed(manualSeed)
# torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=64, type=int, help="image patch size")
parser.add_argument('--hidden_size', default=64, type=int, help="network hidden unit size")
parser.add_argument('--num_input_channel', default=3, type=int, help="for RGB, it is 3")
parser.add_argument('--mask_l1_weight', default=1.0, type=float)
parser.add_argument('--modify_loss_weight', default=0.0, type=float)
parser.add_argument('--modify_as_syn', action="store_true")

parser.add_argument('--Dstep', default=1, type=int, help="Train Discriminator every X step")
parser.add_argument('--Gstep', default=1, type=int, help="Train Generator every X step")

parser.add_argument('--Dprestep', default=0, type=int, help="Pretrain Discriminator X step")
parser.add_argument('--Gprestep', default=0, type=int, help="Pretrain Generator X step")

parser.add_argument('--random_flip', action="store_true")
parser.add_argument('--random_rotate', action="store_true" )

parser.add_argument('--patch_size_ratio', default=10, type=int)
parser.add_argument('--patch_filter_magic', default=1.5, type=float)


parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--lr-beta', default=0.5, type=float)
parser.add_argument('--epoch', default=5, type=int, help="number of epoch to train")
parser.add_argument('--log_every', default=20, type=int, help="print log every X iteration")
parser.add_argument('--save_every', default=200, type=int, help="save model every X iteration")

parser.add_argument('--gpu', default='0', type=str, help="the id of gpu(s) to use")
parser.add_argument('--exp', default='default', type=str, help="name of the experiment")
parser.add_argument('--resume', default=None, type=str, help="resume from a checkpoint, please use a Gen ckpt path")
parser.add_argument('--debug', action="store_true")


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda:0" if len(args.gpu) > 0 else "cpu")
#device = "cpu"

BASE = general.get_project_base()

base_log_dir = os.path.join(BASE, '..', "log")
logdir, savedir= general.prepare_save_env(base_log_dir, args.exp, args)

# project = "syngan"
# run = wandb.init(
                # project=project, dir=logdir, group=args.exp,
                # config=vars(args), tags=["0-2,3", "new", "transforms"],
                # reinit=True, resume="allow", save_code=False, 
                # mode="online" if not args.debug else "offline",
                # )

########################################################################
datafolder = os.path.join(BASE, '..', 'data')
image_size = ( args.image_size, args.image_size )
imagefolder, acne_images, no_acne_images, landmark_dict = load_imagelist_landmark(datafolder)

tf = [ transforms.ToTensor() ]
if args.random_flip:
    tf.append(transforms.RandomHorizontalFlip())
    tf.append(transforms.RandomVerticalFlip())
if args.random_rotate:
    tf.append(transforms.RandomRotation(30))
tf = transforms.Compose(tf)

acne_dataset = Dataset(imagefolder, acne_images, landmark_dict, 
                image_size=image_size, preprocess=False, transforms=tf,
                size_ratio=args.patch_size_ratio, filter_magic=args.patch_filter_magic)

no_acne_dataset = Dataset(imagefolder, no_acne_images, landmark_dict, 
                image_size=image_size, preprocess=False, transforms=tf,
                size_ratio=args.patch_size_ratio, filter_magic=args.patch_filter_magic)

acne_dataloader = torch.utils.data.DataLoader(acne_dataset, args.batch_size, shuffle=True, num_workers=4)
no_acne_dataloader = torch.utils.data.DataLoader(no_acne_dataset, args.batch_size, shuffle=True, num_workers=4)
print("Number Acne Image", len(acne_dataset))
print("Number No-acne Image", len(no_acne_dataset))


#########################################################################

netG = model.Generator(args.hidden_size, num_input_channel=args.num_input_channel)
netD = model.Discriminator(args.hidden_size, num_input_channel=args.num_input_channel)

if args.resume is None:
    netG.apply(model.weights_init)
    netD.apply(model.weights_init)
    save_offset_step = 0
else:
    netG_ckpt = torch.load(args.resume, map_location="cpu")
    netG.load_state_dict(netG_ckpt)

    fname = os.path.basename(args.resume)
    save_offset_step = int(fname[4:-4])
    fname = "Dis" + fname[3:]
    dirname = os.path.dirname(args.resume)
    fname = dirname + '/' + fname
    netD_ckpt = torch.load(fname, map_location="cpu")
    netD.load_state_dict(netD_ckpt)

    # TODO - check argument match



netG.to(device)
netD.to(device)


######################################################################
# Loss Functions and Optimizers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize BCELoss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.lr_beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.lr_beta, 0.999))


# Lists to keep track of progress
global_step = 0
NUM_ITERATION = int(( len(acne_dataloader)+len(no_acne_dataloader)) / 2 )
acne_iter = iter(acne_dataloader)
no_acne_iter = iter(no_acne_dataloader)

def next_batch():
    global acne_iter, no_acne_iter
    try:
        a = acne_iter.next()
    except StopIteration:
        acne_iter = iter(acne_dataloader)
        a = acne_iter.next()
    try:
        n = no_acne_iter.next()
    except StopIteration:
        no_acne_iter = iter(no_acne_dataloader)
        n = no_acne_iter.next()

    # a = transform(a)
    # n = transform(n)
    return [a, n]

def process_image_for_log(name, orig, syn, modify, mask, num_log):
    # global run
    orig = general.to_numpy(orig)
    orig = np.transpose(orig, [0, 2, 3, 1])

    syn = general.to_numpy(syn)
    syn = np.transpose(syn, [0, 2, 3, 1])

    modify = general.to_numpy(modify)
    modify = np.transpose(modify, [0, 2, 3, 1])

    mask = general.to_numpy(mask)
    mask = np.transpose(mask, [0, 2, 3, 1])
    mask = mask[:, :, :, 0]

    # log = {}
    figs = []
    for i in range(num_log):
        fig, axes = plt.subplots(ncols=4, figsize=[16, 4])
        axes[0].imshow(orig[i])
        axes[0].set_title("origin")
        axes[1].imshow(syn[i])
        axes[1].set_title("syn")
        axes[2].imshow(modify[i])
        axes[2].set_title("modify")
        axes[3].imshow(mask[i], cmap='gray')
        axes[3].set_title("mask")

        figs.append(fig)

        # log["%s_%d" % (name, i)] = wandb.Image(fig)

    # plt.close('all')

    return figs

def train_D(acne_images, no_acne_images, log_prefix=""):
    log_dict = {}
    acne_label = torch.full((acne_images.shape[0],),    1.0, dtype=torch.float, device=device)
    no_acne_label = torch.full((no_acne_images.shape[0],), 1.0, dtype=torch.float, device=device)

    netD.zero_grad()
    # Train with real batch ------------------------------------
    acne_aprob, acne_qprob = netD(acne_images)
    no_acne_aprob, no_acne_qprob = netD(no_acne_images)

    # Calculate loss on all-real batch
    acne_label = acne_label.clone().fill_(1.0)
    no_acne_label = no_acne_label.clone().fill_(1.0)
    errD_quality = criterion(acne_qprob, acne_label) + criterion(no_acne_qprob, no_acne_label)
    log_dict[log_prefix+"/D-quality"] = errD_quality.item()

    acne_label = acne_label.clone().fill_(1.0)
    no_acne_label = no_acne_label.clone().fill_(0.0)
    errD_acne = criterion(acne_aprob, acne_label) + criterion(no_acne_aprob, no_acne_label)
    log_dict[log_prefix+"/D-acne"] = errD_acne.item()

    errD_real = errD_acne + errD_quality
    # Calculate gradients for D in backward pass
    errD_real.backward()
    # D_x = output.mean().item()

    # Train with all-fake batch ------------------------------
    # Generate batch of latent vectors
    synimage, modify, mask = netG(no_acne_images)
    if args.modify_as_syn:
        synimage = modify
    
    no_acne_label = no_acne_label.clone().fill_(0.0)
    syn_aprob, syn_qprob = netD(synimage.detach().clone())
    # Calculate D's loss on the synimage
    errD_fake_acne = criterion(syn_aprob, no_acne_label) # no acne
    errD_fake_quality = criterion(syn_qprob, no_acne_label) # fake image
    log_dict[log_prefix+"/D-fake-acne"] = errD_fake_acne.item()
    log_dict[log_prefix+"/D-fake-quality"] = errD_fake_quality.item()
    errD_fake = errD_fake_acne + errD_fake_quality

    # Calculate D's loss on the modify
    if args.modify_loss_weight > 0:
        mod_aprob, mod_qprob = netD(modify.detach().clone())
        errD_modi_acne = criterion(mod_aprob, no_acne_label) # no acne
        errD_modi_quality = criterion(mod_qprob, no_acne_label) # fake image
        log_dict[log_prefix+"/D-modi-acne"] = errD_modi_acne.item()
        log_dict[log_prefix+"/D-modi-quality"] = errD_modi_quality.item()
        err = errD_modi_acne + errD_modi_quality
        errD_fake += args.modify_loss_weight * err

    errD_fake.backward()
    # D_G_z1 = output.mean().item()
    # Add the gradients from the all-real and all-fake batches
    # errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    return synimage, modify, mask, log_dict


def train_G(no_acne_images, synimage=None, modify=None, mask=None, log_prefix=""):
    log_dict = {}

    netG.zero_grad()
    no_acne_label = torch.full((no_acne_images.shape[0],), 1.0, dtype=torch.float, device=device)

    # Since we just updated D, perform another forward pass of all-fake batch through D
    if synimage is None:
        synimage, modify, mask = netG(no_acne_images)
        if args.modify_as_syn:
            synimage = modify

    syn_aprob, syn_qprob = netD(synimage)
    # Calculate G's loss based on this output
    errG_acne = criterion(syn_aprob, no_acne_label) # has acne
    errG_quality = criterion(syn_qprob, no_acne_label) # is real image
    errG_mask = mask.mean() # mask L1 loss
    log_dict[log_prefix+"/G-acne"] = errG_acne.item()
    log_dict[log_prefix+"/G-quality"] = errG_quality.item()
    log_dict[log_prefix+"/G-mask"] = errG_mask.item()
    # Calculate gradients for G
    errG = errG_acne + errG_quality + args.mask_l1_weight * errG_mask

    if args.modify_loss_weight > 0:
        mod_aprob, mod_qprob = netD(modify)
        errG_modi_acne = criterion(mod_aprob, no_acne_label) # no acne
        errG_modi_quality = criterion(mod_qprob, no_acne_label) # fake image
        log_dict[log_prefix+"/G-modi-acne"] = errG_modi_acne.item()
        log_dict[log_prefix+"/G-modi-quality"] = errG_modi_quality.item()
        err = errG_modi_acne + errG_modi_quality
        errG += args.modify_loss_weight * err

    errG.backward()
    # D_G_z2 = output.mean().item()
    # Update G
    optimizerG.step()

    return log_dict


torch.autograd.set_detect_anomaly(True)


print("Pretrain Discriminator for %d steps" % args.Dprestep)
for i in range(args.Dprestep):
    if args.resume is not None:
        break
    acne_images, no_acne_images = next_batch()
    acne_images = acne_images.to(device)
    no_acne_images = no_acne_images.to(device)

    synimage, modify, mask, log_dict = train_D(acne_images, no_acne_images, log_prefix="Dpre")

    if i % args.log_every == 0:
        print("Dpre[%d]" % global_step)
        string = ""
        for k, v in log_dict.items():
            string += "%s:%.3f, " % (k.split('/')[1], v)
        print(string)

print("Pretrain Generator for %d steps" % args.Gprestep)
for i in range(args.Gprestep):
    if args.resume is not None:
        break
    acne_images, no_acne_images = next_batch()
    no_acne_images = no_acne_images.to(device)

    log_dict = train_G(no_acne_images, log_prefix="Gpre")

    if i % args.log_every == 0:
        print("Gpre[%d]" % global_step)
        string = ""
        for k, v in log_dict.items():
            string += "%s:%.3f, " % (k.split('/')[1], v)
        print(string)


print("Starting Training Loop...")
for epoch in range(args.epoch):
    # For each batch in the dataloader
    for i in range(NUM_ITERATION):
        # print(i)
        acne_images, no_acne_images = next_batch()
        acne_images = acne_images.to(device)
        no_acne_images = no_acne_images.to(device)

        log_dict = {}
        synimage = modify = mask = None
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        if global_step % args.Dstep == 0:
            synimage, modify, mask, dlog_dict = train_D(acne_images, no_acne_images, log_prefix="TrainLoss")

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if global_step % args.Gstep == 0:
            glog_dict = train_G(no_acne_images, synimage, modify, mask, log_prefix="TrainLoss")

        ##########################
        log_dict.update(dlog_dict)
        log_dict.update(glog_dict)
        # run.log(log_dict, step=global_step)

        if global_step % args.log_every == 0:
            # num_log = 3
            # log = process_image_for_log("TrainImage/eg", no_acne_images, synimage, modify, mask, num_log)
            # run.log(log)

            print("Iteration[%d]" % global_step)
            string = ""
            for k, v in log_dict.items():
                string += "%s:%.3f, " % (k.split('/')[1], v)
            print(string)
            
        if global_step % args.save_every == 0:
            g = netG.state_dict()
            d = netD.state_dict()
            torch.save(g, os.path.join(savedir, "Gen_%d.pth" % (save_offset_step + global_step) ))
            torch.save(d, os.path.join(savedir, "Dis_%d.pth" % (save_offset_step + global_step) ))

        global_step += 1



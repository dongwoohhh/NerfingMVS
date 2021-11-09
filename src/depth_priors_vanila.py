from glob import glob
import os, sys
sys.path.append('..')
import numpy as np
import torch
import cv2

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import pdb

from models.depth_priors.mannequin_challenge_model_meta import MannequinChallengeModelMeta, MannequinChallengeModel
from options import config_parser
from utils.io_utils import *
from utils.depth_priors_utils import *

from .dataloaders.dtu_mvsnerf import DTUMVSNeRFDataset
from .dataloaders.nerf_llff_data import NerfLLFFDataset
from .dataloaders.nerf_synthetic import NerfSyntheticDataset
from .dataloaders.ibrnet_collected import IBRNetCollectedDataset


from    copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_depth_model(args):
    """Instantiate depth model.
    """
    depth_model = MannequinChallengeModelMeta().to(device=device)
    #depth_model = MannequinChallengeModel()
    grad_vars = depth_model.parameters()
    
    #import pdb; pdb.set_trace()
    
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.meta_lr, betas=(0.9, 0.999))
    #optimizer = torch.optim.Adam(params=grad_vars, lr=args.depth_lrate, betas=(0.9, 0.999))

    start = 0
    """
    basedir = args.basedir
    expname = args.expname
    ckpt_path = os.path.join(basedir, expname, 'depth_priors', 'checkpoints')
    
    # Load checkpoints
    ckpts = [os.path.join(ckpt_path, f) for f in sorted(os.listdir(ckpt_path)) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        decay_rate = 0.1
        decay_steps = args.depth_N_iters
        
        new_lrate = args.depth_lrate * (decay_rate ** (start / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        depth_model.model.netG.load_state_dict(ckpt['netG_state_dict'])
    """
    return depth_model, start, optimizer


def update_inner_loop(vars, grad, lr):
    fast_weights = {}
    named_grad = dict(zip(vars.keys(), grad))

    for key in vars:
        fast_weights.update({key: vars[key] - lr * named_grad[key]})

    return fast_weights

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def create_training_dataset(args):
    train_dataset_names = ['dtu', 'synthetic', 'ibr']
    weights = [0.34, 0.33, 0.33]
    #dtu_train = DTUMVSNeRFDataset(args, mode="train")
    dtu_val = DTUMVSNeRFDataset(args, mode="val")

    #synthetic_train = NerfSyntheticDataset(args, mode="train")
    synthetic_val = NerfSyntheticDataset(args, mode="val")

    #llff_train = NerfLLFFDataset(args, mode="train")
    llff_val = NerfLLFFDataset(args, mode="val")

    #ibr_train = IBRNetCollectedDataset(args, mode="train")
    ibr_val = IBRNetCollectedDataset(args, mode="val")

    dataset_dict = {
        'dtu': DTUMVSNeRFDataset,
        'synthetic': NerfSyntheticDataset,
        'llff': NerfLLFFDataset,
        'ibr': IBRNetCollectedDataset,
    }
    train_datasets = []
    train_weights_samples = []
    for training_dataset_name, weight in zip(train_dataset_names, weights):
        train_dataset = dataset_dict[training_dataset_name](args, mode='train')
        train_datasets.append(train_dataset)
        num_samples = len(train_dataset)
        weight_each_sample = weight / num_samples
        train_weights_samples.extend([weight_each_sample]*num_samples)


    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_weights = torch.from_numpy(np.array(train_weights_samples))
    train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights))

    val_dataset = torch.utils.data.ConcatDataset([dtu_val, synthetic_val, llff_val, ibr_val])


    return train_dataset, val_dataset, train_sampler

def train(args):
    print('Depths prior training begins !')
    
    # Dataloader.

    #train_dataset = DTUMVSNeRFDataset(args, mode="train")
    #val_dataset = DTUMVSNeRFDataset(args, mode="val")
    train_dataset, val_dataset, train_sampler = create_training_dataset(args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.n_tasks,
                                               num_workers=24,#args.workers,
                                               pin_memory=False,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                               num_workers=1,#args.workers,
                                               pin_memory=False,
                                               shuffle=True)
    train_loader = iter(cycle(train_loader))
    val_loader_iterator = iter(cycle(val_loader))

    # Model.
    depth_model, global_step_depth, meta_optimizer = create_depth_model(args)

    save_dir = os.path.join(args.basedir, args.expname, 'depth_priors')

    # Summary.
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'summary'))

    update_step = args.update_step

    i_batch = 0
    n_training_steps = 150000
    vis_step = args.vis_step
    save_step = 1000
    depth_model.train()
    start = global_step_depth + 1

    while global_step_depth < n_training_steps - start:
        #for train_data in train_loader:
        train_data = next(train_loader)
        query_images = train_data["query_images"].to(device=device)[:, 0]
        query_depths = train_data["query_depths"].to(device=device)[:, 0]
        query_masks = train_data["query_masks"].to(device=device)[:, 0]

        query_sz = query_images.shape[1]
        #import pdb; pdb.set_trace()
        # Forward query images.
        query_preds = depth_model(query_images, vars=None)
        loss_q = compute_depth_loss(query_preds, query_depths, query_masks)
        #grad = torch.autograd.grad(loss, depth_model.parameters())

        # optimize theta parameters
        meta_optimizer.zero_grad()
        loss_q.backward()
        meta_optimizer.step()

        global_step_depth += 1

        if global_step_depth % 10 == 0:
            print(global_step_depth, loss_q.item())
            writer.add_scalar("Loss/train", loss_q.item(), global_step_depth)

        if global_step_depth % save_step == 0:
            # Summary writers
            path = os.path.join(save_dir, 'checkpoints', '{:06d}.tar'.format(global_step_depth))
            torch.save({
                'global_step': global_step_depth,
                'net_state_dict': depth_model.state_dict(),
                'optimizer_state_dict': meta_optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if global_step_depth % vis_step == 0:
            # vis training images
            #import pdb; pdb.set_trace()
            vis_func(query_images[0], query_preds[0], 'train', save_dir, global_step_depth)

            # vis test images
            #torch.cuda.empty_cache()
            #net = deepcopy(depth_model)
            depth_model.eval()
            val_data = next(val_loader_iterator)

            query_images_i = val_data["query_images"].to(device=device)[0]
            query_depths_i = val_data["query_depths"].to(device=device)[0]
            query_masks_i = val_data["query_masks"].to(device=device)[0]

            with torch.no_grad():
                query_preds_i = depth_model(query_images_i, vars=None)
                loss_val = compute_depth_loss(query_preds_i.unsqueeze(0), query_depths_i, query_masks_i)

                vis_func(query_images_i[0], query_preds_i, 'val'.format(update_step), save_dir, global_step_depth)
                writer.add_scalar("Loss/val", loss_val.item(), global_step_depth)

                #del net
    print('depths prior training done!')


def vis_func(image, depth, name, save_dir, step):
    image_np = image.cpu().detach().numpy()
    image_np = image_np.transpose((1, 2, 0))
    image_np = np.uint8(image_np * 255)

    depth_np = depth.cpu().detach().numpy()
    depth_color = visualize_depth(depth_np)

    cv2.imwrite(os.path.join(save_dir, 'results', '{}_{}_image.png'.format(step, name)), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, 'results', '{}_{}_depth.png'.format(step, name)), depth_color)



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)

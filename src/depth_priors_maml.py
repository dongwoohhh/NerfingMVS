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


def train(args):
    print('Depths prior training begins !')
    
    # Dataloader.
    train_dataset = DTUMVSNeRFDataset(args, mode="train")
    val_dataset = DTUMVSNeRFDataset(args, mode="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.n_tasks,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               shuffle=True)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               shuffle=True)
    val_loader_iterator = iter(cycle(val_loader))
    # Model.
    #depth_model, global_step_depth, optimizer_depth = create_depth_model(args)
    depth_model, global_step_depth, meta_optimizer = create_depth_model(args)

    save_dir = os.path.join(args.basedir, args.expname, 'depth_priors')

    update_step = args.update_step

    i_batch = 0
    n_training_steps = 150000
    vis_step = args.vis_step
    save_step = 1000
    depth_model.train()
    start = global_step_depth + 1

    while global_step_depth < n_training_steps - start:
        for train_data in train_loader:

            support_images = train_data["support_images"].to(device=device)
            support_depths = train_data["support_depths"].to(device=device)
            support_masks = train_data["support_masks"].to(device=device)

            query_images = train_data["query_images"].to(device=device)
            query_depths = train_data["query_depths"].to(device=device)
            query_masks = train_data["query_masks"].to(device=device)

            task_num = support_images.shape[0]
            query_sz = query_images.shape[1]

            losses_q = [0 for _ in range(update_step + 1)]
            # Inner loop
            for i_task in range(task_num):
                support_images_i = support_images[i_task]
                support_depths_i = support_depths[i_task]
                support_masks_i = support_masks[i_task]

                query_images_i = query_images[i_task]
                query_depths_i = query_depths[i_task]
                query_masks_i = query_masks[i_task]
                #import pdb; pdb.set_trace()
                """
                sum = 0
                for t in depth_model.parameters():
                    sum+= t.sum()
                print(sum)
                """
                # 1. run the i-th task and compute loss for k=0
                support_preds_i = depth_model(support_images_i, vars=None)
                support_loss_i = compute_depth_loss(support_preds_i, support_depths_i, support_masks_i)
                grad = torch.autograd.grad(support_loss_i, depth_model.parameters())
                fast_weights = update_inner_loop(depth_model.vars, grad, args.update_lr)

                # this is the loss and accuracy before first update.
                with torch.no_grad():
                    query_preds_i = depth_model (query_images_i, vars=depth_model.vars)
                    query_loss_i = compute_depth_loss(query_preds_i, query_depths_i, query_masks_i)
                    losses_q[0] += query_loss_i
                
                # this is the loss and accuracy after the first update.
                with torch.no_grad():
                    query_preds_i = depth_model (query_images_i, vars=fast_weights)
                    query_loss_i = compute_depth_loss(query_preds_i, query_depths_i, query_masks_i)
                    losses_q[1] += query_loss_i
                
                for k in range(1, update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    support_preds_i = depth_model(support_images_i, vars=fast_weights)
                    support_loss_i = compute_depth_loss(support_preds_i, support_depths_i, support_masks_i)
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(support_loss_i, fast_weights.values())
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = update_inner_loop(depth_model.vars, grad, args.update_lr)

                    query_preds_i = depth_model(query_images_i, vars=fast_weights)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    query_loss_i = compute_depth_loss(query_preds_i, query_depths_i, query_masks_i)
                    losses_q[k + 1] += query_loss_i
                """
                sum = 0
                for t in depth_model.parameters():
                    sum+= t.sum()
                print(sum)
                sum = 0
                for t in fast_weights:
                    sum+= fast_weights[t].sum()
                print(sum)
                """

            # end of all tasks
            # sum over all losses on query set across all tasks
            loss_q = losses_q[-1] / task_num
            
            # optimize theta parameters
            meta_optimizer.zero_grad()
            loss_q.backward()
            meta_optimizer.step()

            global_step_depth += 1

            if global_step_depth % 10 == 0:
                print(global_step_depth, loss_q.item())

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
                vis_func(query_images_i[0], query_preds_i[0], 'train', save_dir, global_step_depth)

                # vis test images
                #torch.cuda.empty_cache()
                net = deepcopy(depth_model)
                net.eval()
                val_data = next(val_loader_iterator)
                
                support_images_i = val_data["support_images"].to(device=device)[0]
                support_depths_i = val_data["support_depths"].to(device=device)[0]
                support_masks_i = val_data["support_masks"].to(device=device)[0]

                query_images_i = val_data["query_images"].to(device=device)[0]
                query_depths_i = val_data["query_depths"].to(device=device)[0]
                query_masks_i = val_data["query_masks"].to(device=device)[0]
                
                support_preds_i = net(support_images_i, vars=None)
                support_loss_i = compute_depth_loss(support_preds_i, support_depths_i, support_masks_i)
                grad = torch.autograd.grad(support_loss_i, net.parameters())
                fast_weights = update_inner_loop(net.vars, grad, args.update_lr)
            
                with torch.no_grad():
                    query_preds_i = net(query_images_i, vars=net.vars)
                    #query_loss_i = compute_depth_loss(query_preds_i, query_depths_i, query_masks_i)
                    vis_func(query_images_i[0], query_preds_i[0], 'val_0', save_dir, global_step_depth)

                for k in range(1, update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    support_preds_i = net(support_images_i, vars=fast_weights)
                    support_loss_i = compute_depth_loss(support_preds_i, support_depths_i, support_masks_i)
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(support_loss_i, fast_weights.values())
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = update_inner_loop(net.vars, grad, args.update_lr)
                
                with torch.no_grad():
                    query_preds_i = net(query_images_i, vars=fast_weights)
                    vis_func(query_images_i[0], query_preds_i[0], 'val_{}'.format(update_step), save_dir, global_step_depth)

                del net
    print('depths prior training done!')


def vis_func(image, depth, name, save_dir, step):
    image_np = image.cpu().detach().numpy()
    image_np = image_np.transpose((1, 2, 0))
    image_np = np.uint8(image_np * 255)

    depth_np = depth.cpu().detach().numpy()
    depth_color = visualize_depth(depth_np)

    cv2.imwrite(os.path.join(save_dir, 'results', '{}_{}_image.png'.format(step, name)), image_np)
    cv2.imwrite(os.path.join(save_dir, 'results', '{}_{}_depth.png'.format(step, name)), depth_color)



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)

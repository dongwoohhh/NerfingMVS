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


def train(args):
    print('Depths prior training begins !')
    
    # Dataloader.
    train_dataset = DTUMVSNeRFDataset(args, mode="train")
    #val_dataset = DTUMVSNeRFDataset(args, mode="train")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               shuffle=True)
    # Model.
    #depth_model, global_step_depth, optimizer_depth = create_depth_model(args)
    depth_model, global_step_depth, meta_optimizer = create_depth_model(args)

    update_step = args.update_step

    i_batch = 0
    n_training_steps = 150000
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
                    support_preds_i = depth_model (support_images_i, vars=fast_weights)
                    support_loss_i = compute_depth_loss(support_preds_i, support_depths_i, support_masks_i)
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(support_loss_i, fast_weights.values())
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = update_inner_loop(depth_model.vars, grad, args.update_lr)

                    query_preds_i = depth_model (query_images_i, vars=fast_weights)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    query_loss_i = compute_depth_loss(query_preds_i, query_depths_i, query_masks_i)
                    losses_q[k + 1] += query_loss_i
                #import pdb; pdb.set_trace()
            # end of all tasks
            # sum over all losses on query set across all tasks
            loss_q = losses_q[-1] / task_num

            # optimize theta parameters
            meta_optimizer.zero_grad()
            loss_q.backward()
            meta_optimizer.step()

            global_step_depth += 1
            import pdb; pdb.set_trace()
            #raise NotImplementedError


    image_list = load_img_list(args.datadir)
    
    raise NotImplementedError
    # Summary writers
    save_dir = os.path.join(args.basedir, args.expname, 'depth_priors')
    writer = SummaryWriter(os.path.join(save_dir, 'summary'))

    
    images = load_rgbs(image_list, os.path.join(args.datadir, 'images'), 
                       args.depth_H, args.depth_W)
    images_train = images.clone()
    depths, masks = load_colmap(image_list, args.datadir, 
                                args.depth_H, args.depth_W)

    depths_train = torch.from_numpy(depths).to(device)
    depths_mask_train = torch.from_numpy(masks).to(device)

    N_rand_depth = args.depth_N_rand
    N_iters_depth = args.depth_N_iters
    
    i_batch = 0
    depth_model.train()
    start = global_step_depth + 1
    
    for i in trange(start, N_iters_depth):
        batch = images_train[i_batch:i_batch + N_rand_depth]
        depth_gt, mask_gt = depths_train[i_batch:i_batch + N_rand_depth], depths_mask_train[i_batch:i_batch + N_rand_depth]
        depth_pred = depth_model(batch)
        loss = compute_depth_loss(depth_pred, depth_gt, mask_gt)

        optimizer_depth.zero_grad()
        loss.backward()
        optimizer_depth.step()
        decay_rate = 0.1
        decay_steps = args.depth_N_iters
        new_lrate = args.depth_lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer_depth.param_groups:
            param_group['lr'] = new_lrate
        i_batch += N_rand_depth
        
        if i_batch >= images_train.shape[0]:
            
            print("Shuffle depth data after an epoch!")
            rand_idx = torch.randperm(images_train.shape[0])
            images_train = images_train[rand_idx]
            depths_train = depths_train[rand_idx]
            depths_mask_train = depths_mask_train[rand_idx]
            i_batch = 0

        if i % args.depth_i_weights==0:
            path = os.path.join(save_dir, 'checkpoints', '{:06d}.tar'.format(i))
            torch.save({
                'global_step': i,
                'netG_state_dict': depth_model.model.netG.state_dict(),
                'optimizer_state_dict': optimizer_depth.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            
        if i%args.depth_i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")
        
        global_step_depth += 1
    print('depths prior training done!')
    
    with torch.no_grad():
        depth_model.eval()
        for i, image_name in enumerate(image_list):
            frame_id = image_name.split('.')[0]
            batch = images[i:i + 1]
            depth_pred = depth_model.forward(batch).cpu().numpy()
            depth_color = visualize_depth(depth_pred)
            cv2.imwrite(os.path.join(save_dir, 'results', '{}_depth.png'.format(frame_id)), depth_color)
            np.save(os.path.join(save_dir, 'results', '{}_depth.npy'.format(frame_id)), depth_pred)
    print('results have been saved in {}'.format(os.path.join(save_dir, 'results')))

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)

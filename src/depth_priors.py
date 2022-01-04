import os, sys
sys.path.append('..')
import numpy as np
import torch
import cv2

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import pdb

from models.depth_priors.mannequin_challenge_model import MannequinChallengeModel
from options import config_parser
from utils.io_utils import *
from utils.depth_priors_utils import *
from .load_llff import load_llff_data
from scipy.interpolate import griddata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_depth_model(args):
    """Instantiate depth model.
    """
    depth_model = MannequinChallengeModel()
    grad_vars = depth_model.parameters()
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.depth_lrate, betas=(0.9, 0.999))

    start = 0
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

    return depth_model, start, optimizer



def train(args):
    print('Depths prior training begins !')
    image_list = load_img_list(args.datadir)
    depth_model, global_step_depth, optimizer_depth = create_depth_model(args)

    # Summary writers
    save_dir = os.path.join(args.basedir, args.expname, 'depth_priors')
    writer = SummaryWriter(os.path.join(save_dir, 'summary'))

    
    images = load_rgbs(image_list, os.path.join(args.datadir, 'images'), 
                       None, None,)
                       #args.depth_H, args.depth_W)
    #import pdb; pdb.set_trace()
    image_H = images.shape[-2]
    image_W = images.shape[-1]

    ratio = image_W / image_H
    output_H = 480
    output_W = int(output_H * ratio)
    #import pdb; pdb.set_trace()
    images = load_rgbs(image_list, os.path.join(args.datadir, 'images'), 
                       output_H, output_W,)

    images_train = images.clone()

    
    depths, masks = load_colmap(image_list, args.datadir, 
                                output_H, output_W)
                                #args.depth_H, args.depth_W)

    # save depth
    depthdir = os.path.join(args.datadir, 'depth')
    if not os.path.exists(depthdir):
            os.mkdir(depthdir)

    for i, image_name in enumerate(image_list):
        depths_save = torch.from_numpy(depths)
        masks_save = torch.from_numpy(masks)
        depth_path = os.path.join(depthdir, image_name+'.pt')
        if not os.path.exists(depth_path):
            depth_cat = torch.stack([depths_save[i], masks_save[i]])
            torch.save(depth_cat, depth_path)


    depths_train = torch.from_numpy(depths).to(device)
    depths_mask_train = torch.from_numpy(masks).to(device)
    
    N_rand_depth = args.depth_N_rand
    N_iters_depth = args.depth_N_iters
    
    i_batch = 0
    depth_model.train()
    start = global_step_depth + 1
    n_images = len(image_list)

    for i in trange(start, N_iters_depth):
        n_batch = min(i_batch + N_rand_depth, n_images)
        batch = images_train[i_batch:n_batch]
        depth_gt, mask_gt = depths_train[i_batch:n_batch], depths_mask_train[i_batch:n_batch]
        
        depth_pred = depth_model(batch)
        if depth_gt.shape[0] == 1:
            depth_pred = depth_pred.unsqueeze(0)

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
        depthdir = os.path.join(args.datadir, 'depth_dense')
        if not os.path.exists(depthdir):
            os.mkdir(depthdir)

        depths_pred = []
        for i, image_name in enumerate(image_list):
            frame_id = image_name.split('.')[0]
            batch = images[i:i + 1]
            depth_pred = depth_model.forward(batch).cpu()
            
            #depth_pred_save = torch.nn.functional.interpolate(depth_pred.unsqueeze(0), size=(image_H, image_W), mode='bilinear')
            torch.save(depth_pred, os.path.join(depthdir, image_name+'.pt'))

            depth_pred = depth_pred.numpy()
            depths_pred.append(depth_pred)
            depth_color = visualize_depth(depth_pred.squeeze())
            cv2.imwrite(os.path.join(save_dir, 'results', '{}_depth.png'.format(frame_id)), depth_color)
            cv2.imwrite(os.path.join(depthdir, image_name+'.png'), depth_color)
            np.save(os.path.join(save_dir, 'results', '{}_depth.npy'.format(frame_id)), depth_pred)

    depths_pred = np.stack(depths_pred)
    N_views = depths_pred.shape[0]
    
    _, poses, _, _, _, _, sc = load_llff_data(args.datadir, factor=None, recenter=True, spherify=None, N_views=N_views)
    intrinsics, c2w_mats = batch_parse_llff_poses(poses)

    depths_pred = align_scales(depths_pred, depths, masks, sc)

    wdepthdir = os.path.join(args.datadir, 'warped_depth')
    if not os.path.exists(wdepthdir):
            os.mkdir(wdepthdir)
    
    for i_tar, image_name in enumerate(image_list):
        print('{} / {}'.format(i_tar, len(image_list)))
        K_tar = torch.from_numpy(intrinsics[i_tar]).float()
        c2w_tar = torch.from_numpy(c2w_mats[i_tar]).float()

        src_depths = np.concatenate([depths_pred[:i_tar], depths_pred[i_tar+1:]], axis=0)
        K_src = np.concatenate([intrinsics[:i_tar], intrinsics[i_tar+1:]], axis=0)
        c2w_src = np.concatenate([c2w_mats[:i_tar], c2w_mats[i_tar+1:]], axis=0)

        src_depths = torch.from_numpy(src_depths).float()
        K_src = torch.from_numpy(K_src).float()
        c2w_src = torch.from_numpy(c2w_src).float()


        depths_warped, depth_warped_median = warp_src_to_tgt(K_tar, c2w_tar, K_src, c2w_src, src_depths)
        depths_warped = torch.cat([depths_warped[:i_tar], torch.zeros_like(depths_warped[0:1]), depths_warped[i_tar:]], dim=0)
        
        torch.save(depths_warped, os.path.join(wdepthdir, image_name+'.pt'))
        depth_warped_median = depth_warped_median.numpy()
        depth_warped_median = visualize_depth(depth_warped_median.squeeze())
        cv2.imwrite(os.path.join(wdepthdir, image_name+'.png'), depth_warped_median)
        #import pdb; pdb.set_trace()

    print('results have been saved in {}'.format(os.path.join(save_dir, 'results')))


def parse_llff_pose(pose):
    '''
    convert llff format pose to 4x4 matrix of intrinsics and extrinsics (opencv convention)
    Args:
        pose: matrix [3, 4]
    Returns: intrinsics [4, 4] and c2w [4, 4]
    '''
    h, w, f = pose[:3, -1]
    c2w = pose[:3, :4]
    c2w_4x4 = np.eye(4)
    c2w_4x4[:3] = c2w
    c2w_4x4[:, 1:3] *= -1
    intrinsics = np.array([[f, 0, w / 2., 0],
                           [0, f, h / 2., 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    return intrinsics, c2w_4x4


def batch_parse_llff_poses(poses):
    all_intrinsics = []
    all_c2w_mats = []
    for pose in poses:
        intrinsics, c2w_mat = parse_llff_pose(pose)
        all_intrinsics.append(intrinsics)
        all_c2w_mats.append(c2w_mat)
    all_intrinsics = np.stack(all_intrinsics)
    all_c2w_mats = np.stack(all_c2w_mats)
    return all_intrinsics, all_c2w_mats


def align_scales(depth_priors, colmap_depths, colmap_masks, sc):
    ratio_priors = []
    for i in range(depth_priors.shape[0]):
        ratio_priors.append(np.median(colmap_depths[i][colmap_masks[i]]) / np.median(depth_priors[i][colmap_masks[i]]))
    ratio_priors = np.stack(ratio_priors)
    ratio_priors = ratio_priors[:, np.newaxis, np.newaxis]

    depth_priors = depth_priors * sc * ratio_priors #align scales
    return depth_priors

def warp_src_to_tgt(K_tar, c2w_tar, K_src, c2w_src, src_depths):
    n_views, height, width = src_depths.shape

    K_t = K_tar[:3, :3]
    c2w_t = c2w_tar

    K_src = K_src[:, :3, :3]
    #c2w_src = c2w_src
    
    depths_warped = []
    #rgb_warped = []
    # Forward warping
    for i in range(n_views):
        K_i = K_src[i]
        c2w_i = c2w_src[i]
        depth_i = src_depths[i]

        x_pixel = torch.arange(width).reshape(1,width).repeat([height,1]).cpu()
        y_pixel = torch.arange(height).reshape(height,1).repeat([1,width]).cpu()
        ones = torch.ones_like(y_pixel).reshape(-1).cpu()

        xy = torch.stack([x_pixel.reshape(-1), y_pixel.reshape(-1), ones]).float()
        xy = torch.matmul(torch.inverse(K_i), xy)

        xyz = depth_i.reshape(-1) * xy
        xyz_i = torch.cat([xyz, ones[None].float()], dim=0)

        xyz_world = torch.matmul(c2w_i, xyz_i)
        xyz_t = torch.matmul(torch.inverse(c2w_t), xyz_world)
        xyz_t = torch.matmul(K_t, xyz_t[:3, :])

        z_t = xyz_t[2, :]
        xy_t = xyz_t[:2, :] / z_t[None]
        #rgb_t = rgb_i.reshape(-1, 3)
        
        # interpolate zero holes.
        x_grid = x_pixel.float() / (width-1)
        y_grid = y_pixel.float() / (height-1)

        x_t = xy_t[0] / (width-1)
        y_t = xy_t[1] / (height-1)

        points = torch.stack([y_t, x_t], dim=-1)
        values = z_t
        
        """
        depth_i_nearest = griddata(points, values, (y_grid, x_grid), method='nearest') #linear
        depth_i_nearest = torch.from_numpy(depth_i_nearest)
        depth_i = depth_i_nearest
        """
        """
        depth_i_linear = griddata(points, values, (y_grid, x_grid), method='linear') #linear
        depth_i_linear = torch.from_numpy(depth_i_linear).float()
        depth_i_interp = depth_i_linear
        #
        #depth_i = depth_i_linear
        
        mask_nans = torch.isnan(depth_i_interp)
        depth_i_interp[mask_nans] = depth_i_nearest[mask_nans]
        depth_i = depth_i_interp
        """
        # Visualization.
        """
        depth_vis1 = depth_i_interp / torch.max(depth_i_interp)
        #depth_vis2 = depth_i_nearest / torch.max(depth_i_nearest)
        
        imageio.imwrite('warping_test/depth_warped_{}_interp.png'.format(i), depth_vis1)
        #imageio.imwrite('warping_test/depth_warped_{}_nearest.png'.format(i), depth_vis2)
        """

        mask = (xy_t[0] >= -0.5) & (xy_t[0] < width-0.5) & (xy_t[1] >= -0.5) & (xy_t[1] < height-0.5)

        x = torch.round(xy_t[0, mask])
        y = torch.round(xy_t[1, mask])
        z = z_t[mask]
        #rgb = rgb_t[mask]
        #import pdb; pdb.set_trace()
        depth_i_vis = torch.zeros(height, width).float().cpu()
        #rgb_i = torch.zeros(height, width, 3).float()
        z, indices = torch.sort(z, descending=True)
        x = x[indices]
        y = y[indices]

        depth_i_vis[y.long(),x.long()] = z
        depth_i = depth_i_vis
        #rgb_i[y.long(),x.long(), :] = rgb
        
        depth_i_vis = depth_i_vis.unsqueeze(-1)
        mask_empty = depth_i_vis==0
        mask_empty = mask_empty.to(torch.uint8)

        depth_vis = depth_i_vis / torch.max(depth_i_vis)
        depth_vis = depth_vis.numpy()
        imageio.imwrite('warping_test/depth_warped_{}.png'.format(i), depth_vis)
        #imageio.imwrite('warping_test/rgb_warped_{}.png'.format(i), rgb_i)
        

        depths_warped.append(depth_i)
        #rgb_warped.append(rgb_i)

    depths_warped = torch.stack(depths_warped).unsqueeze(1)
    #rgb_warped = torch.stack(rgb_warped)
    #import pdb;pdb.set_trace()
    depths_warped_np = depths_warped.numpy()
    depths_warped_np[depths_warped_np==0] = float('nan')
    depth_warped_median = np.nanmedian(depths_warped_np, axis=0)[0]
    #depth_warped_median = np.nanmin(depths_warped_np, axis=0)[0]
    depth_warped_median[np.isnan(depth_warped_median)] = 0.0

    #import pdb;pdb.set_trace()
    
    #Visualization.
    
    #depth_vis_median = depth_warped_median[0] / torch.max(depth_warped_median[torch.logical_not(torch.isnan(depth_warped_median))])
    
    #depth_vis_median[torch.isnan(depth_vis_median)] = 0
    #rgb_warped_median = torch.nanmedian(rgb_warped, dim=0)[0]
    #rgb_warped_median[torch.isnan(depth_vis_median)] = 0
    depth_vis_median = depth_warped_median / np.max(depth_warped_median)
    imageio.imwrite('warping_test/depth_warped_median.png', depth_vis_median)
    
    #imageio.imwrite('warping_test/rgb_warped_median.png', rgb_warped_median)
    
    import pdb;pdb.set_trace()
    
    return depths_warped, depth_warped_median





if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args)

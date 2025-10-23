import torch, re
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2
import os
from datetime import datetime
import shutil
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
import imageio
from skimage.metrics import structural_similarity
import math


lpips_alex = lpips.LPIPS(net="alex")  # best forward scores
lpips_vgg = lpips.LPIPS(
    net="vgg"
)  # closer to "traditional" perceptual loss, when used for optimization

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10.0 * np.log(x + TINY_NUMBER) / np.log(10.0)


def compute_pose_diff(pose1, pose2):
    assert pose1.shape == pose2.shape
    b, v, _ = pose1.shape
    pose1 = pose1.view(b, v, 4, 4)
    pose2 = pose2.view(b, v, 4, 4)
    rot1, rot2 = pose1[:, :, :3, :3], pose2[:, :, :3, :3]
    c1, c2 = pose1[:, :, :-1, -1], pose2[:, :, :-1, -1]
    rot_diff = torch.matmul(rot1, rot2.transpose(-2, -1)).view(b, v, -1)
    c_diff = (c1 - c2).view(b, v, -1)
    return torch.cat([rot_diff, c_diff], dim=-1)

def save_current_code(outdir):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code_{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            "data*",
            "pretrained*",
            "logs*",
            "out*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
    )


def img2mse(x, y, mask=None):
    """
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    """
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (
            torch.sum(mask) * x.shape[-1] + TINY_NUMBER
        )


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def show_attention_map(view_attn, epipolar_attn, view_interaction_map, epipolar_interaction_map, out_folder, prefix, global_step, mask=None):
    if mask is not None:
        mask = mask[0].detach().cpu().numpy() > 0
    view_attn = view_attn.detach().cpu()/10.
    view_attn = img_HWC2CHW(colorize(view_attn, cmap_name="jet", mask=mask))
    view_attn = view_attn.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(
        out_folder, prefix[:-1] + "_{:03d}_views_attn1.png".format(global_step)
    )
    imageio.imwrite(filename, view_attn)
    
    view_interaction_map = view_interaction_map.detach().cpu()/10.
    view_interaction_map = img_HWC2CHW(colorize(view_interaction_map, cmap_name="jet", mask=mask))
    view_interaction_map = view_interaction_map.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(
        out_folder, prefix[:-1] + "_{:03d}_views_attn2.png".format(global_step)
    )
    imageio.imwrite(filename, view_interaction_map)
    
    epipolar_interaction_depth = epipolar_interaction_map.detach().cpu()
    epipolar_interaction_depth = img_HWC2CHW(colorize(epipolar_interaction_depth, cmap_name="jet"))
    epipolar_interaction_depth = epipolar_interaction_depth.permute(1, 2, 0).detach().cpu().numpy()
    filename = os.path.join(
        out_folder, prefix[:-1] + "_{:03d}_AE_depth.png".format(global_step)
    )
    imageio.imwrite(filename, epipolar_interaction_depth)


def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x,
    cmap_name="jet",
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
    """
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new


# tensor
def colorize(x, cmap_name="jet", mask=None, range=None, append_cbar=False, cbar_in_image=False):
    device = x.device
    x = x.cpu().numpy()
    # if mask is not None:
    #     mask = mask.cpu().numpy() > 0.99
    #     kernel = np.ones((3, 3), np.uint8)
    #     mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x


def get_ssim(pred_img, gt_img):
    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()
    # ssim = structural_similarity(pred_img, gt_img, multichannel=True)
    ssim = structural_similarity((pred_img * 255.0).astype(np.uint8),
                                 (gt_img * 255.0).astype(np.uint8),
                                 multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return ssim


def lpips(img1, img2, net="vgg", format="NCHW"):
    if format == "HWC":
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == "NHWC":
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    if net == "alex":
        return lpips_alex(img1, img2)
    elif net == "vgg":
        return lpips_vgg(img1, img2)
    
    
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


#################################################  MVS  helper functions   #####################################
from kornia.utils import create_meshgrid

def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """

    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

        # added by cyl
        invalid_mask = torch.abs(src_grid_d[:, 2:]) < 1e-8
        src_grid_d[:, 2:][invalid_mask] = 1e-8


        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid


def adjust_sigma(warmup_steps, max_steps, max_sigma, DPS, step):

    # If we are in the warmup phase
    if step < warmup_steps:
        # Linear warmup
        sigma = max_sigma * step / warmup_steps
    # If we are in the decay phase
    else:
        # Subtract warmup steps from step and max_steps
        step -= warmup_steps
        max_steps -= warmup_steps

        # Cosine decay
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        # Calculate the end sigma value
        end_sigma = 1e-5
        # Calculate the current sigma value
        sigma = max_sigma * q + end_sigma * (1 - q)
    # Update sigma in the DPS module
    DPS.TOPK.sigma = sigma

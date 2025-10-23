import os
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from vet3dnerf.data_loaders import dataset_dict
from vet3dnerf.render_image import render_single_image
from vet3dnerf.model import EVENeRF
from vet3dnerf.sample_ray import RaySamplerSingleImage
from utils import img_HWC2CHW, colorize, img2psnr, lpips, get_ssim
import config
import torch.distributed as dist
from vet3dnerf.projection import Projector
from vet3dnerf.data_loaders.create_training_dataset import create_training_dataset
import imageio
import tqdm
from glob import glob


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def img2video(args):
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    imgs_path = os.path.join(out_folder, "*render_for_video.png")
    imgs = sorted(glob(imgs_path))
    out_frames = []
    crop_ratio = args.crop_ratio
    for img in imgs:
        image = imageio.imread(img)
        h, w = image.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        # crop out image boundaries
        out_frame = image[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_frames.append(out_frame)
    out_dir = os.path.join(out_folder, "{}.mp4".format(args.expname))
    print("saved video to {}".format(out_dir))
    imageio.mimwrite(out_dir, out_frames, fps=30, quality=8)

@torch.no_grad()
def render(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    assert args.eval_dataset == "llff_render" or args.eval_dataset == "shiny_render" or args.eval_dataset == "nerf_synthetic_render", ValueError(
        "rendering mode available only for llff dataset"
    )
    dataset = dataset_dict[args.eval_dataset](args, scenes=args.eval_scenes)
    loader = DataLoader(dataset, batch_size=1)
    # iterator = iter(loader)

    # Create GNT model
    model = EVENeRF(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    # create projector
    projector = Projector(device=device)

    for indx, data in enumerate(tqdm.tqdm(loader, leave=True)):
        if args.local_rank == 0:
            tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                render_stride=args.render_stride,
                prefix="val/" if args.run_val else "train/",
                out_folder=out_folder,
                # vis_depth=args.vis_depth,
            )
            torch.cuda.empty_cache()


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    render_stride=1,
    prefix="",
    out_folder="",
    # vis_depth=False,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.view_selection_net is not None:
            # src_ids, view_weight, volume_feat = model.view_selection_net(ray_batch["src_rgbs"].permute(0, 1, 4, 2, 3),
            #                                                              ray_batch["depth_range"], args.inv_uniform,
            #                                                              ray_batch["proj_mats"])
            # ray_batch["src_rgbs"] = ray_batch["src_rgbs"][:, src_ids].view(
            #     ray_batch["src_rgbs"].shape[0], args.num_source_views, *ray_batch["src_rgbs"].shape[2:])
            # ray_batch["src_cameras"] = ray_batch["src_cameras"][:, src_ids].view(
            #     ray_batch["src_cameras"].shape[0], args.num_source_views, *ray_batch["src_cameras"].shape[2:])
            volume_feat = model.view_selection_net(ray_batch["src_rgbs"].permute(0, 1, 4, 2, 3),
                                                   ray_batch["depth_range"], args.inv_uniform,
                                                   ray_batch["proj_mats"])
        if (model.feature_net is not None) and (model.view_selection_net is not None):
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            # featmaps = view_weight.squeeze(0) * featmaps + featmaps
            # featmaps = view_weight.squeeze(0).unsqueeze(2).unsqueeze(3) * featmaps
        else:
            featmaps = [None, None]

        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
            volume_feat=volume_feat,
        )

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        average_im = average_im[::render_stride, ::render_stride]

    average_im = img_HWC2CHW(average_im)

    rgb_pred = ret["outputs_render"]["rgb"].detach().cpu()
    rgb_show = img_HWC2CHW(rgb_pred)
    rgb_show = rgb_show.permute(1, 2, 0).numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_render_for_video.png".format(global_step))
    imageio.imwrite(filename, rgb_show)

    # if vis_depth:
    #     depth_pred = ret["outputs_render"]["depth"].detach().cpu()
    #     depth_show = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
    #     depth_show = depth_show.permute(1, 2, 0).detach().cpu().numpy()
    #     filename = os.path.join(
    #         out_folder, prefix[:-1] + "_{:03d}_depth.png".format(global_step)
    #     )
    #     imageio.imwrite(filename, depth_show)


if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    parser.add_argument("--crop_ratio", type=float, default=0.05, help="crop ratio of images")
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    render(args)

    img2video(args)

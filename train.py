import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from vet3dnerf.data_loaders import dataset_dict
from vet3dnerf.render_ray import render_rays
from vet3dnerf.render_image import render_single_image
from vet3dnerf.model import ThreeDVETNeRFModel
from vet3dnerf.sample_ray import RaySamplerSingleImage
from vet3dnerf.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, adjust_sigma
import config
import torch.distributed as dist
from vet3dnerf.projection import Projector
from vet3dnerf.data_loaders.create_training_dataset import create_training_dataset
import imageio

import logging
from datetime import datetime
import random


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def train(args):

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

    # SummaryWriter
    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(out_folder, "log"))
        handlers = [logging.StreamHandler()]
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        handlers.append(logging.FileHandler(out_folder + f'/{dt_string}.log', mode='w'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%m-%d %H:%M:%S', handlers=handlers,
        )

    # random seed
    setup_seed(args.random_seed + args.local_rank)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        # worker_init_fn=lambda _: np.random.seed(),
        worker_init_fn=worker_init_fn,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create model
    model = ThreeDVETNeRFModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    while global_step < model.start_step + args.n_iters + 1:
        # np.random.seed()

        if args.distributed:
            train_sampler.set_epoch(epoch)

        for train_data in train_loader:
            time0 = time.time()

            # Start of core optimization loop

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(
                1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
            )
            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )

            volume_feat = model.feature3d_net(ray_batch["src_rgbs"].permute(0, 1, 4, 2, 3),
                                              ray_batch["depth_range"], args.inv_uniform,
                                              ray_batch["proj_mats"])
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            ret = render_rays(
                ray_batch=ray_batch,
                model=model,
                projector=projector,
                featmaps=featmaps,
                volume_feat=volume_feat,
                N_samples=args.N_samples,
                inv_uniform=args.inv_uniform,
                det=args.det,
                white_bkgd=args.white_bkgd,
            )

            # compute loss
            model.optimizer.zero_grad()
            loss, rgb_loss, scalars_to_log = criterion(ret["outputs_render"], ray_batch, scalars_to_log)

            loss.backward()
            for name, param in model.three_dvet_net.named_parameters():
                if param.grad is None:
                    print(name)
            scalars_to_log["loss"] = loss.item()
            scalars_to_log["rgb_loss"] = rgb_loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret["outputs_render"]["rgb"],
                                        ray_batch["rgb"]).item()
                    # scalars_to_log["mse"] = mse_error
                    scalars_to_log["psnr"] = mse2psnr(mse_error)

                    # logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                    logstr = "Epoch: {}  step: {} ".format(epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                    logging.info(logstr)
                    logging.info("each iter time {:.05f} seconds".format(dt))
                    # logging.info(view_weight.permute(0, 2, 1).data)

                    # write log
                    for k in scalars_to_log:
                        if k in ['loss', 'rgb_loss', 'psnr', 'lr']:
                            writer.add_scalar(k, scalars_to_log[k], global_step)

                if global_step % args.i_weights == 0:
                    logging.info("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                    fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0 or global_step == 15:
                    logging.info("Logging a random validation view...")
                    val_data = next(val_loader_iterator)
                    tmp_ray_sampler = RaySamplerSingleImage(
                        val_data, device, render_stride=args.render_stride
                    )
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                    log_view(
                        global_step,
                        args,
                        model,
                        tmp_ray_sampler,
                        projector,
                        gt_img,
                        render_stride=args.render_stride,
                        prefix="val/",
                        out_folder=out_folder,
                    )
                    torch.cuda.empty_cache()

                    logging.info("Logging current training view...")
                    tmp_ray_train_sampler = RaySamplerSingleImage(
                        train_data, device, render_stride=1
                    )
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                    log_view(
                        global_step,
                        args,
                        model,
                        tmp_ray_train_sampler,
                        projector,
                        gt_img,
                        render_stride=1,
                        prefix="train/",
                        out_folder=out_folder,
                    )
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature3d_net is not None:
            volume_feat = model.feature3d_net(ray_batch["src_rgbs"].permute(0, 1, 4, 2, 3),
                                              ray_batch["depth_range"], args.inv_uniform,
                                              ray_batch["proj_mats"])
        if (model.feature_net is not None) and (model.feature3d_net is not None):
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
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

    # average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
    average_im = ray_batch["src_rgbs"].cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_render"]["rgb"].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3 * w_max)
    rgb_im[:, : average_im.shape[-2], : average_im.shape[-1]] = average_im
    rgb_im[:, : rgb_gt.shape[-2], w_max : w_max + rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, : rgb_pred.shape[-2], 2 * w_max : 2 * w_max + rgb_pred.shape[-1]] = rgb_pred

    rgb_im = rgb_im.permute(1, 2, 0).detach().cpu().numpy()
    rgb_im = (255 * rgb_im).astype(np.uint8)
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}.png".format(global_step))
    imageio.imwrite(filename, rgb_im)

    # write scalar
    rgb_pred = ret["outputs_render"]["rgb"].detach().cpu()
    psnr_curr_img = img2psnr(rgb_pred, gt_img)
    # logging.info(prefix + "psnr_image: ", psnr_curr_img)
    logging.info(prefix + f"psnr_image: {psnr_curr_img}")
    model.switch_to_train()


if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        # dist.init_process_group("gloo")
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(args.local_rank)

    train(args)

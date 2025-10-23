import torch
import os
from vet3dnerf.feature_network import Easy_Conv2d
from vet3dnerf.three_dvet_network import ThreeDVETNeRF
from vet3dnerf.feature3d_network import Feature3d_Net


def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class ThreeDVETNeRFModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device("cuda:{}".format(args.local_rank))

        self.three_dvet_net = ThreeDVETNeRF(
            args,
            in_feat_ch=args.conv_feature_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            # posediffnc_dim=64,
        ).to(device)

        # create feature extraction network
        self.feature_net = Easy_Conv2d(inplanes=3,
                                    planes=args.conv_feature_dim,
                                    outplanes=args.conv_feature_dim).to(device)

        # create view selection network
        self.feature3d_net = Feature3d_Net(args, inplanes=3, planes=32, outplanes=32).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.three_dvet_net.parameters())
        learnable_params += list(self.feature_net.parameters())
        learnable_params += list(self.feature3d_net.parameters())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.three_dvet_net.parameters()},
                {"params": self.feature3d_net.parameters(), "lr": 0.00025},
                {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
            ],
            lr=args.lrate_eve,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            self.three_dvet_net = torch.nn.parallel.DistributedDataParallel(
                self.three_dvet_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.feature3d_net = torch.nn.parallel.DistributedDataParallel(
                self.feature3d_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

    def switch_to_eval(self):
        self.three_dvet_net.eval()
        self.feature_net.eval()
        self.feature3d_net.eval()

    def switch_to_train(self):
        self.three_dvet_net.train()
        self.feature_net.train()
        self.feature3d_net.train()

    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "three_dvet_net": de_parallel(self.three_dvet_net).state_dict(),
            "feature_net": de_parallel(self.feature_net).state_dict(),
            "feature3d_net": de_parallel(self.feature3d_net).state_dict(),
        }

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.three_dvet_net.load_state_dict(to_load["three_dvet_net"])
        self.feature_net.load_state_dict(to_load["feature_net"])
        self.feature3d_net.load_state_dict(to_load["feature3d_net"])


    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0
        return step

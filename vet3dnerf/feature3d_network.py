import torch
import torch.nn as nn
from vet3dnerf.feature_network import Easy_Conv2d
from utils import compute_pose_diff, homo_warp
from .base_network import MultiHeadAttention
import torch.nn.functional as F
import torchvision.transforms as transforms
# from inplace_abn import InPlaceABN
import imageio
import numpy as np
from torch.utils.checkpoint import checkpoint


def min_max_norm(in_):
    max_ = in_.max(1)[0].unsqueeze(1).expand_as(in_)
    min_ = in_.min(1)[0].unsqueeze(1).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)


class ConvReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1)):
        super(ConvReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=1, keepdim=False)
    var = torch.sum(weight * (x - mean)**2, dim=1, keepdim=False)
    return mean, var


class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x


class Conv_crossPlane(nn.Module):
    def __init__(self):
        super(Conv_crossPlane, self).__init__()

        # self.conv0 = ConvReLU3D(32, 8, kernel_size=3, padding=1)
        self.conv0 = ConvReLU3D(16, 8, kernel_size=3, padding=1)

        self.conv1 = ConvReLU3D(8, 16, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv2 = ConvReLU3D(16, 16, kernel_size=3, padding=1)

        self.conv3 = ConvReLU3D(16, 32, kernel_size=3, stride=(2, 2, 2), padding=1)
        self.conv4 = ConvReLU3D(32, 32, kernel_size=3, padding=1)

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1, stride=(2, 2, 2)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2)),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        # self.conv7 = nn.Conv3d(8, 1, kernel_size=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))

        conv4 = self.conv4(self.conv3(conv2))

        x = conv2 + self.conv5(conv4)[:, :, :, :conv2.shape[3], :conv2.shape[4]]
        del conv2
        x = conv0 + self.conv6(x)[:, :, :, :conv0.shape[3], :conv0.shape[4]]
        del conv0

        # x = self.conv7(x)

        return conv4, x


class Feature3d_Net(nn.Module):
    def __init__(self, args, inplanes, planes, outplanes, stride=1, groups=1, dilation=1, dropout=0.1,):
        super(Feature3d_Net, self).__init__()

        self.select_number = args.num_source_views
        self.psv_planes = args.N_samples

        self.planes = planes
        self.feature = FeatureNet()

        # n_head = args.head
        # D = 1
        # self.cross_transformers = nn.ModuleList([MultiHeadAttention(
        #     n_head, outplanes, outplanes // n_head, outplanes // n_head, dropout) for _ in range(D)])
        #
        # self.view_weight = nn.Linear(outplanes, 1)
        self.sigmoid = nn.Sigmoid()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # self.conv_perPlane = nn.Sequential(ConvReLU3D(32, 8, kernel_size=3, padding=1),
        #                                    ConvReLU3D(8, 16, kernel_size=3, padding=1),
        #                                    ConvReLU3D(16, 16, kernel_size=3, padding=1),
        #                                    ConvReLU3D(16, 32, kernel_size=3, padding=1),
        #                                    ConvReLU3D(32, 32, kernel_size=3, padding=1),)
        self.conv_perPlane = nn.Sequential(ConvReLU3D(32, 16, kernel_size=3, padding=(0, 1, 1),))
        self.conv_crossPlane = Conv_crossPlane()
        self.avg_pool_3d = nn.AdaptiveAvgPool3d(1)

    def build_volume_cost(self, imgs, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        src_feats = feats.permute(1, 0, 2, 3, 4).contiguous()  # (V-1, B, C, h, w)
        proj_mats = proj_mats.permute(1, 0, 2, 3).contiguous()  # (V-1, B, 3, 4)

        if pad > 0:
            src_feats = F.pad(src_feats, (pad, pad, pad, pad), "constant", 0)

        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1, H, W).permute(1, 0, 2, 3, 4)

        feat_volume = torch.empty((B, V, C, D, *src_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        in_masks = torch.empty((B, V, D, *src_feats.shape[-2:]), device=feat_volume.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs, src_feats, proj_mats)):
            feat_volume[:, i], grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            # img_warp, _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i] = in_mask.float()

            # for aa in range(D):
            #     img_a = img_warp[0, :, aa, :, :].permute(1, 2, 0).detach().cpu().numpy()
            #     img_a = (255 * img_a).astype(np.uint8)
            #     imageio.imwrite("./test/image{}_{}.png".format(i, aa), img_a)

        weight = in_masks / (torch.sum(in_masks, dim=1, keepdim=True) + 1e-8)
        feat_volume_mean, feat_volume_var = fused_mean_variance(feat_volume, weight.unsqueeze(2))
        del weight

        return feat_volume, feat_volume_mean, feat_volume_var

    def forward(self, src_imgs, depth_range, inv_uniform, proj_mats):
        b, n, c, H, W = src_imgs.shape

        if src_imgs.shape[-1] >= 800:
            hres_input = True
        else:
            hres_input = False

        feat_srcs = self.feature(self.normalize(src_imgs.contiguous().view(b * n, c, H, W)))
        feat_srcs = feat_srcs.view(b, n, *feat_srcs.shape[1:])

        ## PSV
        D = self.psv_planes // 4
        h, w = feat_srcs.shape[-2:]
        t_vals = torch.linspace(0., 1., steps=D, device=src_imgs.device, dtype=src_imgs.dtype)  # (B, D)
        near, far = depth_range[0, 0], depth_range[0, 1]
        if not inv_uniform:
            depth_values = near * (1. - t_vals) + far * t_vals
        else:
            depth_values = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
        depth_values = depth_values.unsqueeze(0)
        if hres_input and self.training:
            volume_feat_ini, volume_feat_mean, volume_feat_var = checkpoint(self.build_volume_cost,
                                                                            src_imgs, feat_srcs, proj_mats,
                                                                            depth_values,
                                                                            preserve_rng_state=False,
                                                                            use_reentrant=True)  # (B, V, C, D, H, W)
        else:
            volume_feat_ini, volume_feat_mean, volume_feat_var = self.build_volume_cost(src_imgs, feat_srcs, proj_mats, depth_values, pad=0)  # (B, V, C, D, H, W)
        volume_feat = torch.empty((b, D, 16, n, *volume_feat_ini.shape[-2:]), device=volume_feat_ini.device, dtype=torch.float)
        for i_idx in range(n):
            volume_feat_current_ini = volume_feat_ini[:,i_idx]
            volume_feat_current = torch.stack([volume_feat_current_ini, volume_feat_mean, volume_feat_var], dim=1)
            volume_feat_current = torch.transpose(volume_feat_current, 1, 3).view(b * D, 32, 3, h, w)
            volume_feat_current = self.conv_perPlane(volume_feat_current).view(b, D, 16, h, w)
            volume_feat[:, :, :, i_idx] = volume_feat_current
            # volume_feat.append(volume_feat_current)

        # volume_feat = torch.transpose(volume_feat, 1, 3).view(b * D, 32, n, h, w)
        # volume_feat = self.conv_perPlane(volume_feat).view(b, D, 32, n, h, w)
        volume_feat = torch.transpose(volume_feat, 1, 3).view(b * n, 16, D, h, w)
        q, volume_feat = self.conv_crossPlane(volume_feat)
        volume_feat = volume_feat.view(b, n, 8, D, h, w)

        # q = self.avg_pool_3d(q)
        # q = q.view(b, n, 32)
        # for cross_trans in self.cross_transformers:
        #     q, _, _ = cross_trans(q, q, q)
        # view_weight = self.view_weight(q)
        # view_weight = self.sigmoid(view_weight)
        #
        # view_weight = min_max_norm(view_weight)
        # _, src_ids = view_weight.topk(self.select_number, dim=1)
        # src_ids = src_ids.view(b, self.select_number)
        #
        # view_weight = view_weight[:, src_ids].view(b, self.select_number, 1)
        # volume_feat = volume_feat[:, src_ids].view(b, self.select_number, 8, D, h, w)

        # return src_ids, view_weight, volume_feat
        return volume_feat

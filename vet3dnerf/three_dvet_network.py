import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .condition_nerf import CondNeRF
from .base_network import MultiHeadAttention, ConvAutoEncoder, MultiHeadAttention_v1

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

class View_Cross_Transformer(nn.Module):
    def __init__(self, n_head, d_model,  dropout=0.1, D=2):
        super(View_Cross_Transformer, self).__init__()

        self.posediff_linear = nn.Sequential(nn.Linear(12, d_model), nn.ReLU())
        self.cross_transformers = nn.ModuleList([MultiHeadAttention(n_head, d_model, 
                                                             d_model//n_head, 
                                                             d_model//n_head, dropout) for _ in range(D)])

    def forward(self, ray_query, pose_diff):
        pos = self.posediff_linear(pose_diff)
        q = ray_query + pos 
        for cross_trans in self.cross_transformers:
            q, _, _ = cross_trans(q, q, q)
        out = torch.sigmoid(q)
        return out
    

class View_Dual_Transformer(nn.Module):
    def __init__(self, args, posenc_dim=3, viewenc_dim=3, dropout=0.1, d_pose=4):
        super(View_Dual_Transformer, self).__init__()
        n_head = args.head
        d_model = args.netwidth
        d_3d = d_model
        n_samples = args.N_samples
        self.view_transformer = MultiHeadAttention_v1(n_head, d_model,
                                                      d_model//n_head, d_model//n_head,
                                                      dropout, d_pose, d_3d)
        self.epipolar_interaction_net = ConvAutoEncoder(d_model*2+posenc_dim+viewenc_dim, d_model, n_samples)

    def forward(self, feats, feats_3d, mask, ray_diff, pts):
        B, N, V, _ = feats.shape
        feats, mask, ray_diff, feats_3d = feats.flatten(0, 1), mask.flatten(0, 1), ray_diff.flatten(0, 1), feats_3d.flatten(0, 1)
        view_agg_feats, view_attn, view_V =  self.view_transformer(feats, feats, feats, mask, ray_diff, feats_3d) # (B*N, V, C)
        view_agg_feats = view_agg_feats.reshape(B, N, *view_agg_feats.shape[1:])
        view_attn = view_attn.reshape(B, N, *view_attn.shape[1:])
        view_V = view_V.reshape(B, N, *view_V.shape[1:])

        V = torch.cat(torch.var_mean(view_V, dim=-2), dim=-1) # (B, N, 2C)
        V = V.transpose(1, 2).contiguous()  # (B, 2C, N)
        pts = pts.transpose(1, 2).contiguous()  # (B, C, N)

        feats_3d = feats_3d.reshape(B, N, *feats_3d.shape[1:])
        epipolar_interaction_map = self.epipolar_interaction_net(V, pts).transpose(1, 2).contiguous().unsqueeze(-2)  # (B, N, 1, C)
        feats_3d = feats_3d * epipolar_interaction_map + feats_3d

        return view_agg_feats, feats_3d, view_attn, epipolar_interaction_map


class Feat3d_Agg(nn.Module):
    def __init__(self, args, posenc_dim=3, viewenc_dim=3, dropout=0.1, d_pose=4):
        super(Feat3d_Agg, self).__init__()
        n_head = args.head
        d_model = args.netwidth
        n_samples = args.N_samples
        self.view_transformer = MultiHeadAttention(n_head, d_model,
                                                   d_model//n_head, d_model//n_head,
                                                   dropout)

    def forward(self, feats, feats_2d):
        B, N, V, _ = feats_2d.shape
        feats, feats_2d = feats.flatten(0, 1), feats_2d.flatten(0, 1)
        view_agg_feats, view_attn, _ =  self.view_transformer(feats, feats_2d, feats_2d) # (B*N, V, C)
        view_agg_feats = view_agg_feats.reshape(B, N, *view_agg_feats.shape[1:])

        return view_agg_feats


class Epipolar_Dual_Transformer(nn.Module):
    def __init__(self, args, dropout=0.1, d_pose=4):
        super(Epipolar_Dual_Transformer, self).__init__()
        n_head = args.head
        d_model = args.netwidth
        self.epipolar_transformer = MultiHeadAttention(n_head, d_model,
                                                       d_model // n_head, d_model // n_head,
                                                       dropout, d_pose)
        self.view_interaction_net = View_Cross_Transformer(n_head, d_model,
                                                           dropout, args.cross_depth)
        self.feat3d_fc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), )
        self.feat3d_fc_1 = nn.Sequential(nn.Linear(d_model, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats, feats_3d, mask, ray_diff, pose_diff):
        B, V, N, _ = feats.shape

        soft_mask = self.feat3d_fc(feats_3d)
        feats = self.sigmoid(soft_mask) * feats
        feats, mask, ray_diff, soft_mask = feats.flatten(0, 1), mask.flatten(0, 1), ray_diff.flatten(0, 1), soft_mask.flatten(0, 1)

        epipolar_agg_feats, epipolar_attn, epipolar_V = self.epipolar_transformer(feats, feats, feats, mask,
                                                                                  ray_diff)  # (B*V, N, C)
        epipolar_agg_feats = epipolar_agg_feats.reshape(B, V, *epipolar_agg_feats.shape[1:])
        epipolar_attn = epipolar_attn.reshape(B, V, *epipolar_attn.shape[1:])

        # Q = epipolar_V.max(dim=-2, keepdim=True)[0].reshape(B, V, -1)  # B*V, C
        soft_mask = F.softmax(self.feat3d_fc_1(soft_mask), dim=1)
        Q = torch.sum(soft_mask * epipolar_V, dim=1).reshape(B, V, -1)  # B*V, C

        view_interaction_map = self.view_interaction_net(Q, pose_diff)
        view_interaction_map = view_interaction_map.reshape(B, V, 1, -1)

        interaction_feats = epipolar_agg_feats * view_interaction_map
        return interaction_feats, epipolar_attn, view_interaction_map


class ThreeDVETNeRF(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3):
        super(ThreeDVETNeRF, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        self.feat3d_fc = nn.Sequential(
            nn.Linear(8, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        self.view_dual_trans = nn.ModuleList([])
        self.epipolar_dual_trans = nn.ModuleList([])
        for i in range(args.trans_depth):
            # view transformer
            view_dual_trans = View_Dual_Transformer(args, posenc_dim, viewenc_dim)
            self.view_dual_trans.append(view_dual_trans)
            # epipolar transformer
            epipolar_dual_trans = Epipolar_Dual_Transformer(args)
            self.epipolar_dual_trans.append(epipolar_dual_trans)

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.feat3d_agg = Feat3d_Agg(args, posenc_dim, viewenc_dim)
        self.cond_nerf = CondNeRF(args, posenc_dim, viewenc_dim)
        self.sigmoid = nn.Sigmoid()
        self.feature_linear = torch.nn.Linear(128, args.netwidth)
        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(viewenc_dim + args.netwidth * 2, args.netwidth)])
        self.rgb_linear = torch.nn.Linear(args.netwidth, 3)
        self.feature_linear.apply(weights_init)
        self.views_linears.apply(weights_init)

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d, pose_diff, feat_3d):
        B, N, V, _ = rgb_feat.shape
        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)

        # project rgb features to netwidth
        feats = self.rgbfeat_fc(rgb_feat) # [B, N, V, C]
        ray_diff_ = ray_diff.transpose(1, 2)
        mask_ = mask.transpose(1, 2)

        # vis_feat = self.sigmoid(vis_feat).view(B, V, N, 1)
        feats_3d = self.feat3d_fc(feat_3d)

        # transformer modules
        for i, (view_trans, epipolar_trans) in enumerate(
            zip(self.view_dual_trans, self.epipolar_dual_trans)
        ):  
            feats_raw = feats

            # view dual transformer to update q
            feats, feats_3d, _, _ = view_trans(feats, feats_3d, mask, ray_diff, embed)

            feats_ = feats.transpose(1, 2).contiguous()
            feats_3d_ = feats_3d.transpose(1, 2).contiguous()
            # epipolar dual transformer to update q
            feats_, _, _ = epipolar_trans(feats_, feats_3d_, mask_, ray_diff_, pose_diff)
            feats = feats_.transpose(1, 2).contiguous() + feats_raw

        agg_feats = feats.mean(dim=2) # (B, N, C)
        agg_feats_3d = self.feat3d_agg(feats_3d, feats).mean(dim=2)

        alpha, feature = self.cond_nerf(input_pts, input_views, agg_feats_3d, mask)

        feature = self.feature_linear(feature)
        h = torch.cat([feature, agg_feats, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = l(h)
            h = F.relu(h)
        rgb = torch.sigmoid(self.rgb_linear(h))  # [n_rays, n_samples, 3]

        return rgb, alpha


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.build import MODELS
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from timm.models.layers import trunc_normal_
from utils.logger import *
from models.dlo_constrains_loss import estimate_pointcloud_local_coord_frames, group_points
import models.dlo_constrains_loss as DCL


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center = pointops.fps(xyz, self.num_group)
        idx = pointops.knn(center, xyz, self.group_size)[0]
        neighborhood = pointops.index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_dim, memory_dim):
        """
        PCN based encoder
        """
        super().__init__()
        self.memory_module = MemoryModule(encoder_dim, memory_dim)
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_dim, 1)
        )

    def forward(self, x, flag=True):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 512 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 512
        feature_memory = self.memory_module(feature_global, flag)  # B 512
        feature_out = torch.cat((feature_global, feature_memory), dim=1)
        return feature_out


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )

    def forward(self, z):
        bs = z.size(0)

        pcd = self.mlp1(z).reshape(bs, -1, 3)  # B M C(3)

        return pcd


class pointsEnhancement(nn.Module):
    def __init__(self, group_nums=40, group_size=70):
        super().__init__()
        self._group_nums = group_nums
        self._group_size = group_size

    def forward(self, pointclouds):
        B, N, dim = pointclouds.shape
        # [b, num_points, group_size, 3],# [b, num_points, 3]
        _, group_means = group_points(pointclouds, num_points=self._group_nums, group_size=self._group_size)
        # curvatures [b,group_nums,3] coord_frames [b,group_nums,3,3],group_points [b,group_nums,group_size,dim]
        curvatures, coord_frames, grouped_points = estimate_pointcloud_local_coord_frames(pointclouds=pointclouds,
                                                                                          sampled_points=group_means,
                                                                                          neighborhood_size=self._group_size)

        main_axis = coord_frames[:, :, :, 2].unsqueeze(2)  # [b, group_nums, 1, dim]

        main_axis_expanded = main_axis.expand_as(grouped_points)  # [b, group_nums, group_size, dim]

        enhanced_points = torch.cat([grouped_points, main_axis_expanded],
                                    dim=-1)  # [b, group_nums, group_size, dim+dim]

        enhanced_points = enhanced_points.view(B, -1, enhanced_points.shape[-1])

        indices = torch.linspace(0, enhanced_points.shape[1] - 1, steps=N).long().to(enhanced_points.device)

        sampled_points = torch.index_select(enhanced_points, 1, indices)
        return sampled_points


class DLO_constrains_loss(nn.Module):
    def __init__(self, group_nums=30, group_size=100, beta=1):
        super().__init__()
        self._group_nums = group_nums
        self._group_size = group_size
        self._beta = beta

    def forward(self, pointclouds):
        b, N, dim = pointclouds.shape
        _, group_means = DCL.group_points(pointclouds, num_points=self._group_nums, group_size=self._group_size)
        # curvatures [b,group_nums,3] coord_frames [b,group_nums,3,3]
        curvatures, coord_frames, _ = DCL.estimate_pointcloud_local_coord_frames(pointclouds=pointclouds,
                                                                                 sampled_points=group_means,
                                                                                 neighborhood_size=self._group_size)
        first_vec = curvatures[:, :, 2] / (curvatures[:, :, 0] + curvatures[:, :, 1] + curvatures[:, :, 2] + 1e-9)
        second_vec = curvatures[:, :, 1] / (curvatures[:, :, 0] + curvatures[:, :, 1] + curvatures[:, :, 2] + 1e-9)
        elongated_shape_loss = -torch.sum(first_vec - second_vec) / b
        dlo_smooth_loss = DCL.angular_cost(group_means=group_means, points_directions=coord_frames[:, :, :, 2])
        loss = elongated_shape_loss + self._beta * dlo_smooth_loss
        return loss


class MemoryModule(nn.Module):
    def __init__(self, encoder_dim, memory_dim):
        super(MemoryModule, self).__init__()
        self.encoder_dim = encoder_dim
        self.memory_dim = memory_dim
        self.forget_gate = nn.Linear(self.encoder_dim, self.memory_dim)
        self.input_gate = nn.Linear(self.encoder_dim, self.memory_dim)
        self.output_gate = nn.Linear(self.encoder_dim, self.memory_dim)
        self.candidate_memory = nn.Linear(self.encoder_dim, self.memory_dim)
        self.memory = nn.Parameter(torch.randn(1, self.memory_dim).to('cuda'))
        self.alpha = 0.25  # the weight of the update for memory

    def forward(self, x, flag=True):
        f = torch.sigmoid(self.forget_gate(x))
        i = torch.sigmoid(self.input_gate(x))
        o = torch.sigmoid(self.output_gate(x))
        c = torch.tanh(self.candidate_memory(x))

        # Create a new memory_detach with the same size as x.size(0)
        memory_detach = self.memory.expand(x.size(0), -1).clone()
        memory_detach = f * memory_detach + i * c
        h = o * torch.tanh(memory_detach)

        # Update the original memory
        # self.memory.data = self.alpha * (m.max(dim=0, keepdim=True).values.detach()) + (
        #             1 - self.alpha) * self.memory.data
        if flag:
            self.memory.data = memory_detach.max(dim=0, keepdim=True).values.detach()
        return h


@MODELS.register_module()
class MP2CDLO(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # define parameters
        self.config = config
        self.num_group = config.num_group
        self.group_size = config.group_size
        self.mask_ratio = config.mask_ratio
        self.feat_dim = config.feat_dim
        self.n_points = config.n_points
        self.nbr_ratio = config.nbr_ratio
        self.encoder_dim = config.encoder_dim
        self.memory_dim = config.memory_dim
        self.enhance_group_num = config.enhance_group_num
        self.enhance_group_size = config.enhance_group_size
        self.counter = 0
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(self.encoder_dim, self.memory_dim)
        self.generator = Decoder(latent_dim=self.feat_dim, num_output=self.n_points)
        self.pointenhance = pointsEnhancement(group_nums=self.enhance_group_num, group_size=self.enhance_group_size)

        # init weights
        self.apply(self._init_weights)
        # init loss
        self._get_lossfnc_and_weights(config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_lossfnc_and_weights(self, config):
        # define loss functions
        self.shape_criterion = ChamferDistanceL1()
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        self.dlo_constrains_loss = DLO_constrains_loss(group_nums=config.sample_group_num,
                                                       group_size=config.sample_group_size,
                                                       beta=config.dlo_direction_weight)
        self.shape_matching_weight = config.shape_matching_weight
        self.shape_recon_weight = config.shape_recon_weight
        self.latent_weight = config.latent_weight
        self.dlo_constrains_weight = config.dlo_constrains_weight

    def _group_points(self, nbrs, center, B, G):
        nbr_groups = []
        center_groups = []
        perm = torch.randperm(G)
        acc = 0
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc + self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts):
        self.counter += 1
        # group points
        nbrs, center = self.group_divider(pts)  # neighborhood, center
        B, G, _ = center.shape
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        feat = self.encoder(self.pointenhance(rebuild_points.view(B, -1, 3)), True)

        # complete shape generation
        pred = self.generator(feat).contiguous()

        # shape reconstruction loss
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        idx = pointops.knn(center_groups[0], pred, int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3),
                                                                          nbrs_pred).mean()
        # shape completion loss
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred, int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3),
                                                                                nbrs_pred).mean()
        # latent reconstruction loss
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        feat_recon = self.encoder(self.pointenhance(nbrs_pred.view(B, -1, 3).detach()), True)
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)
        # fine-tune the weight of dlo constrains loss; 46 is the number of batch in one epoch(46=total_data_size/batch_size)
        if self.counter % (46 * 100) == 0 and self.dlo_constrains_weight < 0.064:
            self.dlo_constrains_weight *= 2
        # dlo constrains loss
        dlo_constrains_loss = self.dlo_constrains_weight * self.dlo_constrains_loss(pred)

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + dlo_constrains_loss

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, dlo_constrains_loss

    def forward(self, partial):
        # group points
        B, _, _ = partial.shape
        feat = self.encoder(self.pointenhance(partial), False)
        pred = self.generator(feat).contiguous()
        return pred

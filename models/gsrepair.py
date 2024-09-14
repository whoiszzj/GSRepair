import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum


def scale_activation(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1)) + 1.5  # 0.5 is the minimum scale


@register('gsrepair')
class GSRepair(nn.Module):

    def __init__(self, encoder_spec, offset_net_spec, scale_net_spec, rot_net_spec, color_net_spec):
        super().__init__()

        self.feat = None
        self.encoder = models.make(encoder_spec)

        self.mlp_net_in_dim = self.encoder.out_dim + 1  # add factor as a feature

        self.offset_net = models.make(offset_net_spec, args={'in_dim': self.mlp_net_in_dim})
        self.scale_net = models.make(scale_net_spec, args={'in_dim': self.mlp_net_in_dim})
        self.rot_net = models.make(rot_net_spec, args={'in_dim': self.mlp_net_in_dim})
        self.color_net = models.make(color_net_spec, args={'in_dim': self.mlp_net_in_dim})

        self.offset_activation = F.tanh
        self.scale_activation = scale_activation
        self.rot_activation = F.tanh
        self.color_activation = F.sigmoid

        self.BLOCK_W, self.BLOCK_H = 16, 16
        self.background = torch.ones(3, device=self.device)

    def get_xy(self, input_shape, factor):
        with torch.no_grad():
            coord = make_coord(input_shape, flatten=False)
        offset = self.offset_activation(self.offset_net(self.feat))
        offset = offset * 3 * factor
        return coord + offset

    def get_scale(self):
        return self.scale_activation(self.scale_net(self.feat))

    def get_rot(self):
        return self.rot_activation(self.rot_net(self.feat)) * torch.pi

    def get_color(self):
        return self.color_activation(self.color_net(self.feat))

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)

    def gaussian_render(self, xy, scale, rot, color, H, W):
        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )  #
        out_imgs = []
        for i in xy.shape[0]:
            pixels, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
                xy[i], scale[i], rot[i], H, W, tile_bounds
            )
            opacity = torch.ones_like(rot[i])
            out_img = rasterize_gaussians_sum(
                pixels, depths, radii, conics, num_tiles_hit,
                color[i], opacity[i], H, W, self.BLOCK_H, self.BLOCK_W,
                background=self.background,
                return_alpha=False
            )
            out_imgs.append(out_img)

        return torch.stack(out_imgs, dim=0)  # [B, 3, H, W]

    def forward(self, inp, target_shape):
        input_shape = inp.shape[-2:]
        factor_h, factor_w = target_shape[0] / input_shape[0], target_shape[1] / input_shape[1]
        factor = max(factor_h, factor_w)
        self.gen_feat(inp)
        factor_feat = torch.ones(inp.shape[0], 1, inp.shape[2], inp.shape[3], device=inp.device) * factor
        self.feat = torch.cat([self.feat, factor_feat], dim=1)  # add factor as a feature [B, C + 1, H, W]
        # flatten
        self.feat = self.feat.view(inp.shape[0], inp.shape[1] + 1, -1).permute(0, 2, 1)  # [B, H * W, C + 1]
        xy = self.get_xy(inp.shape[-2:], 2 * factor / max(target_shape[0], target_shape[1]))  # [B, H * W, 2]
        scale = self.get_scale()  # [B, H * W, 1]
        rot = self.get_rot()  # [B, H * W, 1]
        color = self.get_color()  # [B, H * W, 3]
        out_imgs = self.gaussian_render(xy, scale, rot, color, target_shape[0], target_shape[1])
        return out_imgs

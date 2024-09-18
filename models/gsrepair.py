import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt, patches

import models
from models import register
from utils import make_coord
from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
import cv2


def scale_activation(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1)) + 1.5  # 0.5 is the minimum scale


@register('gsrepair')
class GSRepair(nn.Module):

    def __init__(self, encoder_spec, offset_net_spec, scale_net_spec, rot_net_spec, color_net_spec):
        super().__init__()

        self.feat = None
        self.encoder = models.make(encoder_spec)

        self.mlp_net_in_dim = self.encoder.out_dim * 9  # + 1  # add factor as a feature

        self.offset_net = models.make(offset_net_spec, args={'in_dim': self.mlp_net_in_dim})
        self.scale_net = models.make(scale_net_spec, args={'in_dim': self.mlp_net_in_dim})
        self.rot_net = models.make(rot_net_spec, args={'in_dim': self.mlp_net_in_dim})
        self.color_net = models.make(color_net_spec, args={'in_dim': self.mlp_net_in_dim})

        self.offset_activation = F.tanh
        # self.scale_activation = scale_activation
        # self.scale_activation = F.softplus
        self.scale_activation = F.sigmoid
        self.rot_activation = F.sigmoid
        self.color_activation = F.tanh

        self.BLOCK_W, self.BLOCK_H = 16, 16
        self.background = torch.ones(3).cuda()

    def get_xy(self, input_shape, factor):
        with torch.no_grad():
            coord = make_coord(input_shape, flatten=True).unsqueeze(0).to(self.feat.device)
        offset = self.offset_activation(self.offset_net(self.feat))
        offset = offset * 3 * factor
        return coord + offset

    def get_scale(self):
        return self.scale_activation(self.scale_net(self.feat))

    def get_rot(self):
        return self.rot_activation(self.rot_net(self.feat)) * 2 * torch.pi

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
        for i in range(xy.shape[0]):
            pixels, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
                xy[i], scale[i], rot[i], H, W, tile_bounds
            )
            opacity = torch.ones_like(rot[i])
            out_img = rasterize_gaussians_sum(
                pixels, depths, radii, conics, num_tiles_hit,
                color[i], opacity, H, W, self.BLOCK_H, self.BLOCK_W,
                background=self.background,
                return_alpha=False
            )
            out_img = torch.clamp(out_img, 0, 1)
            out_img = out_img.permute(2, 0, 1).contiguous()
            out_imgs.append(out_img)

        return torch.stack(out_imgs, dim=0)

    def forward(self, inp, target_shape):
        input_shape = inp.shape[-2:]
        factor_h, factor_w = target_shape[0] / input_shape[0], target_shape[1] / input_shape[1]
        factor = max(factor_h, factor_w)
        self.gen_feat(inp)
        self.feat = self.feat.permute(0, 1, 3, 2) # swap H and W to make sure it corresponds to the xy
        # flatten
        B, C, X, Y = self.feat.shape
        self.feat = F.unfold(self.feat, 3, padding=1).view(B, C * 9, X, Y)  # [B, C * 9, X, Y]
        self.feat = self.feat.permute(0, 2, 3, 1).reshape(B, X * Y, C * 9)  # [B, X * Y, C]
        self.feat = torch.layer_norm(self.feat, self.feat.shape[2:])
        xy = self.get_xy((X, Y), 2 * factor / max(target_shape[0], target_shape[1]))  # [B, X * Y, 2]
        scale = self.get_scale() * 2 * factor  # [B, X * W, 2]
        rot = self.get_rot()  # [B, X * Y, 1]
        color = self.get_color()  # [B, X * Y, 3]
        # self.debug(inp[0], xy[0], scale[0], rot[0], target_shape)
        out_imgs = self.gaussian_render(xy, scale, rot, color, target_shape[0], target_shape[1])
        return out_imgs


    def debug(self, img, xy, scale, rot, target_shape):
        pixels = xy * torch.tensor([target_shape[1], target_shape[0]], device=xy.device)
        pixels = pixels.detach().cpu().numpy()
        scales = scale.detach().cpu().numpy()
        rots = rot.detach().cpu().numpy()
        fig, ax = plt.subplots()
        temp_img = img.detach().cpu().numpy()
        temp_img = temp_img.transpose(1, 2, 0)
        temp_img = cv2.resize(temp_img, (target_shape[1], target_shape[0]))
        plt.imshow(temp_img)
        for i in range(pixels.shape[0]):
            rot = rots[i] / (2 * math.pi) * 360
            scale = scales[i]
            pixel = pixels[i]
            ell = patches.Ellipse(pixel, scale[0] * 6, scale[1] * 6, angle=rot, edgecolor='r', facecolor='none')
            ax.add_patch(ell)
        plt.show()
        # save image
        plt.close()

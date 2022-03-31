# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> pregnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   23/08/2021 21:55
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, in_dim=512, proj_layer=None,
                 normalize_input=True, vladv2=False, projection=False, cls=False, n_classes=452):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.cls = cls
        self.n_classes = n_classes

        self.projection = projection
        if self.projection:
            if proj_layer is None:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_dim, 512, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(512),
                    # nn.ReLU(inplace=True),
                )
            else:
                self.proj = nn.Sequential(*proj_layer)

        if self.cls:
            self.cls_head = nn.Linear(in_features=512, out_features=self.n_classes)

    def init_params(self, clsts, traindescs):
        print("Init centroids")
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / (np.linalg.norm(clsts, axis=1, keepdims=True) + 1e-5)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x, atten=None):
        if atten is not None:
            if atten.shape[2] != x.shape[2] or atten.shape[3] != x.shape[3]:
                with torch.no_grad():
                    atten = F.interpolate(atten, size=(x.shape[2], x.shape[3]), mode='bilinear')
            x = x * atten.expand_as(x)

        # if self.projection:
        x_enc = self.proj(x)
        # else:
        #     x_enc = x
        if self.cls:
            x_vec = F.adaptive_avg_pool2d(x_enc, output_size=1).reshape(N, -1)
            cls_feat = self.cls_head(x_vec)

        if self.normalize_input:
            x = F.normalize(x_enc, p=2, dim=1)  # across descriptor dim

        N, C, H, W = x.shape
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        output = {
            'feat': vlad,
            'img_desc': x_enc,
        }

        if self.cls:
            output['cls'] = cls_feat
        return output


import os
import pdb
from tqdm import tqdm
import time

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Quantize_kMeans():
    def __init__(self, num_clusters=100, num_iters=10):
        self.num_clusters = num_clusters
        self.num_kmeans_iters = num_iters
        self.nn_index = torch.empty(0)
        self.centers = torch.empty(0)
        self.vec_dim = 0
        self.cluster_ids = torch.empty(0)
        self.excl_clusters = []
        self.excl_cluster_ids = []
        self.cluster_len = torch.empty(0)
        self.max_cnt = 0
        self.n_excl_cls = 0

    def get_dist(self, x, y, mode='sq_euclidean'):
        """Calculate distance between all vectors in x and all vectors in y.

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean_chunk':
            step = 65536
            if x.shape[0] < step:
                step = x.shape[0]
            dist = []
            for i in range(np.ceil(x.shape[0] / step).astype(int)):
                dist.append(torch.cdist(x[(i*step): (i+1)*step, :].unsqueeze(0), y.unsqueeze(0))[0])
            dist = torch.cat(dist, 0)
        elif mode == 'sq_euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

    # Update centers in non-cluster assignment iters using cached nn indices.
    def update_centers(self, feat):
        tst_ = time.time()
        feat = feat.detach().reshape(-1, self.vec_dim)
        # Update all clusters except the excluded ones in a single operation
        # Add a dummy element with zeros at the end
        feat = torch.cat([feat, torch.zeros_like(feat[:1]).cuda()], 0)
        self.centers = torch.sum(feat[self.cluster_ids, :].reshape(
            self.num_clusters, self.max_cnt, -1), dim=1)
        if len(self.excl_cluster_ids) > 0:
            for i, cls in enumerate(self.excl_clusters):
                # Division by num_points in cluster is done during the one-shot averaging of all
                # clusters below. Only the extra elements in the bigger clusters are added here.
                self.centers[cls] += torch.sum(feat[self.excl_cluster_ids[i], :], dim=0)
        self.centers /= (self.cluster_len + 1e-6)

    # Update centers during cluster assignment using mask matrix multiplication
    # Mask is obtained from distance matrix
    def update_centers_(self, feat, cluster_mask=None, nn_index=None, avg=False):
        feat = feat.detach().reshape(-1, self.vec_dim)
        tst_uc = time.time()
        centers = (cluster_mask.T @ feat)
        tsp_uc = time.time()
        if avg:
            self.centers /= counts.unsqueeze(-1)
        return centers

    def equalize_cluster_size(self):
        """Make the size of all the clusters the same by appending dummy elements.

        """
        # Find the maximum number of elements in a cluster, make size of all clusters
        # equal by appending dummy elements until size is equal to size of max cluster.
        # If max is too large, exclude it and consider the next biggest. Use for loop for
        # the excluded clusters and a single operation for the remaining ones for
        # updating the cluster centers.

        unq, n_unq = torch.unique(self.nn_index, return_counts=True)
        # Find max cluster size and exclude clusters greater than a threshold
        topk = 100
        if len(n_unq) < topk:
            topk = len(n_unq)
        max_cnt_topk, topk_idx = torch.topk(n_unq, topk)
        self.max_cnt = max_cnt_topk[0]
        print('max: ', self.max_cnt)
        idx = 0
        self.excl_clusters = []
        self.excl_cluster_ids = []
        while(self.max_cnt > 5000):
            self.excl_clusters.append(unq[topk_idx[idx]])
            idx += 1
            if idx < topk:
                self.max_cnt = max_cnt_topk[idx]
            else:
                break
        self.n_excl_cls = len(self.excl_clusters)
        self.excl_clusters = sorted(self.excl_clusters)
        print('n excl: ', self.n_excl_cls)
        # Store the indices of elements for each cluster
        all_ids = []
        cls_len = []
        # cls_len = torch.zeros(self.num_clusters, dtype=torch.long)
        j = 0
        for i in range(self.num_clusters):
            cur_cluster_ids = torch.where(self.nn_index == i)[0]
            # For excluded clusters, use only the first max_cnt elements
            # for averaging along with other clusters. Separately average the
            # remaining elements just for the excluded clusters.
            cls_len.append(torch.Tensor([len(cur_cluster_ids)]))
            if i in self.excl_clusters:
                self.excl_cluster_ids.append(cur_cluster_ids[self.max_cnt:])
                cur_cluster_ids = cur_cluster_ids[:self.max_cnt]
            # Append dummy elements to have same size for all clusters
            all_ids.append(torch.cat([cur_cluster_ids, -1 * torch.ones((self.max_cnt - len(cur_cluster_ids)),
                                                                       dtype=torch.long).cuda()]))
        all_ids = torch.cat(all_ids).type(torch.long)
        cls_len = torch.cat(cls_len).type(torch.long)
        self.cluster_ids = all_ids
        self.cluster_len = cls_len.unsqueeze(1).cuda()

    def cluster_assign(self, feat, feat_scaled=None):

        # quantize with kmeans
        feat = feat.detach()
        feat = feat.reshape(-1, self.vec_dim)
        if feat_scaled is None:
            feat_scaled = feat
            scale = feat[0] / (feat_scaled[0] + 1e-8)
        if len(self.centers) == 0:
            self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

        tst = time.time()
        # start kmeans
        chunk = True
        ttot_uc = 0
        ttot_dist = 0
        counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6
        centers = torch.zeros_like(self.centers)
        tinit = time.time()
        print(['*'] * 1)
        for iteration in range(self.num_kmeans_iters):
            # chunk for memory issues
            if chunk:
                self.nn_index = None
                i = 0
                chunk = 10000
                while True:
                    dist = self.get_dist(feat[i*chunk:(i+1)*chunk, :], self.centers)
                    curr_nn_index = torch.argmin(dist, dim=-1)
                    # Assign a single cluster when distance to multiple clusters is same
                    dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32)
                    curr_centers = self.update_centers_(feat[i*chunk:(i+1)*chunk, :], dist, curr_nn_index, avg=False)
                    counts += dist.detach().sum(0) + 1e-6
                    centers += curr_centers
                    if self.nn_index == None:
                        self.nn_index = curr_nn_index
                    else:
                        self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                    i += 1
                    if i*chunk > feat.shape[0]:
                        break

            self.centers = centers / counts.unsqueeze(-1)
            # Reinitialize to 0
            centers[centers != 0] = 0.
            counts[counts > 0.1] = 0.
        titers = time.time()

        if chunk:
            self.nn_index = None
            i = 0
            # chunk = 100000
            while True:
                dist = self.get_dist(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
                curr_nn_index = torch.argmin(dist, dim=-1)
                if self.nn_index == None:
                    self.nn_index = curr_nn_index
                else:
                    self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                i += 1
                if i * chunk > feat.shape[0]:
                    break
        self.equalize_cluster_size()

    def rescale(self, feat, scale=None):
        """Scale the feature to be in the range [-1, 1] by dividing by its max value.

        """
        if scale is None:
            return feat / (abs(feat).max(dim=0)[0] + 1e-8)
        else:
            return feat / (scale + 1e-8)

    def forward_pos(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._xyz.shape[1]
        if assign:
            self.cluster_assign(gaussian._xyz)
        else:
            self.update_centers(gaussian._xyz)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._xyz_q = gaussian._xyz - gaussian._xyz.detach() + sampled_centers

    def forward_dc(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2]
        if assign:
            self.cluster_assign(gaussian._features_dc)
        else:
            self.update_centers(gaussian._features_dc)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_dc_q = gaussian._features_dc - gaussian._features_dc.detach() + sampled_centers.reshape(-1, 1, 3)

    def forward_frest(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2]
        if assign:
            self.cluster_assign(gaussian._features_rest)
        else:
            self.update_centers(gaussian._features_rest)
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest_q = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)

    def forward_scale(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._scaling.shape[1]
        if assign:
            self.cluster_assign(gaussian._scaling)
        else:
            self.update_centers(gaussian._scaling)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._scaling_q = gaussian._scaling - gaussian._scaling.detach() + sampled_centers

    def forward_rot(self, gaussian, assign=False):
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1]
        if assign:
            self.cluster_assign(gaussian._rotation)
        else:
            self.update_centers(gaussian._rotation)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._rotation_q = gaussian._rotation - gaussian._rotation.detach() + sampled_centers

    def forward_scale_rot(self, gaussian, assign=False):
        """Combine both scaling and rotation for a single k-Means"""
        if self.vec_dim == 0:
            self.vec_dim = gaussian._rotation.shape[1] + gaussian._scaling.shape[1]
        feat_scaled = torch.cat([self.rescale(gaussian._scaling), self.rescale(gaussian._rotation)], 1)
        feat = torch.cat([gaussian._scaling, gaussian._rotation], 1)
        if assign:
            self.cluster_assign(feat, feat_scaled)
        else:
            self.update_centers(feat)
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._scaling_q = gaussian._scaling - gaussian._scaling.detach() + sampled_centers[:, :3]
        gaussian._rotation_q = gaussian._rotation - gaussian._rotation.detach() + sampled_centers[:, 3:]

    def forward_dcfrest(self, gaussian, assign=False):
        """Combine both features_dc and rest for a single k-Means"""
        if self.vec_dim == 0:
            self.vec_dim = (gaussian._features_rest.shape[1] * gaussian._features_rest.shape[2] +
                            gaussian._features_dc.shape[1] * gaussian._features_dc.shape[2])
        if assign:
            self.cluster_assign(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
        else:
            self.update_centers(torch.cat([gaussian._features_dc, gaussian._features_rest], 1))
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_dc_q = gaussian._features_dc - gaussian._features_dc.detach() + sampled_centers[:, :3].reshape(-1, 1, 3)
        gaussian._features_rest_q = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers[:, 3:].reshape(-1, deg, 3)

    def replace_with_centers(self, gaussian):
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)


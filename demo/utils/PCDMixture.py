# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import glob
import os
import time
import sys
import numpy as np
import torch

sys.path.append('../../')
from extensions.pointops.functions import pointops
from sklearn.mixture import GaussianMixture
from scipy.spatial import cKDTree
from torch_kmeans import KMeans as torch_Kmeans
from sklearn.cluster import KMeans as sklearn_Kmeans


def remove_statistical_outliers(input_queue, output_queue, nb_neighbors=20, std_ratio=2.0):
    while True:
        points = input_queue.get()

        kdtree = cKDTree(points)

        distances, _ = kdtree.query(points, k=nb_neighbors)
        mean_distances = np.mean(distances, axis=1)

        mean = np.mean(mean_distances)
        std_dev = np.std(mean_distances)

        distance_threshold = mean + std_ratio * std_dev

        inlier_mask = mean_distances <= distance_threshold
        inliers = points[inlier_mask]
        outliers = points[~inlier_mask]
        output_queue.put((inliers, outliers))


class Mixture:
    def __init__(self, dir_path, num_group=50, group_size=64, threshold=0.025):
        self.dir_path = dir_path
        self.num_group = num_group
        self.group_size = group_size
        self.threshold = threshold
        self.sklearn_kmeans = sklearn_Kmeans(n_clusters=self.num_group, random_state=3407, init='k-means++', verbose=0,
                                             n_init=10)
        self.torch_kmeans = torch_Kmeans(n_clusters=self.num_group, verbose=0, init_method='rnd', seed=3407)

        self.gmm = GaussianMixture(n_components=self.num_group, covariance_type="full", verbose=0,
                                   random_state=3407,
                                   init_params='k-means++', n_init=1)

    def remove_statistical_outliers(self, points, nb_neighbors=20, std_ratio=2.0):

        kdtree = cKDTree(points)

        distances, _ = kdtree.query(points, k=nb_neighbors)
        mean_distances = np.mean(distances, axis=1)

        mean = np.mean(mean_distances)
        std_dev = np.std(mean_distances)

        distance_threshold = mean + std_ratio * std_dev

        inlier_mask = mean_distances <= distance_threshold
        inliers = points[inlier_mask]
        outliers = points[~inlier_mask]

        return inliers, outliers

    def load_file(self, group_number):
        '''
            Load the partials and pred point clouds from the specified directory
            return: partials_cloud, pred_cloud (Open3D point cloud objects)
        '''
        pattern = os.path.join(self.dir_path, f'*_{group_number}.npy')
        files = glob.glob(pattern)
        if len(files) != 2:
            print(f"Found {len(files)} files, expected 2. Please check the naming convention and try again.")
            return None, None
        partials_cloud_path = [f for f in files if 'pointcloud' in f][0]
        pred_cloud_path = [f for f in files if 'pred' in f][0]

        center = partials_cloud_path.split('_')[-3]
        center = [float(value) for value in center.replace('[', '').replace(']', '').split()]
        sigma = float(partials_cloud_path.split('_')[-2])

        partials_points = np.load(partials_cloud_path) * sigma + center
        pred_points = np.load(pred_cloud_path) * sigma + center

        partials_points, _ = self.remove_statistical_outliers(partials_points, nb_neighbors=10, std_ratio=2)
        pred_points, _ = self.remove_statistical_outliers(pred_points, nb_neighbors=10, std_ratio=2)

        return partials_points, pred_points

    def group_points(self, xyz, num_group, group_size):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center = pointops.fps(xyz, num_group)
        idx = pointops.knn(center, xyz, group_size)[0]
        neighborhood = pointops.index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)  # Centering neighborhood points
        return neighborhood, center

    def group_points_with_center(self, xyz, num_group, group_size):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center = pointops.fps(xyz, num_group)
        idx = pointops.knn(center, xyz, group_size)[0]
        neighborhood = pointops.index_points(xyz, idx)
        # neighborhood = neighborhood - center.unsqueeze(2)  # Centering neighborhood points
        return neighborhood, center

    def find_new_centers(self, pred_center, partials_center):
        '''
            Find pred_center points that need to be added to partials_cloud.
        '''
        pred_center_np = pred_center.cpu().numpy()
        partials_center_np = partials_center.cpu().numpy()

        # Calculate distances
        distances = np.linalg.norm(pred_center_np[:, :, None, :] - partials_center_np[:, None, :, :], axis=-1)

        # Find the minimum distances
        min_distances = distances.min(axis=2)

        # Find the indices where the minimum distance is greater than the threshold
        indices = np.where(min_distances > self.threshold)

        # Correctly select the new centers based on the correct axis
        new_centers = pred_center_np[indices]

        if new_centers.shape[0] > 0:
            return new_centers
        else:
            return None

    def combine_points_if_close(self, pred_center, pred_neighborhood, partials_center):
        '''
            Combine pred_neighborhood to partials_cloud if the distance between
            pred_center and partials_center is greater than the threshold.
        '''
        new_centers = self.find_new_centers(pred_center, partials_center)

        if new_centers is None:
            return None

        combined_points = []

        for center in new_centers:
            # Find the index of the new center in the pred_center array
            pred_idx = np.where(np.all(pred_center.cpu().numpy()[0] == center, axis=1))[0][0]

            # Combine pred_neighborhood to partials_center
            combined = pred_neighborhood[0, pred_idx, :, :] + pred_center[0, pred_idx, :].unsqueeze(0)
            combined_points.append(combined.cpu().numpy())

        if len(combined_points) > 0:
            combined_points = np.vstack(combined_points)
            return combined_points
        else:
            return None

    def cluster_with_gmm(self, pred_cloud, n_clusters=50):

        points = pred_cloud

        self.gmm.fit(points)

        labels = self.gmm.predict(points)

        centers = self.gmm.means_

        return labels, centers

    def run(self, group_number, dtype='GMM'):
        # group_divider = Group(num_group=num_group, group_size=group_size)
        partials_cloud_points, pred_cloud_points = self.load_file(group_number)

        # Convert clouds to torch.Tensor
        pred_cloud_tensor = torch.from_numpy(pred_cloud_points).unsqueeze(0).float().cuda()
        partials_cloud_tensor = torch.from_numpy(partials_cloud_points).unsqueeze(0).float().cuda()
        time1 = time.time()

        # Use Group class to group pred_cloud
        pred_neighborhood, pred_center = self.group_points(pred_cloud_tensor, num_group=self.num_group,
                                                           group_size=self.group_size)
        # Use FPS to get centers of partials_cloud
        partials_center = pointops.fps(partials_cloud_tensor, self.num_group)

        # Combine points if close
        combined_point_cloud = self.combine_points_if_close(pred_center, pred_neighborhood, partials_center)
        if combined_point_cloud is not None:
            final_cloud = np.concatenate((partials_cloud_points, combined_point_cloud), axis=0)
        else:
            final_cloud = partials_cloud_points

        every_k_points = int(len(final_cloud) / 1024)

        downsampled_cloud = final_cloud[::every_k_points]
        # labels, centers = self.cluster_with_gmm(downsampled_cloud, self.num_group)
        if dtype == 'GMM':
            labels, centers = self.cluster_with_gmm(downsampled_cloud, self.num_group)
            centers_tensor = torch.from_numpy(centers).to(device='cuda')
        elif dtype == 'torch-Kmeans':
            result = self.torch_kmeans(torch.from_numpy(downsampled_cloud).unsqueeze(0).cuda())
            centers = result.centers.squeeze(0)
            centers_tensor = centers.to(device='cuda').double()
        elif dtype == 'sklearn-Kmeans':
            result = self.sklearn_kmeans.fit(downsampled_cloud)
            centers = result.cluster_centers_
            centers_tensor = torch.from_numpy(centers).to(device='cuda').double()
        else:
            print('Please select the correct dtype!')
            return None

        # downsampled_cloud_tensor = torch.from_numpy(downsampled_cloud).unsqueeze(0).float().cuda()
        time2 = time.time()
        # print(f"Time cost of GMM is:{time2 - time1:0.4f}")      

        points_tensor = torch.from_numpy(downsampled_cloud).to(device='cuda')

        return points_tensor, partials_cloud_points, centers_tensor

import os
import sys
import time
import numpy as np
import open3d as o3d

sys.path.append('../')
from utils.B_spline import B_spline  # B_spline interpolation
from utils.data_process_inference import Preprocess  # Preprocess the point cloud
from utils.predict_rw_single import ModelInference  # Single Frame Inference
from utils.PCDMixture import Mixture  # Combine the results of original point cloud and predict point cloud
from utils.Visualization import Visualizer  # Visualize the rgb image and point clouds
from utils.PointsSort import connect_keypoints  # Sort the keypoints


class ModelTest():
    def __init__(self, config_path, weights_path, input_dir, result_dir, num_group=50, group_size=64, delta=0.001,
                 dtype='BSpline2'):
        self.config_path = config_path
        self.weights_path = weights_path
        self.pcd_dir = os.path.join(input_dir, 'pcd/pcd_normalized')  # save the normalized point cloud
        self.png_dir = os.path.join(input_dir, 'rgb')
        self.result_dir = result_dir
        self.num_group = num_group
        self.group_size = group_size
        self.delta = delta
        self.dtype = dtype
        self.preprocess = Preprocess(input_dir)  # Preprocess the point cloud (normalize and downsample)
        self.model_inference = ModelInference(self.config_path, self.weights_path, self.pcd_dir, self.result_dir)
        self.mixture = Mixture(self.result_dir, num_group=self.num_group, group_size=self.group_size)
        self.Bspline = B_spline(dtype=self.dtype, delta=self.delta)
        self.visualizer = Visualizer()  # Visualize the results using open3d and threading

    def run(self, group_num):
        self.preprocess.preprocess(group_num)
        self.model_inference.inference(group_num)
        downsampled_cloud, partial_cloud, centers = self.mixture.run(group_num, dtype='sklearn-Kmeans')

        connection_loss, connection_list, connection_pairs = connect_keypoints(downsampled_cloud, centers)

        control_points = [centers[i].tolist() for i in connection_list]
        control_points, curve_points = self.Bspline.run(control_points=control_points)

        downsampled_clouds = self.Bspline.down_sample(curve_points, num=50)
        # Save the curve points if needed
        os.makedirs(os.path.join(self.result_dir, f'../key_points'), exist_ok=True)
        np.save(os.path.join(self.result_dir, f'../key_points/curve_points_{group_num}.npy'), downsampled_clouds)
        if self.png_dir is not None:
            png_path = os.path.join(self.png_dir, f'color_image_{group_num}.png')
            rgb = o3d.io.read_image(png_path)
        else:
            print('Please provide the path to the color image')
        self.visualizer.visualize_points_cloud_and_png(downsampled_clouds, partial_cloud, rgb)


if __name__ == '__main__':
    config_path = '../cfg/config.yaml'  # model config
    weights_path = '../cfg/model.pth'  # mdoel weights
    input_dir = '../data'  # input data (pcd and rgb is needed)
    result_dir = f'{input_dir}/results/'  # save the results
    model_test = ModelTest(config_path, weights_path, input_dir=input_dir, result_dir=result_dir, num_group=50,
                           group_size=64, delta=0.001, dtype='BSpline2')
    index_list = ['0091', '0102', '0156', '0166', '0190', '0196', '0285', '0500', '0590', '0644']
    time_cost = 0
    for group_num in index_list:
        t1 = time.time()
        model_test.run(group_num)
        t2 = time.time()
        time.sleep(0.1)
        time_cost += t2 - t1
        print(f"Time cost:{t2 - t1}")

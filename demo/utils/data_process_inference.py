import os
import numpy as np
import shutil


class Preprocess():
    def __init__(self, input_dir, num_points=2048):
        self.num_points = num_points
        self.input_dir = input_dir
        self.pcd_dir = os.path.join(input_dir, 'pcd')
        self.pcd_normalized_dir = os.path.join(input_dir, 'pcd/pcd_normalized')

        if os.path.exists(self.pcd_normalized_dir):
            shutil.rmtree(self.pcd_normalized_dir)
        os.makedirs(self.pcd_normalized_dir, exist_ok=True)

    def preprocess(self, group_num, num_points=2048):
        file_path = os.path.join(self.pcd_dir, f'pointcloud_{group_num}.npy')
        # 读取.npy文件
        points = np.load(file_path)

        # 执行随机采样，将点云采样为2048个点
        sampled_points = random_sample(points, num_points)

        # 计算点云数据的质心
        centroid = np.mean(sampled_points, axis=0)
        centroid = np.around(centroid, decimals=3)

        # 将点云归一化，使其质心在原点
        sampled_points = sampled_points - centroid

        # 计算点云数据的最大范数，即点云数据中点到原点的最远距离
        dist = np.max(np.sqrt(np.sum(sampled_points ** 2, axis=1)))
        sampled_points = sampled_points / dist
        # 获取文件名中的id
        base_name = os.path.basename(file_path)
        id_part = base_name.split('_')[-1]
        id = id_part.split('.')[0]
        new_file_name = f"pointcloud_{centroid}_{dist:0.3f}_{int(id):04d}.npy"
        save_path = os.path.join(self.pcd_normalized_dir, new_file_name)
        np.save(save_path, sampled_points)
        return sampled_points


def random_sample(points, num_points, device=None):
    np.random.seed(3407)

    indices = np.random.choice(len(points), num_points, replace=False if len(points) >= num_points else True)

    sampled_points = points[indices]
    return sampled_points


if __name__ == '__main__':
    kinect_data_dir = '../data/kinect_data_0828'

    id = "0000"
    file_path = os.path.join(kinect_data_dir, f'pointcloud_{id}.npy')

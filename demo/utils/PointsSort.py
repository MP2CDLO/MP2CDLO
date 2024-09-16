import os
import sys

work_path = os.path.abspath(os.path.join(os.getcwd(), "../utils"))
print(work_path)
sys.path.append('../')
sys.path.append(work_path)
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from typing import Tuple
import torch
from pytorch3d.ops import knn_points
from extensions.pointops.functions import pointops
from PCDMixture import Mixture


def group_points(pointclouds: torch.Tensor, num_points: int, group_size: int):
    """
    Groups the points in the input point clouds.
    Args:
        **pointclouds**: Input point clouds as a tensor of shape
            `(minibatch, N, dim)`.
        **num_points**: Number of points to sample.
        **group_size**: Number of points in each group.
    Returns:
        **group_points**: A tensor of shape `(minibatch, num_points, group_size, dim)`.
        **sampled_points**: A tensor of shape `(minibatch, num_points, dim)`.
    """
    # 对输入点云进行FPS采样和KNN搜索，实现分组
    sample_points = pointops.fps(pointclouds, num_points)  # [b, num_points, 3]
    knn_index = pointops.knn(x=sample_points, src=pointclouds, k=group_size)[0]  # [b, num_points, group_size]

    # 将 pointcloud 和 knn_index 调整为适合 torch.gather 的形状
    pointclouds = pointclouds.unsqueeze(1).expand(-1, num_points, -1, -1)
    knn_index = knn_index.unsqueeze(-1).expand(-1, -1, -1, 3)

    group_points = torch.gather(pointclouds, 2, knn_index)  # [b, num_points, group_size, 3]
    group_means = torch.mean(group_points, dim=2)  # [b, num_points, 3]

    return group_points, group_means


def get_point_covariances_sampled(
        points_padded: torch.Tensor,
        sampled_points: torch.Tensor,
        neighborhood_size: int,
        num_points_per_cloud: torch.Tensor = None,
        num_points_per_sampled_cloud: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the per-point covariance matrices by of the 3D locations of
    K-nearest neighbors of each point.

    Args:
        **points_padded**: Input point clouds as a padded tensor
            of shape `(minibatch, num_points, dim)`.
        **num_points_per_cloud**: Number of points per cloud
            of shape `(minibatch,)`.
        **neighborhood_size**: Number of nearest neighbors for each point
            used to estimate the covariance matrices.
        **sampled_points**: Sampled points from the point clouds.

    Returns:
        **covariances**: A batch of per-point covariance matrices
            of shape `(minibatch, dim, dim)`.
        **k_nearest_neighbors**: A batch of `neighborhood_size` nearest
            neighbors for each of the sampled points
            of shape `(minibatch, num_sampled_points, neighborhood_size, dim)`.
    """

    # get K nearest neighbor idx for each sampled point
    k_nearest_neighbors = knn_points(
        sampled_points,
        points_padded,
        lengths1=num_points_per_sampled_cloud,
        lengths2=num_points_per_cloud,
        K=neighborhood_size,
        return_nn=True,
    ).knn
    # obtain the mean of the neighborhood
    pt_mean = k_nearest_neighbors.mean(2, keepdim=True)
    # compute the diff of the neighborhood and the mean of the neighborhood
    central_diff = k_nearest_neighbors - pt_mean
    # per-nn-point covariances
    per_pt_cov = central_diff.unsqueeze(4) * central_diff.unsqueeze(3)
    # per-point covariances
    covariances = per_pt_cov.mean(2)

    return covariances, k_nearest_neighbors


def _disambiguate_vector_directions(pcl, knns, vecs: torch.Tensor) -> torch.Tensor:
    """
    Disambiguates normal directions according to [1].

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    # parse out K from the shape of knns
    K = knns.shape[2]
    # the difference between the mean of each neighborhood and
    # each element of the neighborhood
    df = knns - pcl[:, :, None]
    # projection of the difference on the principal direction
    proj = (vecs[:, :, None] * df).sum(3)
    # check how many projections are positive
    n_pos = (proj > 0).type_as(knns).sum(2, keepdim=True)
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * K)).type_as(knns)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs


def estimate_pointcloud_local_coord_frames(
        pointclouds: torch.Tensor,
        sampled_points: torch.Tensor,
        neighborhood_size: int = 50,
        disambiguate_directions: bool = True,
):
    if pointclouds.dim() != 3:
        pointclouds = pointclouds.unsqueeze(0)
    if sampled_points.dim() != 3:
        sampled_points = sampled_points.unsqueeze(0)
    b, N, dim = pointclouds.shape
    _, n, _ = sampled_points.shape

    if (N <= neighborhood_size):
        raise ValueError(
            "The neighborhood_size argument has to be"
            + " >= size of each of the point clouds."
        )

    # compute the covariance matrices
    # get knn neighbors and covariances
    cov, knns = get_point_covariances_sampled(
        points_padded=pointclouds,
        sampled_points=sampled_points,
        neighborhood_size=neighborhood_size,
    )
    # get the local coord frames as principal directions of
    # the per-point covariance
    # this is done with torch.symeig / torch.linalg.eigh, which returns the
    # eigenvectors (=principal directions) in an ascending order of their
    # corresponding eigenvalues, and the smallest eigenvalue's eigenvector
    # corresponds to the normal direction; or with a custom equivalent.
    # curvatures: 特征值 [b,n,3], local_coord_frames: 特征向量 [b,n,3,3]
    curvatures, local_coord_frames = torch.linalg.eigh(cov)

    # disambiguate the directions of individual principal vectors
    if disambiguate_directions:
        # disambiguate normal
        n = _disambiguate_vector_directions(
            sampled_points, knns, local_coord_frames[:, :, :, 0]
        )
        # disambiguate the main curvature
        z = _disambiguate_vector_directions(
            sampled_points, knns, local_coord_frames[:, :, :, 2]
        )
        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)
        # cat to form the set of principal directions
        local_coord_frames = torch.stack((n, y, z), dim=3)

    return curvatures, local_coord_frames


def test_get_point_covariances_sampled():
    # 创建假数据
    minibatch = 10
    num_points = 100
    num_sampled_points = 50
    dim = 3
    neighborhood_size = 5
    num_sampled_points = 50

    points_padded = torch.randn(minibatch, num_points, dim)
    num_points_per_cloud = torch.full((minibatch,), num_points)
    sampled_points = torch.randn(minibatch, num_sampled_points, dim)
    num_points_per_sampled_cloud = torch.full((minibatch,), num_sampled_points)

    # 运行函数
    covariances, loocal_frame = estimate_pointcloud_local_coord_frames(
        points_padded,
        sampled_points,
        neighborhood_size,
        disambiguate_directions=True,
    )

    print(covariances.shape)
    # print(covariances)
    print(loocal_frame.shape)
    print(loocal_frame[0, 0, :, 0])
    print(loocal_frame[0, 0, :, 2])
    print(torch.dot(loocal_frame[0, 0, :, 1], loocal_frame[0, 0, :, 2]))


# test_get_point_covariances_sampled()

def draw_lines_between_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point_pair in points:
        point1 = point_pair[0]
        point2 = point_pair[1]
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])

    plt.show()


def visualize_selected_points(point_pairs=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if point_pairs is not None:
        for point_pair in point_pairs:
            point1 = point_pair[0]
            point2 = point_pair[1]
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])

    # 设置相同的比例
    def set_axes_equal(ax):
        '''确保x, y, z轴具有相同的比例.'''
        extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        centers = np.mean(extents, axis=1)
        max_size = max(abs(extents[:, 1] - extents[:, 0]))
        r = max_size / 2
        ax.set_xlim([centers[0] - r, centers[0] + r])
        ax.set_ylim([centers[1] - r, centers[1] + r])
        ax.set_zlim([centers[2] - r, centers[2] + r])

    set_axes_equal(ax)

    plt.show()


"""
获得points中距离sample_points中每个点最近的k个点的索引
"""


def knn(points, sample_points, k):
    tree = KDTree(points)
    _, indices = tree.query(x=sample_points, k=k)  # k+1 because the point itself is included
    return indices  # include the point itself


"""
用Kd树来加速FPS,寻找全局最远点
"""


def fps_with_kdtree(points, num_points, start_points=None):
    N, _ = points.shape
    selected_points = np.zeros((num_points, 3))
    if start_points is None:
        selected_points[0] = points[np.random.randint(0, N)]
    else:
        selected_points[0] = start_points
    distances = np.full(N, np.inf)
    tree = KDTree(selected_points[:1, :])

    for i in range(1, num_points):
        distances = np.minimum(distances, tree.query(points)[0] ** 2)
        selected_points[i] = points[np.argmax(distances)]
        tree = KDTree(selected_points[:i + 1, :])

    return selected_points


def get_normal(pointcloud, num_points, group_size=100):
    child_group_num = 3
    child_group_size = group_size / (child_group_num - 1)
    # 对输入点云进行FPS采样和KNN搜索，实现分组
    sample_points = fps_with_kdtree(pointcloud, num_points)
    knn_index = knn(pointcloud, sample_points, group_size)
    # print(knn_index.shape)
    group_points = pointcloud[knn_index]
    # print(group_points.shape)
    # 对每个分组计算法向量
    child_group_means = np.zeros((group_points.shape[0], child_group_num, 3))
    for i in range(group_points.shape[0]):
        mean = np.mean(group_points[i], axis=0)
        sample_points = fps_with_kdtree(points=group_points[i], num_points=child_group_num, start_points=mean)
        child_knn_index = knn(group_points[i], sample_points, child_group_size)
        child_group_points = group_points[i][child_knn_index]
        for j in range(child_group_points.shape[0]):
            child_group_means[i][j] = np.mean(child_group_points[j], axis=0)
    return group_points, child_group_means  # 返回的是分组后的点和每个分组的特征点，特征点格式是（num_points,child_group_num,3）


def compute_normal(group_means):
    normals = np.zeros((group_means.shape[0], 2, 3))
    for i in range(group_means.shape[0]):
        centriod = group_means[i][0]
        direction = (group_means[i][1] - group_means[i][2]) / np.linalg.norm(group_means[i][1] - group_means[i][2])
        normals[i][0] = centriod
        normals[i][1] = direction
    return normals


def compute_distance(points1, points2):
    """

    return distance [N, M]
    """
    points1 = points1.unsqueeze(1)
    points2 = points2.unsqueeze(0)
    return torch.sqrt(torch.sum((points1 - points2) ** 2, dim=-1))


def compute_angles(points1, points2):
    """

    return angles [N, M]
    """
    points1 = points1.unsqueeze(1)
    points2 = points2.unsqueeze(0)

    dot_product = torch.sum(points1 * points2, dim=-1)

    norm1 = torch.norm(points1, dim=-1)
    norm2 = torch.norm(points2, dim=-1)

    cos_angle = dot_product / (norm1 * norm2)

    return torch.clip(cos_angle, -1, 1)  # 将弧度转换为度


def get_connections_loss(points, directions, start_point=None, m=1):
    """

    points [N,3] N<=30
    directions [N,3]
    """
    n, dim = points.shape
    points_location = points
    points_direction = directions

    connection_pairs = torch.zeros((n - 1, 2, 3)).to('cuda')
    i = 0

    connect_loss = 0
    connect_state = torch.zeros(n)
    connect_list = [[] for _ in range(n)]
    unconnected_set = list(range(n))
    connected_set = [unconnected_set.pop(np.random.randint(0, len(unconnected_set)))]

    endpoints = [connected_set[0], connected_set[0]]

    while unconnected_set:
        unconnected_points_location = points_location[unconnected_set]
        end_points_location = points_location[endpoints]
        unconnected_points_direction = points_direction[unconnected_set]
        end_points_direction = points_direction[endpoints]

        loss_matrix = m * compute_distance(end_points_location, unconnected_points_location) + (1 - m) * compute_angles(
            end_points_direction, unconnected_points_direction)  # 计算连接损失

        min_val = torch.min(loss_matrix)

        min_val, min_idx = torch.min(loss_matrix.view(-1), dim=0)
        row = min_idx // loss_matrix.shape[1]
        col = min_idx % loss_matrix.shape[1]

        connection_pairs[i] = torch.stack([points_location[unconnected_set[col]], points_location[endpoints[row]]])
        i += 1

        connected_set.append(unconnected_set[col])
        connect_state[unconnected_set[col]] += 1
        connect_state[endpoints[row]] += 1
        connect_list[endpoints[row]].append(unconnected_set[col])
        connect_list[unconnected_set[col]].append(endpoints[row])
        unconnected_set.pop(col)
        connect_loss += min_val
        endpoints = torch.where(connect_state == 1)[0].tolist()
    return connect_loss, connection_pairs, connect_list


def generate_connection_list(connect_list):
    connection_list = []
    start_index = [i for i, sublist in enumerate(connect_list) if len(sublist) == 1][0]
    connection_list.append(start_index)
    connection_list.append(connect_list[start_index][0])
    for i in range(2, len(connect_list)):
        if connect_list[connection_list[i - 1]][0] != connection_list[i - 2]:
            connection_list.append(connect_list[connection_list[i - 1]][0])
        else:
            connection_list.append(connect_list[connection_list[i - 1]][1])
    return connection_list


def connect_keypoints(pointclouds, keypoints):
    curvatures, local_coord_frames = estimate_pointcloud_local_coord_frames(pointclouds, keypoints)
    directions = local_coord_frames[0, :, :, 2]

    connection_loss, connection_pairs, connect_list = get_connections_loss(keypoints, directions)

    connection_list = generate_connection_list(connect_list)
    return connection_loss, connection_list, connection_pairs


if __name__ == "__main__":
    dir_path = '../predictions'
    group_num = '0030'

    num_group = 30
    group_size = 64
    GMMmixture = Mixture(dir_path, num_group=num_group, group_size=group_size)
    downsampled_cloud, labels, centers = GMMmixture.run(group_num)
    connection_loss, connecntion_list, connection_pairs = connect_keypoints(downsampled_cloud, centers)

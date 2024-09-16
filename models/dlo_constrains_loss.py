from typing import Tuple
import torch
from pytorch3d.ops import knn_points
from extensions.pointops.functions import pointops

def angular_cost(group_means,points_directions):
    """
    Compute the angular cost between the mean of the group and the direction of the points.
    Args:
        **group_means**: A tensor of shape `(minibatch, num_points, dim)`.
        **points_directions**: A tensor of shape `(minibatch, num_points, dim)`.
    Returns:
        **angular_cost**: A tensor of shape `(minibatch, num_points)`.
    """
    num_points = group_means.shape[1]
    knn_index = pointops.knn(x=group_means, src=group_means,k=3)[0] # [b, num_points, group_size]
    

    points_directions = points_directions.unsqueeze(1).expand(-1, num_points, -1, -1)
    knn_index = knn_index.unsqueeze(-1).expand(-1, -1, -1, 3)

    group_directions = torch.gather(points_directions, 2, knn_index) # [b, num_points, group_size, 3]

    angular_cost = 1-(torch.abs(torch.sum(group_directions[:,:,0,:] * group_directions[:,:,1,:],dim=-1))+torch.abs(torch.sum(group_directions[:,:,0,:] * group_directions[:,:,2,:], dim=-1)))/2
    loss = torch.sum(angular_cost) / (angular_cost.shape[0]*angular_cost.shape[1])
    return loss


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
    sample_points = pointops.fps(pointclouds, num_points) # [b, num_points, 3]
    knn_index = pointops.knn(x=sample_points, src=pointclouds,k=group_size)[0] # [b, num_points, group_size]

    pointclouds = pointclouds.unsqueeze(1).expand(-1, num_points, -1, -1)
    knn_index = knn_index.unsqueeze(-1).expand(-1, -1, -1, 3)

    group_points = torch.gather(pointclouds, 2, knn_index) # [b, num_points, group_size, 3]
    group_means = torch.mean(group_points, dim=2) # [b, num_points, 3]

    return group_points, group_means

def get_point_covariances_sampled(
    points_padded: torch.Tensor,
    sampled_points: torch.Tensor,
    neighborhood_size: int,
    num_points_per_cloud: torch.Tensor =None,
    num_points_per_sampled_cloud: torch.Tensor=None,
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
    b , N , dim = pointclouds.shape
    _ , n , _ = sampled_points.shape

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

    return curvatures, local_coord_frames,knns

def test_get_point_covariances_sampled():
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


    covariances, loocal_frame = estimate_pointcloud_local_coord_frames(
        points_padded,
        sampled_points,
        neighborhood_size,
        disambiguate_directions=True,
    )

  
    print(covariances.shape)
    # print(covariances)
    print(loocal_frame.shape)
    print(loocal_frame[0,0,:,0])
    print(loocal_frame[0,0,:,2])
    print(torch.dot(loocal_frame[0,0,:,1],loocal_frame[0,0,:,2]))

if __name__ == "__main__":
    test_get_point_covariances_sampled()

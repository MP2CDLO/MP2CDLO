import numpy as np
import scipy
import bezier
from geomdl import BSpline, utilities


class B_spline():
    def __init__(self, dtype='BSpline', delta=0.01):
        self.dtype = dtype
        self.delta = delta

    def run(self, control_points):

        # np.save("error_points.npy", control_points)

        if self.dtype == 'BSpline':
            # Create a 3-dimensional B-spline Curve
            curve = BSpline.Curve()
            # Set degree
            curve.degree = 3
            # Set control points
            curve.ctrlpts = control_points
            # Set knot vector
            curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
            # Set evaluation delta (controls the number of curve points)
            curve.delta = self.delta
            # Get curve points (the curve will be automatically evaluated)
            curve_points = curve.evalpts
        elif self.dtype == 'Bezier':

            group_size = 4

            num_groups = len(control_points) // group_size

            curve_points = []
            for i in range(num_groups):
                group_control_points = control_points[i * group_size:(i + 1) * group_size]
                nodes1 = np.asfortranarray(
                    np.array(group_control_points).T
                )
                curve = bezier.Curve(nodes1, degree=group_size - 1)

                group_curve_points = curve.evaluate_multi(np.linspace(0.0, 1.0, 200)).T.tolist()
                curve_points.extend(group_curve_points)
        elif self.dtype == 'BSpline2':

            control_points = np.array(control_points)

            x = control_points[:, 0]
            y = control_points[:, 1]
            z = control_points[:, 2]

            tck, u = scipy.interpolate.splprep([x, y, z], s=0.0001)

            new_points = np.array(scipy.interpolate.splev(np.linspace(0, 1, int(10 // self.delta)), tck))

            curve_points = list(new_points.transpose())
        else:
            raise ValueError("Invalid dtype")

        return control_points, curve_points

    def down_sample(self, pointclouds, num=50):
        # Generate indices for downsampling
        indices = np.linspace(0, len(pointclouds) - 1, num, dtype=int)
        # print(indices)
        # Downsample the point cloud
        downsampled_pointclouds = [pointclouds[i] for i in indices]

        return downsampled_pointclouds


if __name__ == '__main__':
    import sys

    sys.path.append('../')
    from utils.PCDMixture import Mixture
    from demo.utils.Visualization import Visualizer
    from utils.PointsSort import connect_keypoints

    dir_path = '../experiments/results'
    group_num = '0100'

    num_group = 30
    group_size = 64
    mixture = Mixture(dir_path, num_group=num_group, group_size=group_size)
    Bspline = B_spline(dtype='BSpline2', delta=0.01)
    downsampled_cloud, partial_cloud, centers = mixture.run(group_num, dtype='sklearn-Kmeans')

    connection_loss, connection_list, connection_pairs = connect_keypoints(downsampled_cloud, centers)

    control_points = [centers[i].tolist() for i in connection_list]
    control_points, curve_points = Bspline.run(control_points=control_points)

    downsampled_clouds = Bspline.down_sample(curve_points, num=50)
    visualizer = Visualizer()
    visualizer.visualize_point_clouds(downsampled_clouds, partial_cloud)

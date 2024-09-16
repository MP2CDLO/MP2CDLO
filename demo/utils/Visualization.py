import open3d as o3d
import json
import open3d as o3d
import numpy as np
import queue
from threading import Thread


class Visualizer():
    def __init__(self, width=1280, height=720, fixed_camera=None):
        self.width = width
        self.height = height
        self.fixed_camera = fixed_camera

        self.rgb_queue = queue.Queue()
        self.rgb_vis = Thread(target=self.visualization_with_thread,
                              args=(self.rgb_queue, 'RGB', (self.width, self.height)), daemon=True)
        self.rgb_vis.start()

        self.pcd_queue = queue.Queue()
        self.pcd_vis = Thread(target=self.visualization_with_thread,
                              args=(self.pcd_queue, 'Pointcloud', (self.width, self.height)), daemon=True)
        self.pcd_vis.start()

    def load_camera_parameters(self, json_file):
        with open(json_file, 'r') as f:
            camera_params = json.load(f)

        extrinsic = np.array(camera_params["extrinsic"]).reshape((4, 4))
        intrinsic_matrix = camera_params["intrinsic"]["intrinsic_matrix"]
        width = camera_params["intrinsic"]["width"]
        height = camera_params["intrinsic"]["height"]

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height,
                                                      intrinsic_matrix[0], intrinsic_matrix[4],
                                                      intrinsic_matrix[6], intrinsic_matrix[7])
        return extrinsic, intrinsic

    def visualization_with_thread(self, message_queue, name='viewer', resolution=(1280, 720)):
        vis = o3d.visualization.Visualizer()
        vis.create_window(name, resolution[0], resolution[1])
        if self.fixed_camera is not None:
            extrinsic, intrinsic = self.load_camera_parameters(json_file=self.fixed_camera)
            view_control = vis.get_view_control()
            camera_parameters = view_control.convert_to_pinhole_camera_parameters()
            camera_parameters.extrinsic = extrinsic.T
            camera_parameters.intrinsic = intrinsic
            view_control.convert_from_pinhole_camera_parameters(camera_parameters)
        while True:
            try:

                data = message_queue.get(timeout=0.01)

                vis.clear_geometries()
                for item in data:
                    vis.add_geometry(item)
                    vis.update_geometry(item)
                vis.update_renderer()

                # vis.capture_screen_image(f"../experiments/pcd_image_test.png")
            except queue.Empty:
                pass

            vis.poll_events()

            # time.sleep(0.01)

    def create_sphere_at_point(self, point, radius=0.004, color=[88 / 255, 182 / 255,
                                                                 233 / 255]):  # [128 / 255, 0, 128 / 255] [192/255, 50/255, 26/255]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        return sphere

    def create_cylinder_between_points(self, point1, point2, radius=0.0015,
                                       color=[157 / 255, 172 / 255, 203 / 255]):  # [212/255, 86/255, 46/255]
        # print
        direction = point2 - point1
        length = np.linalg.norm(direction)
        direction = direction / length

        z_axis = np.array([0, 0, 1])
        rotation_matrix = np.eye(3)
        if not np.allclose(direction, z_axis):
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.dot(z_axis, direction))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, length)
        cylinder.paint_uniform_color(color)
        cylinder.translate([0, 0, length / 2])  # Move the cylinder along the Z axis
        cylinder.rotate(rotation_matrix, center=[0, 0, 0])
        cylinder.translate(point1)  # Move the cylinder to the correct location
        return cylinder

    def create_connect_between_points(self, skeleton_points):

        skeleton_points = np.array(skeleton_points)

        pcd_skeleton = o3d.geometry.PointCloud()
        pcd_skeleton.points = o3d.utility.Vector3dVector(skeleton_points)
        pcd_skeleton.paint_uniform_color([1, 0, 0])

        combined_mesh = o3d.geometry.TriangleMesh()

        for i in range(len(skeleton_points)):
            sphere = self.create_sphere_at_point(skeleton_points[i])
            combined_mesh += sphere
            if i < len(skeleton_points) - 1:
                cylinder = self.create_cylinder_between_points(skeleton_points[i], skeleton_points[i + 1])
                combined_mesh += cylinder
        return combined_mesh

    def visualize_point_clouds(self, skeleton_points, partial_cloud, fix_camera=False, json_file="camera_info.json"):

        pcd_partial = partial_cloud

        combined_mesh = self.create_connect_between_points(skeleton_points)

        self.pcd_queue.put([pcd_partial, combined_mesh])

    def visualize_points_cloud_and_png(self, skeleton_points, partial_cloud, rgb):
        pcd_partial = o3d.geometry.PointCloud()
        pcd_partial.points = o3d.utility.Vector3dVector(partial_cloud)

        combined_mesh = self.create_connect_between_points(skeleton_points)

        self.pcd_queue.put([pcd_partial, combined_mesh])
        self.rgb_queue.put([rgb])

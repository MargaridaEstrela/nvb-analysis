import os
import cv2
import numpy as np
import scipy.spatial
import open3d as o3d
import pycolmap
import matplotlib.pyplot as plt

COLMAP_PATH = "../colmap_project/n11_5_colmap_v3/sparse/0"
PROJECT_PATH = "../colmap_project/n11_5_colmap_v3"
IMAGE_NAME = "IMG_6596.jpg"

# ---------------------------
# Load COLMAP Reconstruction & Point Cloud
# ---------------------------
def load_colmap_reconstruction(colmap_path):
    """Load COLMAP reconstruction from binary files."""
    return pycolmap.Reconstruction(colmap_path)

def load_point_cloud_from_colmap(reconstruction, max_points=20000):
    """Extract 3D points from COLMAP and return an Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    points, colors = [], []

    for i, (point3D_id, point3D) in enumerate(reconstruction.points3D.items()):
        if i >= max_points:
            break
        points.append(point3D.xyz)
        colors.append(point3D.color / 255.0)  # Normalize colors

    if not points:
        raise ValueError("No 3D points loaded from COLMAP.")

    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

# ---------------------------
# Load Image and Camera Parameters
# ---------------------------
def get_camera_intrinsics(reconstruction, image_name):
    """Extract camera intrinsic matrix from COLMAP reconstruction."""
    for img_id, image in reconstruction.images.items():
        if os.path.basename(image.name) == image_name:
            camera = reconstruction.cameras[image.camera_id]
            camera_model = str(camera.model)

            if "SIMPLE_PINHOLE" in camera_model or "PINHOLE" in camera_model:
                fx = camera.params[0]
                fy = camera.params[0] if "SIMPLE_PINHOLE" in camera_model else camera.params[1]
                cx, cy = camera.params[-2], camera.params[-1]
            elif "SIMPLE_RADIAL" in camera_model or "RADIAL" in camera_model:
                fx = fy = camera.params[0]
                cx, cy = camera.params[1], camera.params[2]
            elif "OPENCV" in camera_model:
                fx, fy = camera.params[0], camera.params[1]
                cx, cy = camera.params[2], camera.params[3]
            else:
                raise ValueError(f"Unsupported camera model: {camera_model}")

            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    raise ValueError(f"Camera intrinsics not found for {image_name}")

def get_camera_pose(reconstruction, image_name):
    """Extract camera pose (extrinsics) from COLMAP."""
    for img_id, image in reconstruction.images.items():
        if os.path.basename(image.name) == image_name:
            R = image.cam_from_world.rotation.matrix()
            t = np.array(image.cam_from_world.translation).reshape(3, 1)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            return T
    raise ValueError(f"Camera pose not found for {image_name}")

# ---------------------------
# Project 3D Points to 2D Image Space
# ---------------------------
def project_point_cloud(pcd, K, T, image_shape):
    """Project 3D point cloud into 2D image space and filter points inside image."""
    points = np.asarray(pcd.points)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    points_cam = (T @ points_h.T)[:3, :]
    projected = K @ points_cam
    projected /= projected[2, :]  # Normalize by depth

    projected_2d = projected[:2, :].T  # Shape: (N, 2)
    depths = points_cam[2, :]  # Extract depth values

    # Filter points inside image dimensions
    height, width = image_shape[:2]
    valid_idx = (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < width) & \
                (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < height)

    return projected_2d[valid_idx], depths[valid_idx]

# ---------------------------
# Feature Selection for Depth Scaling
# ---------------------------
selected_features = []

def select_from_heatmap(event, x, y, flags, param):
    """ Mouse click event to select features from the heatmap. """
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_features.append((x, y))
        print(f"Selected Feature {len(selected_features)}: ({x}, {y})")

def get_feature_indices_from_heatmap(image, projected_points):
    """Allows user to select two points on the heatmap and find the closest keypoints."""
    global selected_features
    selected_features = []

    # Draw the keypoints on the image before selection
    for kp in projected_points:
        cv2.circle(image, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

    cv2.imshow("Select Two Features from Heatmap", image)
    cv2.setMouseCallback("Select Two Features from Heatmap", select_from_heatmap)

    print("Waiting for user to select two features...")
    while len(selected_features) < 2:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print("Features selected successfully!")

    # Find closest keypoints
    kdtree = scipy.spatial.KDTree(projected_points)
    _, indices = kdtree.query(selected_features, k=1)

    print(f"\nSelected Pixel Coordinates: {selected_features[0]}, {selected_features[1]}")
    print(f"Corresponding COLMAP 3D Point Indices: {indices[0]}, {indices[1]}")

    return indices[0], indices[1]

# ---------------------------
# Compute Depth Scale Factor and Update Heatmap
# ---------------------------
def compute_depth_scale(image, projected_points, depths):
    """Select two features from the heatmap and compute the new depth scale."""
    print("\nSelect two features for depth recalibration...")
    idx1, idx2 = get_feature_indices_from_heatmap(image, projected_points)

    print(f"Computing depth scale using features {idx1} and {idx2}")

    colmap_distance = np.abs(depths[idx1] - depths[idx2])
    print(f"COLMAP 3D Distance: {colmap_distance:.4f} units")

    while True:
        try:
            known_real_distance = float(input("\nEnter real-world distance between the features (in meters): "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    scale_factor = known_real_distance / colmap_distance
    print(f"Computed Scale Factor: {scale_factor:.4f}")

    scaled_depths = depths * scale_factor
    return scale_factor, scaled_depths

def overlay_depth_heatmap(image_path, projected_points, scaled_depths):
    """Overlay a depth heatmap on the original image with corrected scale and outlier removal."""
    
    # Load grayscale image
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_height, img_width = image_gray.shape

    inside_image_mask = (
        (projected_points[:, 0] >= 0) & (projected_points[:, 0] < img_width) &
        (projected_points[:, 1] >= 0) & (projected_points[:, 1] < img_height)
    )

    visible_points = projected_points[inside_image_mask]
    visible_depths = scaled_depths[inside_image_mask]

    if len(visible_depths) == 0:
        print("No valid depth points inside the image!")
        return

    q1, q3 = np.percentile(visible_depths, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    inlier_mask = (visible_depths >= lower_bound) & (visible_depths <= upper_bound)
    filtered_points = visible_points[inlier_mask]
    filtered_depths = visible_depths[inlier_mask]

    depth_max = np.max(filtered_depths)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image_gray, cmap='gray')

    sc = ax.scatter(
        filtered_points[:, 0], 
        filtered_points[:, 1], 
        c=filtered_depths, 
        cmap='jet', 
        s=5, 
        alpha=0.75, 
        vmin=0, 
        vmax=depth_max
    )

    cbar = plt.colorbar(sc, label="Depth (meters)")
    plt.title("Depth of Matched Keypoints")

    annot = ax.annotate(
        "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    annot.set_visible(False)

    def update_annot(event):
        """Update annotation with depth value on hover."""
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                index = ind["ind"][0]
                pos = filtered_points[index]
                depth_value = filtered_depths[index]

                annot.xy = pos
                annot.set_text(f"Depth: {depth_value:.2f}m")
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", update_annot)

    plt.show()


# ---------------------------
# Main Execution
# ---------------------------
def main():
    image_path = os.path.join(PROJECT_PATH, "images", IMAGE_NAME)

    reconstruction = load_colmap_reconstruction(COLMAP_PATH)
    pcd = load_point_cloud_from_colmap(reconstruction)
    K = get_camera_intrinsics(reconstruction, IMAGE_NAME)
    T = get_camera_pose(reconstruction, IMAGE_NAME)

    image = cv2.imread(image_path)
    projected_points, depths = project_point_cloud(pcd, K, T, image.shape)

    scale_factor, scaled_depths = compute_depth_scale(image, projected_points, depths)
    overlay_depth_heatmap(image_path, projected_points, scaled_depths)

if __name__ == "__main__":
    main()
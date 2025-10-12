from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # TODO: solve PnP problem using OpenCV
    # Hint: you may use "Descriptors Matching and ratio test" first
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc_query, desc_model, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points_query = np.array([kp_query[m.queryIdx] for m in good_matches])
    points_model = np.array([kp_model[m.trainIdx] for m in good_matches])
    return cv2.solvePnPRansac(points_model, points_query, cameraMatrix, distCoeffs)

def rotation_error(R1, R2):
    #TODO: calculate rotation error
    R1, R2 = R.from_quat(R1), R.from_quat(R2)
    error = R1 * R2.inv()
    return error.magnitude() # * (180.0 / np.pi)  # convert to degrees

def translation_error(t1, t2):
    #TODO: calculate translation error
    return np.linalg.norm(t1 - t2)

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    #TODO: visualize the camera pose using Open3D
    geometry_list = []

    # load point cloud
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    geometry_list.append(pcd)

    # create camera pyramid
    s = 0.05
    h = 0.05
    trajectory_points = []
    for c2w in Camera2World_Transform_Matrixs:
        points = np.array([
            [0, 0, 0, 1],
            [-h, -h, s, 1],
            [h, -h, s, 1],
            [h, h, s, 1],
            [-h, h, s, 1]
        ])
        points = (c2w @ points.T).T[:, :3]
        trajectory_points.append(points[0, :3])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

        # create lineSet
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(points)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.paint_uniform_color([1, 0, 0])
        geometry_list.append(lineset)
    
    # create trajectory line
    trajectory_points_np = np.array(trajectory_points)
    lines = [[i, i + 1] for i in range(len(trajectory_points_np) - 1)]
    trajectory_lineset = o3d.geometry.LineSet()
    trajectory_lineset.points = o3d.utility.Vector3dVector(trajectory_points_np)
    trajectory_lineset.lines = o3d.utility.Vector2iVector(lines)
    trajectory_lineset.paint_uniform_color([0, 1, 0])
    geometry_list.append(trajectory_lineset)

    o3d.visualization.draw_geometries(geometry_list)

def create_cube(vertices, density=20):
    points = []
    colors = []
    
    # define faces and colors
    faces = [
        ([0, 1, 5, 4], [255, 0, 0]),    # front (red)
        ([2, 3, 7, 6], [0, 255, 0]),    # back (green)
        ([6, 7, 5, 4], [0, 0, 255]),    # bottom (blue)
        ([2, 3, 1, 0], [255, 255, 0]),  # top (yellow)
        ([3, 1, 5, 7], [255, 0, 255]),  # left (magenta)
        ([2, 0, 4, 6], [0, 255, 255])   # right (cyan)
    ]
    
    # generate points for each face
    for face_vertices_idx, color in faces:
        face_vertices = vertices[face_vertices_idx]
        # grid of points on each face
        for i in range(density):
            for j in range(density):
                alpha = i / (density - 1)
                beta = j / (density - 1)
                # bilinear interpolation
                point = (1 - alpha) * (1 - beta) * face_vertices[0] + \
                       alpha * (1 - beta) * face_vertices[1] + \
                       alpha * beta * face_vertices[2] + \
                       (1 - alpha) * beta * face_vertices[3]
                points.append(point)
                colors.append(color)
    
    return np.array(points), np.array(colors)

def draw_cude(img, rvec, tvec):
    # load cube
    cube_transform = np.load('cube_transform_mat.npy')
    cube_vertices = np.load('cube_vertices.npy')
    
    # generate cube points
    points_3d, colors = create_cube(cube_vertices)

    # apply the transformation to the points
    points_3d = (cube_transform @ np.vstack([points_3d.T, np.ones(points_3d.shape[0])])).T[:, :3]
    
    # project cube points
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, cameraMatrix, None)
    points_2d = points_2d.reshape(-1, 2)

    # calculate the depth of point
    R_mat, _ = cv2.Rodrigues(rvec)
    points_cam = (R_mat @ points_3d.T + tvec).T
    depths = points_cam[:, 2]
    
    # draw cube by painter's algorithm
    # sort by depth (far first)
    order = np.argsort(-depths)
    points_2d = points_2d[order]
    colors = colors[order]
    
    # draw points
    for (x, y), color in zip(points_2d.astype(int), colors):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 10, color.tolist(), -1)

    return img

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    # reorder_by_img_name
    subset = images_df[images_df["IMAGE_ID"].between(164, 294)].copy()
    subset.loc[:, 'order'] = subset['NAME'].str.extract(r'valid_img(\d+)\.jpg')[0].astype(int)
    subset_sorted = subset.sort_values('order')
    IMAGE_ID_LIST = subset_sorted["IMAGE_ID"].tolist()
    # IMAGE_ID_LIST = [i for i in range(164, 294)]
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []

    # video_write_for_ar_cube
    h, w = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ar_cube_video.mp4', fourcc, 10.0, (w, h))

    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        tvec = tvec.reshape(1,3) # Reshape translation vector
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        r_error = rotation_error(rotq, rotq_gt)
        t_error = translation_error(tvec, tvec_gt)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

        # draw cube
        rvec_gt = R.from_quat(rotq_gt).as_rotvec().reshape(3, 1)
        tvec_gt = tvec_gt.reshape(3, 1)
        res = draw_cude(rimg, rvec_gt, tvec_gt)
        out.write(res)

    out.release()

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    print("rotation differences: ", np.median(rotation_error_list))
    print("translation differences: ", np.median(translation_error_list))

    # TODO: result visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # TODO: calculate camera pose in world coordinate system
        R_mat, _ = cv2.Rodrigues(r)
        c2w = np.eye(4)
        c2w[:3, :3] = R_mat.T
        c2w[:3, 3] = -R_mat.T @ t.flatten()
        Camera2World_Transform_Matrixs.append(c2w)

    visualization(Camera2World_Transform_Matrixs, points3D_df)
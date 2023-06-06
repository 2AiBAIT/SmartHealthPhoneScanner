import math
import sys

import cv2
import scipy
from numpy.linalg import LinAlgError
from open3d.examples.open3d_example import draw_registration_result
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from scipy.linalg import norm
from functools import partial
import xlsxwriter
import open3d as o3d
import numpy as np
import trimesh
import os

from trimesh.transformations import rotation_matrix

dir_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/videos_exp1/utilizados_exp/1"
save = "/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align"
groud_truth_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1"
excel_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/Results_1_1.xlsx"
txt_path_colmap = '/home/hcorreia/PycharmProjects/instant-ngp/data/colmap.txt'
txt_path_run = '/home/hcorreia/PycharmProjects/instant-ngp/data/run.txt'

workbook = xlsxwriter.Workbook(excel_path)
worksheet1 = workbook.add_worksheet('model1')
worksheet2 = workbook.add_worksheet('model2')
worksheet3 = workbook.add_worksheet('model3')
worksheet4 = workbook.add_worksheet('model4')
worksheet5 = workbook.add_worksheet('model5')
worksheet6 = workbook.add_worksheet('model6')
worksheet7 = workbook.add_worksheet('model7')



def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1, :] *= -1
       R = np.dot(Vt.T, U.T)


    tr_atr = np.trace (np.dot (H.T, R))
    tr_yp1y = np.trace (np.dot (AA.T, AA))
    scale = np.sqrt (tr_atr / tr_yp1y)

    # translation
    t = centroid_B.T - np.dot (R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t, scale


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(src_tm: "<class 'trimesh'>", dst_tm: "<class 'trimesh'>",
        init_pose=None, max_iterations=20, tolerance=None, samplerate=1):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        samplerate: subsampling rate
    Output:
        T: final homogeneous transformation that maps A on to B
        MeanError: list, report each iteration's distance mean error
    """

    # get vertices and their normals
    src_pts = np.array(src_tm.vertices)
    dst_pts = np.array(dst_tm.vertices)
    src_pt_normals = np.array(src_tm.vertex_normals)
    dst_pt_normals = np.array(dst_tm.vertex_normals)

    # subsampling
    ids = np.random.uniform(0, 1, size=src_pts.shape[0])
    A = src_pts[ids < samplerate, :]
    A_normals = src_pt_normals[ids < samplerate, :]
    ids = np.random.uniform(0, 1, size=dst_pts.shape[0])
    B = dst_pts[ids < 1, :]
    B_normals = dst_pt_normals[ids < 1, :]

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 10000
    MeanError = []
    T_list = []
    StdError = []

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # match each point of source-set to closest point of destination-set,
        matched_src_pts = src[:m, :].T.copy()
        matched_dst_pts = dst[:m, indices].T

        # compute angle between 2 matched vertexs' normals
        matched_src_pt_normals = A_normals.copy()
        matched_dst_pt_normals = B_normals[indices, :]
        angles = np.zeros(matched_src_pt_normals.shape[0])
        for k in range(matched_src_pt_normals.shape[0]):
            v1 = matched_src_pt_normals[k, :]
            v2 = matched_dst_pt_normals[k, :]
            cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles[k] = np.arccos(cos_angle) / np.pi * 180

        # and reject the bad corresponding
        dist_threshold = np.inf
        dist_bool_flag = (distances < dist_threshold)
        angle_threshold = 10
        angle_bool_flag = (angles < angle_threshold)
        reject_part_flag = dist_bool_flag * angle_bool_flag

        matched_src_pts = matched_src_pts[reject_part_flag, :]
        matched_dst_pts = matched_dst_pts[reject_part_flag, :]

        # compute the transformation between the current source and nearest destination points
        T, _, _, scale = best_fit_transform(matched_src_pts, matched_dst_pts)

        # update the current source
        src = np.dot(T, src)
        src = np.dot(scale, src)

        # print iteration
        print('\ricp iteration: %d/%d ...' % (i+1, max_iterations), end='', flush=True)

        # check error
        mean_error = np.mean(distances[reject_part_flag])
        std_error = np.std(distances[reject_part_flag])
        print(mean_error)
        MeanError.append(mean_error)
        StdError.append(std_error)
        T_list.append(T)
        chamfer_dist = pcu.chamfer_distance(matched_dst_pts, matched_src_pts)
        hausdorff_dist = pcu.hausdorff_distance(matched_dst_pts, matched_src_pts)
        array_angles = np.array(angles)
        mean_angle = np.nanmean(array_angles)
        std_angle = np.nanmean(array_angles)
        # hull = scipy.spatial.ConvexHull(matched_dst_pts)
        # density = matched_dst_pts.shape[0] / hull.volume
        if tolerance is not None:
            if (mean_error < 1): #> mean_error
                if (prev_error > mean_error):
                    a = 1
                else:
                    break
        prev_error = mean_error

    # calculate final transformation
    T, _, _, scale= best_fit_transform(A, src[:m, :].T)

    return T, MeanError, chamfer_dist, hausdorff_dist, mean_angle, scale, MeanError, T_list, std_angle, StdError


def plot_trimesh(ax, tm, color='Reds'):
    ax.scatter3D(tm.vertices[:, 2], tm.vertices[:, 0], tm.vertices[:, 1],
                 c=(abs(tm.vertex_normals) @ np.array([0.299, 0.587, 0.114])),
                 cmap=color, alpha=0.2, marker='.')


def toO3d(tm, color):
    """put trimesh object into open3d object"""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tm.faces)
    mesh_o3d.compute_vertex_normals()
    vex_color_rgb = np.array(color)
    vex_color_rgb = np.tile(vex_color_rgb, (tm.vertices.shape[0], 1))
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vex_color_rgb)
    return mesh_o3d


def angle(v1, v2, acute):
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle



dst_tm = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_2_2_9000Test.ply')

dst_tm1 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm1 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_2_3_9000Test.ply')

dst_tm2 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm2 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_2_4_9000Test.ply')

dst_tm3 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm3 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_2_5_9000Test.ply')

dst_tm4 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm4 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_3_2_9000Test.ply')

dst_tm5 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm5 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_3_3_9000Test.ply')

dst_tm6 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm6 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_3_4_9000Test.ply')

dst_tm7 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp1/model1.ply')
src_tm7 = trimesh.load('/home/hcorreia/PycharmProjects/instant-ngp/data/Results1_align_1/model1_3_5_9000Test.ply')


H, MSE, Chamfer_dist, Hausdorff_dist, Mean_angle, scale, MeanError, T_list, std_angle, std_error = icp(src_tm, dst_tm, max_iterations=1, tolerance=1e-5)

H, MSE1, Chamfer_dist, Hausdorff_dist, Mean_angle, scale, MeanError, T_list, std_angle, std_error = icp(
	src_tm1, dst_tm1, max_iterations=1, tolerance=1e-5)

H, MSE2, Chamfer_dist, Hausdorff_dist, Mean_angle, scale, MeanError, T_list, std_angle, std_error = icp(
	src_tm2, dst_tm2, max_iterations=1, tolerance=1e-5)

H, MSE3, Chamfer_dist, Hausdorff_dist, Mean_angle, scale, MeanError, T_list, std_angle, std_error = icp(
	src_tm1, dst_tm3, max_iterations=1, tolerance=1e-5)


# tmp = min(MeanError)
# print(tmp)
# index = MeanError.index(tmp)
# std_min_error = std_error[index]
# res_tm = src_tm.copy()
# res_tm.apply_transform(T_list[index])
# res_tm.export(save_file)
# print(H)

list_quartis = []
plt.boxplot((MSE, MSE1, MSE2, MSE3))
plt.title("Box Plot")
plt.ylabel('Mean Error')
plt.xlabel('Number of imagens')
plt.savefig("Exp3")
plt.show()
plt.close()
plt.clf()
plt.cla()


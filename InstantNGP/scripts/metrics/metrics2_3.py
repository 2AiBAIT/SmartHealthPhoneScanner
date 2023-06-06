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

dir_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/videos_exp2/3"
save = "/home/hcorreia/PycharmProjects/instant-ngp/data/Results2_align"
groud_truth_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp2"
excel_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/Results_2_3.xlsx"
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

    return T, mean_error, chamfer_dist, hausdorff_dist, mean_angle, scale, MeanError, T_list, std_angle, StdError


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


for path in os.listdir(dir_path):
    list_listT1 = []
    list_listT2 = []
    list_listT3 = []
    if os.path.isfile(os.path.join(dir_path, path)):
        print(path)

        video_path = os.path.join(dir_path, path)
        list_n_frames = [2, 3, 4, 5]
        list_aabb_scale = [16]
        list_n_epochs = [5000, 7000, 9000]

        x = path.split(".")
        y = x[0].split('_')
        bigodes_name = y[0] + '.png'
        bigodes_name = os.path.join(save, bigodes_name)
        file_truth = y[0] + '.ply'
        groud_truth = os.path.join(groud_truth_path, file_truth)

        if (y[0] == 'model1'):
            worksheet = worksheet1
        elif (y[0] == 'model2'):
            worksheet = worksheet2
        elif (y[0] == 'model3'):
            worksheet = worksheet3
        elif (y[0] == 'model4'):
            worksheet = worksheet4
        elif (y[0] == 'model5'):
            worksheet = worksheet5
        elif (y[0] == 'model6'):
            worksheet = worksheet6
        else:
            worksheet = worksheet7

        if (y[1] == str(1)):
            row = 2
        elif (y[1] == str(2)):
            row = 62
        else:
            row = 122

        for i in range(len(list_n_frames)):

            # os.system(
            #     'python /home/hcorreia/PycharmProjects/instant-ngp/scripts/colmap2nerf.py --video_in ' + video_path + ' --video_fps ' + str(
            #         list_n_frames[i]))

            for k in range(len(list_n_epochs)):
                col = 6
                col = 11
                file = x[0] + '_' + str(list_n_frames[i]) + '_' + str(list_n_epochs[k]) + '.ply'
                bigodes_name2 = x[0] + '_' + str(list_n_frames[i]) + '_' + str(list_n_epochs[k]) + '.png'
                save_file = os.path.join(save, file)
                bigodes_name2 = os.path.join(save, bigodes_name2)
                print(save_file)

                # os.system(
                #     'python /home/hcorreia/PycharmProjects/instant-ngp/scripts/run.py --epochs ' + str(
                #         list_n_epochs[k]) + ' --save_mesh ' + save_file)

                dst_tm = trimesh.load(groud_truth)
                src_tm = trimesh.load(save_file)

                pcd1 = o3d.io.read_point_cloud(save_file)
                Y1 = len(np.asarray(pcd1.points))

                if (Y1 > 500000):
                    break
                else:

                    H, MSE, Chamfer_dist, Hausdorff_dist, Mean_angle, scale, MeanError, T_list, std_angle, std_error = icp(
                        src_tm, dst_tm, max_iterations=100, tolerance=1e-5)
                    tmp = min(MeanError)
                    print(tmp)
                    index = MeanError.index(tmp)
                    std_min_error = std_error[index]
                    res_tm = src_tm.copy()
                    res_tm.apply_transform(T_list[index])
                    res_tm.export(save_file)
                    print(H)

                    if (list_n_epochs[k] == int(5000)):
                        list_listT1.append(MeanError)
                    if (list_n_epochs[k] == int(7000)):
                        list_listT2.append(MeanError)
                    if (list_n_epochs[k] == int(9000)):
                        list_listT3.append(MeanError)

                    list_quartis = []
                    B = plt.boxplot((MeanError))
                    plt.title("Box Plot")
                    plt.ylabel('Mean Error')
                    plt.xlabel('Number of imagens')
                    plt.savefig(bigodes_name2)
                    plt.close()
                    plt.clf()
                    plt.cla()

                    for item in B['whiskers']:
                        quartis = item.get_ydata()
                        list_quartis.append(quartis)

                    arr = np.array(MeanError.copy())
                    l = arr[(arr > np.quantile(arr, 0.25)) & (arr < np.quantile(arr, 0.75))].tolist()
                    l_mean = np.mean(l)

                    # plt.plot(MSE)
                    # plt.show()

                    # if (MSE > 3):
                    #     origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
                    #     res_tm = dst_tm.copy()
                    #     Ry = rotation_matrix(180, yaxis)
                    #     res_tm.apply_transform(Ry)
                    #     H, MSE, Chamfer_dist, Hausdorff_dist, Mean_angle, scale, MeanError = icp(src_tm, res_tm, max_iterations=1000,
                    #                                                                   tolerance=1e-8)
                    #
                    # if (MSE > 3):
                    #     origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
                    #     res_tm = src_tm.copy()
                    #     Ry = rotation_matrix(180, yaxis)
                    #     res_tm.apply_transform(Ry)
                    #     H, MSE, Chamfer_dist, Hausdorff_dist, Mean_angle, scale, MeanError = icp(src_tm, res_tm, max_iterations=1000,
                    #                                                                   tolerance=1e-10)

                    # show the result by open3d
                    # dst_mesh_o3d1 = toO3d(dst_tm, color=(0.5, 0, 0))
                    # src_mesh_o3d2 = toO3d(src_tm, color=(0, 0, 0.5))
                    # res_mesh_o3d2 = toO3d(res_tm, color=(0, 0, 0.5))
                    # o3d.visualization.draw_geometries([dst_mesh_o3d1, src_mesh_o3d2])
                    # o3d.visualization.draw_geometries([dst_mesh_o3d1, res_mesh_o3d2])

                if (os.path.exists(save_file)):
                    pcd1 = o3d.io.read_point_cloud(save_file)
                    Y = np.asarray(pcd1.points)

                    hull = scipy.spatial.ConvexHull(Y)
                    density = Y.shape[0] / hull.volume

                    with open(txt_path_colmap) as f:
                        with open(txt_path_run) as f1:
                            for line in f.readlines():
                                worksheet.write(row, col, line)
                                col += 1
                            for line1 in f1.readlines():
                                worksheet.write(row, col, line1)
                                col += 1
                                worksheet.write(row, col, str(Chamfer_dist))
                                col += 1
                                worksheet.write(row, col, str(Hausdorff_dist))
                                col += 1
                                worksheet.write(row, col, str(density))
                                col += 1
                                worksheet.write(row, col, str(Mean_angle))
                                col += 1
                                worksheet.write(row, col, str(std_angle))
                                col += 1
                                worksheet.write(row, col, str(tmp))
                                col += 1
                                worksheet.write(row, col, str(std_min_error))
                                col += 1
                                worksheet.write(row, col, str(l_mean))
                                col += 1
                else:
                    col += 1
                    break
                row += 1

        if (file == 'model1_1_' or file == 'model5_1_' or file == 'model5_2_'):
            plt.boxplot((list_listT1[0], list_listT1[1], list_listT1[2], list_listT1[3]))
            plt.title("Box Plot")
            plt.ylabel('Mean Error')
            plt.xlabel('Number of imagens')
            plot_name1 = file + '5000.png'
            plt.savefig(plot_name1)
            plt.close()
            plt.clf()
            plt.cla()

            plt.boxplot((list_listT2[0], list_listT2[1], list_listT2[2], list_listT2[3]))
            plt.title("Box Plot")
            plt.ylabel('Mean Error')
            plt.xlabel('Number of imagens')
            plot_name2 = file + '7000.png'
            plt.savefig(plot_name2)
            plt.close()
            plt.clf()
            plt.cla()

            plt.boxplot((list_listT3[0], list_listT3[1], list_listT3[2], list_listT3[3]))
            plt.title("Box Plot")
            plt.ylabel('Mean Error')
            plt.xlabel('Number of imagens')
            plot_name3 = file + '9000.png'
            plt.savefig(plot_name3)
            plt.close()
            plt.clf()
            plt.cla()

workbook.close()

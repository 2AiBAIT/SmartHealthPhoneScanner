import math

import scipy
from numpy.linalg import LinAlgError
from open3d.cuda.pybind.utility import Vector3dVector
from pyvista import Polygon
from sklearn.neighbors import NearestNeighbors
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
import xlsxwriter
import open3d as o3d
import numpy as np
import trimesh
import os


dir_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/videos_exp2"
save = "/home/hcorreia/PycharmProjects/instant-ngp/data/Results2_align_1"
groud_truth_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp2"
excel_path = "/home/hcorreia/PycharmProjects/instant-ngp/data/Results_3.xlsx"

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


def nearest_neighbor(src, dst, number_neighbors):
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

    neigh = NearestNeighbors(n_neighbors=number_neighbors, algorithm='kd_tree')
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def int32torgb(color):
    rgb = []
    for i in range(1):
        rgb.append(color&0xff)
        color = color >> 8
    return rgb

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
    TList = []
    MeanErrorList = []
    StdErrorList = []
    ChamferErrorList = []
    HausdorffErrorList = []
    MeanAngleErrorList = []
    StdAngleErrorList = []

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T, 1)

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

        matched_src_normal = matched_src_pt_normals[reject_part_flag, :]
        matched_dst_normal = matched_dst_pt_normals[reject_part_flag, :]

        distances1, indices1 = nearest_neighbor(matched_src_pts, matched_dst_pts, 9)
        # compute angle between 2 matched vertexs' normals
        matched_src_pt_normals1 = matched_src_normal.copy()
        matched_dst_pt_normals1 = matched_dst_normal[indices1, :]
        anglesFinal = []
        listColorPoint = []
        soma = 0
        pcd = o3d.io.read_point_cloud("/home/hcorreia/PycharmProjects/instant-ngp/data/Results2_align_1/model4_1_2_5000Test_2.ply")
        ola = np.asarray(pcd.points)
        ola1 = np.asarray(pcd.colors)
        for l in range(matched_src_pt_normals1.shape[0]):
            conj9AngleList = []
            for n in range(9):
                v1 = matched_src_pt_normals1[l, :]
                v2 = matched_dst_pt_normals1[soma + n, :]
                cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                conj9Angle = np.arccos(cos_angle) / np.pi * 180
                conj9AngleList.append(conj9Angle)
            conj9AngleMean = np.mean(conj9AngleList)
            colorPoint = int32torgb(int(conj9AngleMean))
            anglesFinal.append(conj9AngleMean)
            listColorPoint.append(colorPoint)
            soma = soma + 9

        rgbs = [(1 - i, i, 0) for i in anglesFinal]
        pcd = o3d.geometry.PointCloud()
        pcd.points = Vector3dVector(matched_src_pts)
        pcd.colors = Vector3dVector(np.array(rgbs))

        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])

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

        MeanErrorList.append(round(mean_error, 3))
        StdErrorList.append(std_error)

        chamfer_dist = pcu.chamfer_distance(matched_dst_pts, matched_src_pts)
        hausdorff_dist = pcu.hausdorff_distance(matched_dst_pts, matched_src_pts)
        ChamferErrorList.append(chamfer_dist)
        HausdorffErrorList.append(hausdorff_dist)

        array_angles = np.array(anglesFinal)
        mean_angle = np.nanmean(array_angles)
        std_angle = np.nanmean(array_angles)
        MeanAngleErrorList.append(mean_angle)
        StdAngleErrorList.append(std_angle)

        if round(prev_error, 3) > round(mean_error, 3):
            number_min = 0
            prev_error = mean_error
        else:
            number_min = number_min + 1
            if number_min == 20:
                break

        # calculate final transformation
        T, _, _, scale= best_fit_transform(A, src[:m, :].T)
        TList.append(T)

    return TList, MeanErrorList, StdErrorList, ChamferErrorList, HausdorffErrorList, MeanAngleErrorList, StdAngleErrorList, scale, T


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


if __name__ == '__main__':

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):

            video_path = os.path.join(dir_path, path)
            list_n_frames = [2, 3, 4, 5]
            list_n_epochs = [5000, 7000, 9000]

            x = path.split(".")
            y = x[0].split('_')
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
                row = 30
            else:
                row = 50

            for i in range(len(list_n_frames)):

                # os.system(
                #     'python /home/hcorreia/PycharmProjects/instant-ngp/scripts/colmap2nerf.py --video_in ' + video_path + ' --video_fps ' + str(
                #         list_n_frames[i]))

                for k in range(len(list_n_epochs)):

                    col = 2
                    file = x[0] + '_' + str(list_n_frames[i]) + '_' + str(list_n_epochs[k]) + 'Test_2.ply'
                    file_export = '/home/hcorreia/PycharmProjects/instant-ngp/data/Results2_align_1/' + x[
                        0] + '_' + str(
                        list_n_frames[i]) + '_' + str(list_n_epochs[k]) + 'Test_2.ply'
                    save_file = os.path.join(save, file)
                    print(save_file)

                    if os.path.exists(save_file):

                        test1 = x[0] + '_' + str(list_n_frames[i])
                        test2 = x [0]

                        # if test1 == "model2_2_3":

                            # angle = 15
                            # T = [[math.cos(angle), 0, math.sin(angle), 0], [0, 1, 0, 0], [-math.sin(angle), 0, math.cos(angle), 0], [0, 0, 0, 1]]
                            # T =np.array(T)
                            #
                            # src_tm = trimesh.load(file_export)
                            # res_tm = src_tm.copy()
                            # res_tm.apply_transform(T)
                            # res_tm.export(file_export)
                        #
                        #
                        src_tm = trimesh.load(save_file)
                        dst_tm = trimesh.load(groud_truth)

#                         src_DataFile = '/home/hcorreia/PycharmProjects/instant-ngp/data/Results2_align/model1_2_2_9000.ply'
#                         dst_DataFile = '/home/hcorreia/PycharmProjects/instant-ngp/data/ground_exp2/model1.ply'
#
#                         dst_tm = trimesh.load(dst_DataFile)
#                         src_tm = trimesh.load(src_DataFile)
#
#                         src_tm1 = trimesh.load(src_DataFile)
#
                        points = np.asarray(src_tm.vertices)
                        numberPoints = len(points)
                        if numberPoints < 700000:

                            # ICP
                            # TList, MeanErrorList, StdErrorList, ChamferErrorList, HausdorffErrorList, MeanAngleErrorList, StdAngleErrorList, scale
                            HList, MSE, STD, ChamferList, HausdorffList, MSEAngleList, STDAngleList, scale, T = icp(src_tm, dst_tm, max_iterations=1)

                            MeanErrorMin = min(MSE)
                            index = MSE.index(MeanErrorMin)
                            print(MeanErrorMin)
                            print(scale)

                            StdErrorMin = round(STD[index], 3)
                            ChamferErrorMin = round(ChamferList[index], 3)
                            HausdorffErrorMin = round(HausdorffList[index], 3)
                            MeanAngleErrorMin = round(MSEAngleList[index], 3)
                            StdAngleErrorMin = round(STDAngleList[index], 3)

                            res_tm = src_tm.copy()
                            T = HList[index]
                            res_tm.apply_transform(T)
                            res_tm.apply_scale(scale)
                            res_tm.export('/home/hcorreia/PycharmProjects/instant-ngp/data/TESTAR.ply')

#                             src_DataFile = '/home/hcorreia/PycharmProjects/instant-ngp/data/TESTAR.ply'
#                             src_tm = trimesh.load(src_DataFile)
#                             HList, MSE1, STD, ChamferList, HausdorffList, MSEAngleList, STDAngleList, scale, T = icp(
#                                 src_tm, dst_tm, max_iterations=1)
#
#                             MeanErrorMin = min(MSE1)
#                             index = MSE1.index(MeanErrorMin)
#                             print(MeanErrorMin)
#                             print(scale)
#
#                             StdErrorMin = round(STD[index], 3)
#                             ChamferErrorMin = round(ChamferList[index], 3)
#                             HausdorffErrorMin = round(HausdorffList[index], 3)
#                             MeanAngleErrorMin = round(MSEAngleList[index], 3)
#                             StdAngleErrorMin = round(STDAngleList[index], 3)
#
#                             res_tm = src_tm.copy()
#                             T = HList[index]
#                             res_tm.apply_transform(T)
#                             res_tm.apply_scale(scale)
#                             res_tm.export('/home/hcorreia/PycharmProjects/instant-ngp/data/TESTAR2.ply')
#
#                             src_DataFile = '/home/hcorreia/PycharmProjects/instant-ngp/data/TESTAR2.ply'
#                             src_tm = trimesh.load(src_DataFile)
#                             HList, MSE2, STD, ChamferList, HausdorffList, MSEAngleList, STDAngleList, scale, T = icp(
#                                 src_tm, dst_tm, max_iterations=1)
#
#                             MeanErrorMin = min(MSE2)
#                             index = MSE2.index(MeanErrorMin)
#                             print(MeanErrorMin)
#                             print(scale)
#
#                             StdErrorMin = round(STD[index], 3)
#                             ChamferErrorMin = round(ChamferList[index], 3)
#                             HausdorffErrorMin = round(HausdorffList[index], 3)
#                             MeanAngleErrorMin = round(MSEAngleList[index], 3)
#                             StdAngleErrorMin = round(STDAngleList[index], 3)
#
#                             res_tm = src_tm.copy()
#                             T = HList[index]
#                             res_tm.apply_transform(T)
#                             res_tm.apply_scale(scale)
#                             res_tm.export('/home/hcorreia/PycharmProjects/instant-ngp/data/TESTAR3.ply')
#
#                             # src_DataFile = '/home/hcorreia/PycharmProjects/instant-ngp/data/TESTAR3.ply'
#                             # src_tm = trimesh.load(src_DataFile)
#                             # HList, MSE3, STD, ChamferList, HausdorffList, MSEAngleList, STDAngleList, scale, T = icp(
#                             #     src_tm, dst_tm, max_iterations=500)
# 							#
#                             # MeanErrorMin = min(MSE3)
#                             # index = MSE3.index(MeanErrorMin)
#                             # print(MeanErrorMin)
#                             # print(scale)
# 							#
#                             # StdErrorMin = round(STD[index], 3)
#                             # ChamferErrorMin = round(ChamferList[index], 3)
#                             # HausdorffErrorMin = round(HausdorffList[index], 3)
#                             # MeanAngleErrorMin = round(MSEAngleList[index], 3)
#                             # StdAngleErrorMin = round(STDAngleList[index], 3)
# 							#
#                             # res_tm = src_tm.copy()
#                             # T = HList[index]
#                             # res_tm.apply_transform(T)
#                             # res_tm.apply_scale(scale)
#                             # res_tm.export('/home/hcorreia/PycharmProjects/instant-ngp/data/TESTAR4.ply')
#
#                             MSE5 = MSE + MSE1[3:] + MSE2[8:]
#
#                             plt.plot(MSE5)
#                             plt.show()
#
#                             # show the result by open3d
#                             dst_mesh_o3d1 = toO3d(dst_tm, color=(0.5, 0, 0))
#                             src_mesh_o3d2 = toO3d(src_tm1, color=(0, 0, 0.5))
#                             res_mesh_o3d2 = toO3d(res_tm, color=(0, 0, 0.5))
#                             o3d.visualization.draw_geometries([dst_mesh_o3d1, src_mesh_o3d2])
#                             o3d.visualization.draw_geometries([dst_mesh_o3d1, res_mesh_o3d2])
#
#                             hull = scipy.spatial.ConvexHull(points)
#                             densityArea = points.shape[0] / hull.area
#                             densityVolume = points.shape[0] / hull.volume
#
#                             MeanSTDError = str(round(MeanErrorMin, 3)) + u"\u00B1" + str(StdErrorMin)
#                             MeanSTDAngleError = str(MeanAngleErrorMin) + u"\u00B1" + str(StdAngleErrorMin)
#                             print(MeanSTDAngleError)
#
#                             worksheet.write(row, col, str(MeanSTDError))
#                             col += 1
#                             worksheet.write(row, col, str(MeanSTDAngleError))
#                             col += 1
#                             worksheet.write(row, col, str(ChamferErrorMin))
#                             col += 1
#                             worksheet.write(row, col, str(HausdorffErrorMin))
#                             col += 1
#                             worksheet.write(row, col, str(round(densityArea, 3)))
#                             col += 1
#                             worksheet.write(row, col, str(round(densityArea)))
#                             col += 1
#                         else:
#                             break
#
#                         row += 1
# workbook.close()

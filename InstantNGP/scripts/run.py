#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import copy
import os
import commentjson as json

import numpy as np

import shutil
import time
from timeit import default_timer as timer

import pymeshlab

from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf

from tqdm import tqdm

import pyngp as ngp  # noqa
import open3d as o3d


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="/home/hcorreia/PycharmProjects/instant-ngp/data",
						help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="nerf", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"],
						help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--network", default="",
						help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="",
						help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="",
						help="Save this snapshot after training. recommended extension: .msgpack")

	parser.add_argument("--nerf_compatibility", action="store_true",
						help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="",
						help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--near_distance", default=-1, type=float,
						help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.5, type=float,
						help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="",
						help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="/home/hcorreia/PycharmProjects/instant-ngp/data/base_cam.json", help="The camera path to render.")
	parser.add_argument("--video_camera_smoothing", action="store_true",
						help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=18,
						help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_spp", type=int, default=15,
						help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video_out.mp4", help="Filename of the output video.")

	parser.add_argument("--save_mesh", default="/home/hcorreia/PycharmProjects/instant-ngp/data/model1_1+.ply",
						help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=400, type=int,
						help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0,
						help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0,
						help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", default=True, action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", default=True, action="store_true",
						help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", default=False, action="store_true",
						help="Open a second window containing a copy of the main output.")

	parser.add_argument("--sharpen", default=3, help="Set amount of sharpening applied to NeRF training images.")
	parser.add_argument("--epochs", default=9000, help="Set amount of sharpening applied to NeRF training images.")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	start = timer()

	if args.mode == "":
		if args.scene in scenes_sdf:
			args.mode = "sdf"
		elif args.scene in scenes_nerf:
			args.mode = "nerf"
		elif args.scene in scenes_image:
			args.mode = "image"
		elif args.scene in scenes_volume:
			args.mode = "volume"
		else:
			raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	if args.mode == "sdf":
		mode = ngp.TestbedMode.Sdf
		configs_dir = os.path.join(ROOT_DIR, "configs", "sdf")
		scenes = scenes_sdf
	elif args.mode == "nerf":
		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
	elif args.mode == "image":
		mode = ngp.TestbedMode.Image
		configs_dir = os.path.join(ROOT_DIR, "configs", "image")
		scenes = scenes_image
	elif args.mode == "volume":
		mode = ngp.TestbedMode.Volume
		configs_dir = os.path.join(ROOT_DIR, "configs", "volume")
		scenes = scenes_volume
	else:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	base_network = os.path.join(configs_dir, "base.json")
	if args.scene in scenes:
		network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
		base_network = os.path.join(configs_dir, network + ".json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure

	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene = args.scene
		if not os.path.exists(args.scene) and args.scene in scenes:
			scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
		testbed.load_training_data(scene)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window=args.second_window or False)

	if args.load_snapshot:
		print("Loading snapshot ", args.load_snapshot)
		testbed.load_snapshot(args.load_snapshot)
	else:
		testbed.reload_network_from_file(network)

	ref_transforms = {}
	if args.screenshot_transforms:  # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	testbed.shall_train = args.train if args.gui else True

	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]
	if args.mode == "sdf":\
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

	# Optionally match nerf paper behaviour and train on a
	# fixed white bg. We prefer training on random BG colors.
	# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
	# testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = int(args.epochs)

	tqdm_last_update = 0
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					if args.gui:
						testbed.shall_train = False
					else:
						break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

				if old_training_step == n_steps:
					break

	res = args.marching_cubes_res or 256
	print(
		f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
	testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res])

	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(args.save_mesh)
	p = pymeshlab.Percentage(50)
	ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=p)
	m = ms.current_mesh()
	#vert = int(m.vertex_number()/1.5)
	#ms.meshing_decimation_quadric_edge_collapse(targetfacenum=vert, preserveboundary=True, preservenormal=True,
	#											preservetopology=True, qualitythr=0.5)
	ms.apply_coord_taubin_smoothing(lambda_=1, mu=0, stepsmoothnum=20)
	## ms.apply_coord_taubin_smoothing(lambda_=1, mu=0, stepsmoothnum=20)
	# ms.compute_matrix_from_scaling_or_normalization(axisx=57)
	ms.save_current_mesh(args.save_mesh, binary=False)
	# m = ms.current_mesh()
	# end = timer()
	# print(end - start)
	# print(m.vertex_number())



	# pcd = o3d.io.read_point_cloud(args.save_mesh)
	# pcd_rad, ind_rad = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
	# outlier_rad_pcd = pcd.select_by_index(ind_rad, invert=True)
	# outlier_rad_pcd.paint_uniform_color([1., 0., 1.])

	# # Statistical outlier removal:
	# pcd_stat, ind_stat = pcd.remove_statistical_outlier(nb_neighbors=16,
	# 													std_ratio=2.0)
	# outlier_stat_pcd = pcd.select_by_index(ind_stat, invert=True)
	# outlier_stat_pcd.paint_uniform_color([0., 0., 1.])
	#
	# # Translate to visualize:
	# points = np.asarray(pcd_stat.points)
	# points += [3, 0, 0]
	# pcd_stat.points = o3d.utility.Vector3dVector(points)
	#
	# points = np.asarray(outlier_stat_pcd.points)
	# points += [3, 0, 0]
	# outlier_stat_pcd.points = o3d.utility.Vector3dVector(points)
	#
	# # Display:
	# o3d.visualization.draw_geometries([pcd_stat, pcd_rad, outlier_stat_pcd, outlier_rad_pcd])

	# Params class containing the counter and the main LineSet object holding all line subsets
	# class params():
	#
	# 	counter = 0
	# 	full_line_set = o3d.geometry.LineSet()
	#
	#
	# # Callback function for generating the LineSets from each neighborhood
	# def build_edges(vis):
	# 	# Run this part for each point in the point cloud
	# 	if params.counter < len(points):
	# 		# Find the K-nearest neighbors for the current point. In our case we use 6
	# 		[k, idx, _] = pcd_tree.search_knn_vector_3d(points[params.counter, :], 6)
	# 		# Get the neighbor points from the indices
	# 		points_temp = points[idx, :]
	#
	# 		# Create the neighbours indices for the edge array
	# 		neighbours_num = np.arange(len(points_temp))
	# 		# Create a temp array for the center point indices
	# 		point_temp_num = np.zeros(len(points_temp))
	# 		# Create the edges array as a stack from the current point index array and the neighbor indices array
	# 		edges = np.vstack((point_temp_num, neighbours_num)).T
	#
	# 		# Create a LineSet object and give it the points as nodes together with the edges
	# 		line_set = o3d.geometry.LineSet()
	# 		line_set.points = o3d.utility.Vector3dVector(points_temp)
	# 		line_set.lines = o3d.utility.Vector2iVector(edges)
	# 		# Color the lines by either using red color for easier visualization or with the colors from the point cloud
	# 		line_set.paint_uniform_color([1, 0, 0])
	# 		# line_set.paint_uniform_color(colors[params.counter,:])
	#
	# 		# Add the current LineSet to the main LineSet
	# 		params.full_line_set += line_set
	#
	# 		# if the counter just started add the LineSet geometry
	# 		if params.counter == 0:
	# 			vis.add_geometry(params.full_line_set)
	# 		# else update the geometry
	# 		else:
	# 			vis.update_geometry(params.full_line_set)
	# 		# update the render and counter
	# 		vis.update_renderer()
	# 		params.counter += 1
	# 	else:
	# 		# if the all point have been used reset the counter and clear the lines
	# 		params.counter = 0
	# 		params.full_line_set.clear()

	# pcd = o3d.io.read_point_cloud(args.save_mesh)
	# pcd_rad, ind_rad = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
	# outlier_rad_pcd = pcd.select_by_index(ind_rad, invert=True)
	# outlier_rad_pcd.paint_uniform_color([1., 0., 1.])
	#
	# # Statistical outlier removal:
	# pcd_stat, ind_stat = pcd.remove_statistical_outlier(nb_neighbors=20,
	# 													std_ratio=2.0)
	# outlier_stat_pcd = pcd.select_by_index(ind_stat, invert=True)
	# outlier_stat_pcd.paint_uniform_color([0., 0., 1.])
	#
	# # Translate to visualize:
	# points = np.asarray(pcd_stat.points)
	# points += [3, 0, 0]
	# pcd_stat.points = o3d.utility.Vector3dVector(points)
	#
	# points = np.asarray(outlier_stat_pcd.points)
	# points += [3, 0, 0]
	# outlier_stat_pcd.points = o3d.utility.Vector3dVector(points)
	#
	# # Display:
	# o3d.visualization.draw_geometries([pcd_stat, pcd_rad, outlier_stat_pcd, outlier_rad_pcd])
	# o3d.io.write_point_cloud("test.ply",  outlier_stat_pcd)


	# file_object = open('/home/hcorreia/PycharmProjects/instant-ngp/data/run.txt', 'w')
	# file_object.write(str(end - start))
	# file_object.write("\n")
	# file_object.write(str(m.vertex_number()))
	# file_object.write("\n")
	# file_object.write(str(testbed.loss))
	# file_object.write("\n")
	# file_object.close()
	testbed.destroy_window()

	# pcd = o3d.io.read_point_cloud(args.save_mesh)
	#
	# o3d.visualization.draw_geometries([pcd],
	# 								  zoom=0.664,
	# 								  front=[-0.4761, -0.4698, -0.7434],
	# 								  lookat=[1.8900, 3.2596, 0.9284],
	# 								  up=[0.2304, -0.8825, 0.4101])
	# testbed.destroy_window()

	if args.save_snapshot:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(args.save_snapshot, False)

	if args.test_transforms:
		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir = os.path.dirname(args.test_transforms)
		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.rendering_min_transmittance = 1e-4

		testbed.fov_axis = 0
		testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
		testbed.shall_train = False

		with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
			for i, frame in t:
				p = frame["file_path"]
				if "." not in p:
					p = p + ".png"
				ref_fname = os.path.join(data_dir, p)
				if not os.path.isfile(ref_fname):
					ref_fname = os.path.join(data_dir, p + ".png")
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".jpg")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".jpeg")
							if not os.path.isfile(ref_fname):
								ref_fname = os.path.join(data_dir, p + ".exr")

				ref_image = read_image(ref_fname)

				# NeRF blends with background colors in sRGB space, rather than first
				# transforming to linear space, blending there, and then converting back.
				# (See e.g. the PNG spec for more information on how the `alpha` channel
				# is always a linear quantity.)
				# The following lines of code reproduce NeRF's behavior (if enabled in
				# testbed) in order to make the numbers comparable.
				if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
					# Since sRGB conversion is non-linear, alpha must be factored out of it
					ref_image[..., :3] = np.divide(ref_image[..., :3], ref_image[..., 3:4],
												   out=np.zeros_like(ref_image[..., :3]),
												   where=ref_image[..., 3:4] != 0)
					ref_image[..., :3] = linear_to_srgb(ref_image[..., :3])
					ref_image[..., :3] *= ref_image[..., 3:4]
					ref_image += (1.0 - ref_image[..., 3:4]) * testbed.background_color
					ref_image[..., :3] = srgb_to_linear(ref_image[..., :3])

				if i == 0:
					write_image("ref.png", ref_image)

				testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1, :])
				image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

				if i == 0:
					write_image("out.png", image)

				diffimg = np.absolute(image - ref_image)
				diffimg[..., 3:4] = 1.0
				if i == 0:
					write_image("diff.png", diffimg)

				A = np.clip(linear_to_srgb(image[..., :3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[..., :3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				totssim += ssim
				totmse += mse
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr < minpsnr else minpsnr
				maxpsnr = psnr if psnr > maxpsnr else maxpsnr
				totcount = totcount + 1
				t.set_postfix(psnr=totpsnr / (totcount or 1))

		psnr_avgmse = mse2psnr(totmse / (totcount or 1))
		psnr = totpsnr / (totcount or 1)
		ssim = totssim / (totcount or 1)
		print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

	if ref_transforms:
		testbed.fov_axis = 0
		testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
		if not args.screenshot_frames:
			args.screenshot_frames = range(len(ref_transforms["frames"]))
		print(args.screenshot_frames)
		for idx in args.screenshot_frames:
			f = ref_transforms["frames"][int(idx)]
			cam_matrix = f["transform_matrix"]
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
			outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".png"

			print(f"rendering {outname}")
			image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]),
								   args.screenshot_spp, True)
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)
	elif args.screenshot_dir:
		outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
		print(f"Rendering {outname}.png")
		image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
		if os.path.dirname(outname) != "":
			os.makedirs(os.path.dirname(outname), exist_ok=True)
		write_image(outname + ".png", image)

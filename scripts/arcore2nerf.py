import argparse
import numpy as np
import json
import sys
import math
import os
import shutil
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")
    parser.add_argument("--video_in", default="",
                        help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
    parser.add_argument("--video_fps", default=2)
    parser.add_argument("--arcore_nerf_dir", default="C:/Users/Helena/Documents/yolov5",
                        help="input path to the images")
    parser.add_argument("--aabb_scale", default=16, choices=["1", "2", "4", "8", "16"],
                        help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--keep_colmap_coords", action="store_true",
                        help="keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering)")
    parser.add_argument("--out", default="transforms.json", help="output path")
    args = parser.parse_args()
    return args


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def run_ffmpeg(args):
    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)
    images = args.images
    video = args.video_in
    fps = float(args.video_fps) or 1.0
    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip() + "y")[
       :1] != "y":
        sys.exit(1)
    try:
        shutil.rmtree(images)
    except:
        pass
    do_system(f"mkdir {images}")
    do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {images}/%04d.jpg")


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


if __name__ == "__main__":
    args = parse_args()
    if args.video_in != "":
        run_ffmpeg(args)
    AABB_SCALE = int(args.aabb_scale)
    images_dir = os.path.join(args.arcore_nerf_dir, "ImagePath")
    pose_dir = os.path.join(args.arcore_nerf_dir, "PosePath")
    camera_intrinsic_path = os.path.join(args.arcore_nerf_dir, "cam_intrinsics.txt")
    OUT_PATH = args.out
    print(f"outputting to {OUT_PATH}...")
    with open(camera_intrinsic_path, encoding='utf-8-sig', errors='ignore') as f:
        for line in f:

            if line[0] == "#":
                continue

            # cam_intrinsics = line.readlines()
            cam_intrinsics_parameter = line.split(" ")

            w = float(cam_intrinsics_parameter[2])
            h = float(cam_intrinsics_parameter[5])  # resolution
            fl_x = float(cam_intrinsics_parameter[0])  # focalLength
            fl_y = float(cam_intrinsics_parameter[3])  # focalLength
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = float(cam_intrinsics_parameter[1])  # principalPoint
            cy = float(cam_intrinsics_parameter[4])  # principalPoint

            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(
        f"camera:\n\tres={w, h}\n\tcenter={cx, cy}\n\tfocal={fl_x, fl_y}\n\tfov={fovx, fovy}\n\tk={k1, k2} p={p1, p2} ")

    i = 0
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }
    up = np.zeros(3)
    img_names = os.listdir(images_dir)
    for img_name in img_names:
        img_path = os.path.join(images_dir, img_name)
        pose_path = os.path.join(pose_dir, img_name.split('.')[0] + '.txt')

        if not os.path.exists(pose_path):
            continue

        with open(pose_path, encoding='utf-8-sig', errors='ignore') as f:
            j_pose = f.readlines()
            j_pose_parameter = j_pose[0].split()

            b = sharpness(img_path)
            qvec = np.array(tuple(map(float, [j_pose_parameter[6], j_pose_parameter[0], j_pose_parameter[2],
                                              j_pose_parameter[4]])))  # rotation component of this pose
            tvec = np.array(tuple(map(float, [j_pose_parameter[1], j_pose_parameter[3],
                                              j_pose_parameter[5]])))  # translation component of this pose

            R = qvec2rotmat(-qvec)
            t = tvec.reshape([3, 1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

            c2w = m
            c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
            c2w[2, :] *= -1  # flip whole world upside down

            up += c2w[0:3, 1]

            name = 'data/' + img_name
            frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
            out["frames"].append(frame)

    nframes = len(out["frames"])

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.00001:
                totp += p * w
                totw += w
    if totw > 0.0:
        totp /= totw
    print(totp)  # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"
        f["transform_matrix"] = f["transform_matrix"].tolist()

    print(nframes, "frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

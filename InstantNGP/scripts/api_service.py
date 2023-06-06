from fastapi import FastAPI, status, File, UploadFile, Form, responses, Response, Query
import uvicorn
import cv2
from typing import List, Union
import argparse
import commentjson as json
import shutil
import time
from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf
from tqdm import tqdm
import pyngp as ngp  # noqa
import sys
import open3d as o3d
import copy
import run
import os.path as op
import zipfile
import io

# run code -> uvicorn test_api:app --reload

app = FastAPI()


def reproduce_video(name):
	# create a videoCapture Object (this allow to read frames one by one)
	video = cv2.VideoCapture(name)
	# check it's ok
	if video.isOpened():
		print('Video Succefully opened')
	else:
		print('Something went wrong check if the video name and path is correct')

	# define a scale lvl for visualization
	scaleLevel = 3  # it means reduce the size to 2**(scaleLevel-1)

	windowName = 'Video Reproducer'
	cv2.namedWindow(windowName)
	# let's reproduce the video
	while True:
		ret, frame = video.read()  # read a single frame
		if not ret:  # this mean it could not read the frame
			print("Could not read the frame")
			cv2.destroyWindow(windowName)
			break

		reescaled_frame = frame
		for i in range(scaleLevel - 1):
			reescaled_frame = cv2.pyrDown(reescaled_frame)

		cv2.imshow(windowName, reescaled_frame)

		waitKey = (cv2.waitKey(1) & 0xFF)
		if waitKey == ord('q'):  # if Q pressed you could do something else with other keypress
			print("closing video and exiting")
			cv2.destroyWindow(windowName)
			video.release()
			break


def zip_files(filenames):
	zip_filename = "archive.zip"
	s = io.BytesIO()
	zf = zipfile.ZipFile(s, "w")
	for fpath in filenames:
		# Calculate path for file in zip
		fdir, fname = os.path.split(fpath)
		# Add file, at correct path
		zf.write(fpath, fname)
	# Must close zip for all contents to be written
	zf.close()
	# Grab ZIP file from in-memory, make response with correct MIME-type
	resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
		'Content-Disposition': f'attachment;filename={zip_filename}'
	})

	return resp


@app.post("/video/", status_code=status.HTTP_201_CREATED)
def upload_singleFile(file: UploadFile = File(...)):
	print(file.filename)
	# print(file.read())

	# temp = NamedTemporaryFile(delete=False)
	# try:
	#     try:
	#         contents = file.file.read()
	#         with temp as f:
	#             f.write(contents);
	#     except Exception:
	#         return {"message": "There was an error uploading the file"}
	#     finally:
	#         file.file.close()
	#
	#     reproduce_video(temp.name)
	#
	# except Exception:
	#     return {"message": "There was an error processing the file"}
	# finally:
	#     # temp.close()  # the `with` statement above takes care of closing the file
	#     os.remove(temp.name)

	file_location = f"files/{file.filename}"
	with open(file_location, "wb+") as file_object:
		file_object.write(file.file.read())

	return responses.FileResponse("/data/test.obj", media_type="model/3mf",
								  status_code=status.HTTP_201_CREATED, filename="face1.obj")


@app.post('/home/hcorreia/PycharmProjects/instant-ngp/Api/files/', status_code=status.HTTP_201_CREATED)
def upload_files(files: List[UploadFile] = File(), username: str = Form()):
	username = username.replace('"','')
	print(username)

	# path to save file in the username folder
	#path = op.join(os.getcwd(), 'Database', username)
	path = op.join('/home/hcorreia/PycharmProjects/instant-ngp/Api/', 'Database', username)

	# check if the user folder already exists in database
	if not os.path.exists(path):
		os.mkdir(path)

	for file in files:
		print(file.filename)
		file_location = f"/home/hcorreia/PycharmProjects/instant-ngp/Api/files/{file.filename}"
		with open(file_location, "wb+") as file_object:
			file_object.write(file.file.read())

	# os.system('python /home/hcorreia/PycharmProjects/instant-ngp/scripts/arcore_run.py')
	os.system('python /home/hcorreia/PycharmProjects/instant-ngp/scripts/colmap2nerfAPI.py')
	os.system('python /home/hcorreia/PycharmProjects/instant-ngp/scripts/runAPI.py --username ' + username)

	path = op.join("/home/hcorreia/PycharmProjects/instant-ngp/Api/Database/", username)
	filenames = [name for name in os.listdir(path)]
	filenames.sort()
	path1 = op.join(path, str(filenames[-1]))

	return responses.FileResponse(path1, media_type="model/3mf",
		status_code=status.HTTP_201_CREATED, filename="test.obj")


@app.get('/home/hcorreia/PycharmProjects/instant-ngp/Api/Database/{username}/{filename}/')
def download_history_file(username: str, filename: str):
	# get the path in database of the current user
	print(filename)
	#path = op.join(os.getcwd(), 'Database', username, filename)
	path = op.join('/home/hcorreia/PycharmProjects/instant-ngp/Api/', 'Database', username, filename)

	# check if the user have folder history in database
	return responses.FileResponse(
		path,
		media_type="model/3mf",
		status_code=status.HTTP_202_ACCEPTED, filename="PointCloud.ply")


@app.get('/home/hcorreia/PycharmProjects/instant-ngp/Api/Database/{username}/')
def download_history_filenames(username: str):
	# get the path in database of the current user
	#path = op.join(os.getcwd(), 'Database', username)
	path = op.join('/home/hcorreia/PycharmProjects/instant-ngp/Api/', 'Database', username)

	# check if the user have folder history in database
	if op.exists(path) and len(os.listdir(path)) > 0:
		filenames = [name for name in os.listdir(path)]
		filenames.sort()
		print(filenames)
		return filenames

	else:
		return Response(status_code=status.HTTP_204_NO_CONTENT)


# img = ['/home/andre/Desktop/Bolsa/Values_Working/API/data.csv',
# '/home/andre/Desktop/Bolsa/Values_Working/API/uuid_services.txt']
#   return zipfiles(img)


if __name__ == '__main__':
	# app.run(port=7777)
	uvicorn.run(app, host="127.0.0.1", port=8000)

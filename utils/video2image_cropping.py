# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:55:19 2021

@author: L
"""
import os
import cv2
import random
import glob
import dlib
import argparse
import json
from tqdm import tqdm

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap', 
    'NeuralTextures':'manipulated_sequences/NeuralTextures'
}

frame_per_video = 100

# Dlib face detector
detector = dlib.get_frontal_face_detector()

# Dlib landmarks detector model
predictor = dlib.shape_predictor('./utils/shape_predictor_68_face_landmarks.dat')


def DFDC_label_processing(label_file):
    video_label = {}
    with open(label_file) as f:
        content = json.load(f)
    for k,v in  content.items():
        video_label.update({k : v["label"]})
    return video_label

def deleteMinus(value):
	if value < 0:
		return 0
	else:
		return value

def cropSingleFace(image, outputPath, frame, scale=1.3):#
	faces = detector(image, 1)
	face_count = 0
	for num, face in enumerate(faces):
		face_count += 1
		top, bottom, left, right = deleteMinus(face.top()), deleteMinus(face.bottom()), deleteMinus(face.left()), deleteMinus(face.right())
		vCenter, hCenter = (top + bottom)//2, (left+right)//2
		size_bb = int(max(bottom-top, right-left) * scale)	
		top = max(int(vCenter - size_bb // 2), 0)
		bottom = min(int(vCenter + size_bb // 2), image.shape[0])
		left = max(int(hCenter - size_bb // 2), 0)
		right = min(int(hCenter + size_bb // 2), image.shape[1])
		cropped = image[top:bottom, left:right]#cropping
		cv2.imwrite(os.path.join(outputPath, str(frame).zfill(4)+"_"+str(face_count)+'.png'), cropped)
	del faces
	if face_count > 1:
		return 1
	else:
		return 0
		
def frameExtraction(videoPath, face_path, rand=False, fpv=-1):
	"""
	function: ectract frames from a video
	fpv: frames per videoo
	"""
	filename = videoPath.split("/")[-1].split(".")[0]
	cap = cv2.VideoCapture(str(videoPath)) 
	frames_num=cap.get(7)
	frameList = []
    
	if fpv == -1:
		frameList = range(int(frames_num))
	else:
		if rand:
			if not fpv <  frames_num:
				raise ValueError ("Expected frame more than video length")
			else:
				frameList = random.sample(range(0,int(frames_num)), fpv)
		else:
			frameList = range(int(frames_num))[:fpv]

        
	for frameNum in frameList:			
		cap.set(cv2.CAP_PROP_POS_FRAMES, float(frameNum))
		if cap.isOpened(): 
			rval , frame = cap.read() 
			if not os.path.exists(face_path):
				os.mkdir(face_path)
			cropSingleFace(frame, face_path, frameNum)
	cap.release()


def extract_method_videos(data_path, subset, compression):
    #Extraction for FF++ dataset
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = os.path.join(data_path, DATASET_PATHS[subset], compression, 'videos')
    images_path = os.path.join(data_path, DATASET_PATHS[subset], compression, 'images')
    video_list = os.listdir(videos_path)
    video_list.sort()
    for video in tqdm(video_list):
        image_folder = video.split('.')[0]
        frameExtraction(os.path.join(videos_path, video),
                       os.path.join(images_path, image_folder))

	
if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--dataset', '-ds', type=str, \
                    choices=['ff++', 'dfd', 'deeper', 'celebdf'], default="ff++")
	parse.add_argument('--data_path', '-s', type=str)
	parse.add_argument('--compression', '-c', type=str, default="raw")
	args = parse.parse_args()
	if args.dataset == "ff++":
		for ss in DATASET_PATHS.keys():
			args.subset = ss
			extract_method_videos(args.data_path, args.subset , args.compression)
	elif args.dataset == "deeper":
		videos_path = os.path.join(args.data_path, "end_to_end")
		images_path = os.path.join(args.data_path, "end_to_end_images")
		if not os.path.exists(images_path):
			os.mkdir(images_path)
		video_list = os.listdir(videos_path)
		video_list.sort()
		for video in video_list:
			image_folder = video.split('.')[0]
			frameExtraction(os.path.join(videos_path, video),
                           os.path.join(images_path, image_folder))
	elif args.dataset == "celebdf":
		real_videos_path = os.path.join(args.data_path, "Celeb-real")
		fake_videos_path = os.path.join(args.data_path, "Celeb-synthesis")
		real_images_path = os.path.join(args.data_path, "Celeb-real_images")
		fake_images_path = os.path.join(args.data_path, "Celeb-synthesis_images")
		video_path_list = [real_videos_path, fake_videos_path]
		image_path_list = [real_images_path, fake_images_path]
		for idx, i in enumerate(video_path_list):
			if not os.path.exists(image_path_list[idx]):
 				os.mkdir(image_path_list[idx])
			video_list = os.listdir(i)
			video_list.sort()
			for video in video_list:
 				image_folder = video.split('.')[0]
 				frameExtraction(os.path.join(i, video),
                            os.path.join(image_path_list[idx], image_folder))
	elif args.dataset == "dfd":
		real_videos_path = os.path.join(args.data_path, "actors", "raw", "videos")
		fake_videos_path = os.path.join(args.data_path, "DeepFakeDetection", "raw", "videos")
		real_images_path = os.path.join(args.data_path, "actors", "raw", "images")
		fake_images_path = os.path.join(args.data_path, "DeepFakeDetection", "raw", "images")
		video_path_list = [real_videos_path, fake_videos_path]
		image_path_list = [real_images_path, fake_images_path]
		for idx, i in enumerate(video_path_list):
			if not os.path.exists(image_path_list[idx]):
 				os.mkdir(image_path_list[idx])
			video_list = os.listdir(i)
			video_list.sort()
			for video in video_list:
 				image_folder = video.split('.')[0]
 				frameExtraction(os.path.join(i, video),
                            os.path.join(image_path_list[idx], image_folder))	

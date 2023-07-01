#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:57:14 2023

@author: baoping
"""
import cv2

import numpy as np
import torch

cropped_size = (128,128)

def load_image(img_path, compress=None, noise=None):
    image = cv2.imread(img_path, 0)
    image = cv2.resize(image, cropped_size)
    if not compress is None:
        bte = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, compress])[1]
        string = bte.tobytes()
        image = cv2.imdecode(np.frombuffer(string, np.byte), -1)
        del bte,string
    if not noise is None:
        gauss = np.random.normal(0, noise, (image.shape))
        image = image +  gauss
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def get_frame_id(model, frames, batch_size = 128):
    batch = frames
    data = torch.from_numpy(batch)
    data = data.to(torch.device("cuda"))
    output = model(data)#data
    #print("output", output.shape)
    output = output.data.cpu().numpy()
    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))
    #print("feature", feature.shape)
    return feature



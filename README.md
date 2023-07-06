# TI2Net
This is the official implementation of our WACV2023 paper: [TI2Net: Temporal Identity Inconsistency Network for Deepfake Detection](https://openaccess.thecvf.com/content/WACV2023/html/Liu_TI2Net_Temporal_Identity_Inconsistency_Network_for_Deepfake_Detection_WACV_2023_paper.html)

We referred much to the source code of [arcface](https://github.com/ronghuaiyang/arcface-pytorch/tree/master) and [LRNet](https://github.com/frederickszk/LRNet)


# Overview


# Data pre-processing
* Frame extraction and face cropping
  * Frame extraction script is mainly based on the script provided by [FaceForensics++](https://github.com/ondyari/FaceForensics). We combine the script with Dlib to achieve frame extraction and cropping at once. You can find the processing in the [data_preprocessing.ipynb](data_preprocessing.ipynb), which contains the processing of various datasets in our experiments. The required pre-trained landmark detector model can be found in [Google Drive](https://drive.google.com/file/d/1zvKD-66Ye_g6qn9LvBuTSrT8ND-MU1aD/view?usp=drive_link) or [BaiduCloud (coming soon)](), download it and place it in the **utils** folder

* **Bad examples** in the FF++ dataset:
   While detecting human faces from videos, we found some videos that could not be smoothly processed by the face detector and our coding; we list them here so that other researchers can avoid the  noise  from these bad examples in their research works. 
  * Video_281: The target face is not the largest detected face in video frames. Manually select the target face from  detected faces.
  * Video_344: There are two equally large faces, so we need to manually select target faces frame-by-frame.
  * Video_356：From the 3rd second on, a long part of the video contains no human face.
  * Video_370: From the 7th-9th second, a camera switching part contains no human face.
  * Video_492: There are Two equally large faces, so we need to manually select target faces frame-by-frame.
  * Video_509: There is a long part of camera switching, and some frames do not contain human faces
  * Video_738: At around the 10th second, there is a camera switching, and there are multiple equally large faces, so we need to manually select target faces frame-by-frame.
  * Video_908: There is a camera switching at about the 13th second.
  * Video_950: Camera defocused at about 23rd second and human faces cannot be detected here.

# Model training and evaluation



* **Training**:  
  For hardware limitations, our experiments are conducted in a two-stage manner, which includes: 1). Off-line identity extraction  and saving.  2). Identity loading and training. Both of these steps are accomplished in [demo_offline_id.ipynb](demo_offline_id.ipynb)
  * 1). Off-line identity extraction and saving: Identity vectors are extracted with [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html). Download the pre-trained model serving as the backbone of ArcFace via [GoogleDrive](https://drive.google.com/file/d/1NkO1xmJp-mBpDVMXlwEgrvJwX0hXjYqM/view?usp=drive_link) or [BaiduCloud (coming soon)](). Then run the commands in the first part of demo_offline_id.ipynb to extract identity vectors and save them as hard disk files. For each dataset, identity vectors in all frames of all videos are organized in a NumPy array and saved as .npy files. Corresponding to each identity file is a length file comprising the length of each video so that we can recover identity vectors video by video. You may customize the extraction process according to your own need in [video2image_cropping.py](utils/video2image_cropping.py)
  ** We provide the extracted id vectors of the FF++ dataset, you can download it through [GoogleDrive]([https://drive.google.com/drive/folders/1uEBSBGyjZC4DiseYiCRo4ULSauvb7wyw?usp=drive_link](https://drive.google.com/drive/folders/1uEBSBGyjZC4DiseYiCRo4ULSauvb7wyw?usp=drive_link))
  * 2). Identity loading and training: Commands can be found in the second part of [demo_offline_id.ipynb](demo_offline_id.ipynb).
     
 
 * **Evaluation**:
   Commands can be found in the third part of demo_offline_id.ipynb.



# To-Do list
* - [x] Table of bad examples in datasets.
* - [ ] Release extracted ID vectors.
* - [ ] Release pre-trained model.
  - [ ] Update BaiduPan links for pre-trained models

# Citaton
@inproceedings{liu2023ti2net,  
     title={TI2Net: Temporal Identity Inconsistency Network for Deepfake Detection}, <br>
     author={Liu, Baoping and Liu, Bo and Ding, Ming and Zhu, Tianqing and Yu, Xin}, <br>
     booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},<br>
     pages={4691--4700},<br>
     year={2023}<br>
}

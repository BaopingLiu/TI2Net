# TI2Net
This is the official implementation of our WACV2023 paper: [TI2Net: Temporal Identity Inconsistency Network for Deepfake Detection](https://openaccess.thecvf.com/content/WACV2023/html/Liu_TI2Net_Temporal_Identity_Inconsistency_Network_for_Deepfake_Detection_WACV_2023_paper.html)

We referred much to the source code of [arcface](https://github.com/ronghuaiyang/arcface-pytorch/tree/master) and [LRNet](https://github.com/frederickszk/LRNet)


# Overview


# Data pre-processing
* Frame extraction and face cropping
  * Frame extraction script is mainly based on the script provided by [FaceForensics++](https://github.com/ondyari/FaceForensics). We combine the script with Dlib to achieve frame extraction and cropping at once. You can find the processing in the [data_preprocessing.ipynb](data_preprocessing.ipynb), which contains the processing of various datasets in our experiments.
  * Note that there are also some manual works to improve the quality of the extracted human faces.


   


# Model training and evaluation

* Evaluation: 

* Training:  
  For hardware limitations, our experiments are conducted in a two-stage manner, which includes: 1). Off-line identity extraction  and saving.  2). Identity loading and training. Both of these steps are accomplished in [demo_offline_id.ipynb](demo_offline_id.ipynb)
  * 1). Off-line identity extraction and saving: Identity vectors are extracted with [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html) Download the pre-trained model serving as the backbone of [ArcFace](https://drive.google.com/file/d/1NkO1xmJp-mBpDVMXlwEgrvJwX0hXjYqM/view?usp=sharing). Then run the commands in the first part of demo_offline_id.ipynb to extract identity vectors and save them as hard disk files. For each dataset, identity vectors in all frames of all videos are organized in a NumPy array and saved as .npy files. Corresponding to each identity file is a length file comprising the length of each video so that we can recover identity vectors video by video. You may customize the extraction process according to your own need in [video2image_cropping.py](utils/video2image_cropping.py)
  * 2). Identity loading and training: 



# To-Do list
* Table of bad examples in datasets.
* Release extracted ID vectors
* Release pre-trained model


# Citaton
@inproceedings{liu2023ti2net,
  title={TI2Net: Temporal Identity Inconsistency Network for Deepfake Detection}, <br>
  author={Liu, Baoping and Liu, Bo and Ding, Ming and Zhu, Tianqing and Yu, Xin}, <br>
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},<br>
  pages={4691--4700},<br>
  year={2023}<br>
}

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:15:37 2022

@author: Baoping Liu
"""
import argparse
from os.path import join
from models.ti2net import TI2Net
from utils.logger import Logger
from utils.metric import evaluate
from utils.dataset import Pair_Dataset
from train import id_len_to_set 
import torch.utils.data as Data
import torch

def main(model_path, real_ids_path, real_len_path, fake_ids_path, fake_len_path):
    BATCH_SIZE = 64
    DROPOUT_RATE = 0.5
    RNN_UNIT = 512
    add_weights = './weights/torch/'
    real_set = id_len_to_set(real_ids_path, real_len_path)
    fake_set = id_len_to_set(fake_ids_path, fake_len_path)
    
    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = 'cpu'

    logger = Logger()
    logger.print_logs_evaluating()

    test_dataset  = Pair_Dataset(real_set,  fake_set,  phase="test")
    test_iter_A = Data.DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    model = TI2Net(feature_dim=1024, rnn_unit=RNN_UNIT, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(join(add_weights, model_path)))
    
    test_acc, test_auc, test_video_acc, test_video_auc, embeddings = evaluate(model, test_iter_A, device, video_strategy="avg")
    
    print("\n")
    print("#----Evaluation Results----#")
    print("Accuracy (sequence-level): {:.4}".format(test_acc))
    print("AUC (sequence-level): {:.4}".format(test_auc))
    print("Accuracy (video-level): {:.4}".format(test_video_acc))
    print("AUC (video-level): {:.4}".format(test_video_auc))
    print("#------------End-----------#")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating codes of LRNet (PyTorch version).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '--model', type=str, default=None)
    parser.add_argument('--real_id_path', '--rip', type=str, default=None)
    parser.add_argument('--real_len_path', '--rlp', type=str,  default=None)
    parser.add_argument('--fake_id_path', '--fip', type=str,  default=None)
    parser.add_argument('--fake_len_path', '--flp', type=str,  default=None)
    args = parser.parse_args()
    real_id_path = args.real_id_path
    real_len_path = args.real_len_path
    fake_id_path = args.fake_id_path
    fake_len_path = args.fake_len_path
    model_path = args.model
    main(model_path, real_id_path, real_len_path, fake_id_path, fake_len_path)
    
        

    

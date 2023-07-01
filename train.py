# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:15:37 2022

@author: Baoping Liu
"""
import numpy as np
import os
import tensorflow as tf
import argparse
import torch
from torch import nn
import torch.utils.data as Data
from tqdm import tqdm, trange
from torch import optim
from os.path import join
from models.ti2net import TI2Net
from utils.metric import evaluate, calculate_accuracy
from utils.dataset import Pair_Dataset
from sklearn.model_selection import train_test_split
from models.triplet_loss import TripletLoss


BATCH_SIZE = 1
DROPOUT_RATE = 0.5
EPOCHS = 100
LEARNING_RATE = 1e-5
add_weights = './weights/torch/'
RNN_UNIT = 512
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#seq_length = 64

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
device = "CPU" if len(gpus) == 0 else "cuda:0"
print("Using device: {}".format(device))

def train_loop(model, train_iter, test_iter, optimizer,  epochs, device, add_weights_file, train_contr):
    if train_contr:
        print("Training contrastive module")
    else:
        print("Training with binary loss")
    log_training_loss = []
    log_training_accuracy = []
    log_testing_accuracy = []
    best_test_acc = 0.0
    best_test_auc = 0.0
    best_video_acc = 0.0
    best_video_auc = 0.0
    model.to(device)
    model.train()
    print("model", model)
    
    triplet_loss = TripletLoss(margin=0)
    clf_loss = nn.CrossEntropyLoss() #nn.NLLLoss()
    
    print("train iteration each epoch:",  len(train_iter))
    print("test iteration each epoch:",  len(test_iter))
    
    for epoch in trange(1, epochs + 1):
        emb_loss_sum, clf_loss_sum, loss_sum, acc_sum, samples_sum = 0.0, 0.0, 0.0, 0.0, 0
        iter_count = 0
        for X_anchor, X_pos, X_neg, Y_anchor, Y_pos, Y_neg, _, _, _ in train_iter:
            iter_count += 1
            #print("X_anchor", X_anchor.shape)(batch, seq_len, feat_dim)

            X = torch.concat((X_anchor, X_pos, X_neg), axis=0)#(batch*3, seq_len, feat_dim)
            y = torch.concat((Y_anchor, Y_pos, Y_neg), axis=0)
            X = X.to(device)
            y = y.to(device).long()
            y = torch.squeeze(y, -1)
            samples_num = X.shape[0]
            embeddings, output = model(X)

            embedding_loss = triplet_loss(embeddings[0], embeddings[1], embeddings[2]) 
            binary_loss = clf_loss(output, y)
            l = binary_loss + 0.1*embedding_loss# + 0.5*dist_ap
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            loss_sum += l.item() * samples_num
            acc_sum += calculate_accuracy(output, y) * samples_num
            clf_loss_sum += binary_loss.item() * samples_num
            emb_loss_sum += embedding_loss.item() * samples_num
            samples_sum += samples_num
               
        test_acc, test_auc, test_video_acc, test_video_auc, test_embedding = evaluate(model, test_iter, device)
        
        if test_acc >= best_test_acc:
            save_hint = "save the model to {}".format(add_weights_file)
            torch.save(model.state_dict(), add_weights_file)
            best_test_acc = test_acc
        else:
            save_hint = ""
            
        if test_auc > best_test_auc:
            best_test_auc = test_auc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if test_video_acc >  best_video_acc:
            best_video_acc = test_video_acc
        if test_video_auc >  best_video_auc:
            best_video_auc = test_video_auc
            

        tqdm.write("***************************************************************************************************")  
        tqdm.write("epoch:{} training: emb_loss:{:.4} clf_loss:{:.4}, loss:{:.4}, train_acc:{:.4}"
                   .format(epoch, emb_loss_sum/samples_sum, clf_loss_sum/samples_sum, loss_sum / samples_sum, acc_sum / samples_sum)
                   + save_hint)
        tqdm.write("epoch:{} testing: test_seq_acc:{:.4}, test_seq_auc:{:.4}, test_video_acc:{:.4}, test_video_auc:{:.4}"
                   .format(epoch, test_acc, test_auc, test_video_acc, test_video_auc)
                   + save_hint)
        tqdm.write("epoch:{} summary: best_seq_acc:{:.4} best_seq_auc:{:.4} best_video_acc:{:.4} best_video_auc:{:.4}"
                   .format(epoch, best_test_acc, best_test_auc, best_video_acc, best_video_auc)
                   + save_hint)
        tqdm.write("***************************************************************************************************")
        tqdm.write("\n")

        log_training_loss.append(loss_sum/samples_sum)
        log_training_accuracy.append(acc_sum/samples_sum)
        log_testing_accuracy.append(test_acc)
        
    log = {"loss": log_training_loss, "acc_train": log_training_accuracy, "acc_test": log_testing_accuracy}
    return log
  

def id_len_to_set(id_path, len_path):
    id_content = np.load(id_path)
    length_content = np.load(len_path)
    assert len(id_content)==sum(length_content)
    ret_set = []
    start = 0
    for idx, l in enumerate(length_content):
        single = id_content[start:start+l]
        ret_set.append(single)
        start = start + l
    return ret_set
     
def main():
    args = parse.parse_args()
    model_save = args.model
    contrastive = args.train_contrastive
    real_id_path = args.real_path
    real_len_path = args.real_len_path
    fake_id_path = args.fake_path
    fake_len_path = args.fake_len_path
    
    real_set = id_len_to_set(real_id_path, real_len_path)
    fake_set = id_len_to_set(fake_id_path, fake_len_path)
    
    train_real_set, test_real_set = train_test_split( real_set )
    train_fake_set, test_fake_set = train_test_split( fake_set )
    del real_set, fake_set
    """
    ###save test set start
    real_set_ids = None
    real_video_length = []
    for v in test_real_set:
        real_video_length.append(len(v))
        if real_set_ids is None:
            real_set_ids = v
        else:
            real_set_ids = np.concatenate((real_set_ids, v), axis=0)
    real_video_length = np.array(real_video_length)
    print("real_set_ids", real_set_ids.shape)
    print("real_video_length", sum(real_video_length))
    np.save("./id_vectors/FFpp/ffpp_real_ids_test.npy", real_set_ids)
    np.save("./id_vectors/FFpp/ffpp_real_lens_test.npy", real_video_length)
    
    fake_set_ids = None
    fake_video_length = []
    for v in test_fake_set:
        fake_video_length.append(len(v))
        if fake_set_ids is None:
            fake_set_ids = v
        else:
            fake_set_ids = np.concatenate((fake_set_ids, v), axis=0)
    fake_video_length = np.array(fake_video_length)
    print("fake_set_ids", fake_set_ids.shape)
    print("fake_video_length", sum(fake_video_length))
    np.save("./id_vectors/FFpp/ffpp_fake_ids_test.npy", fake_set_ids)
    np.save("./id_vectors/FFpp/ffpp_fake_lens_test.npy", fake_video_length)
    ###save test set end
    """
        
    

    train_dataset = Pair_Dataset( train_real_set, train_fake_set, phase="train" )
    test_dataset  = Pair_Dataset( test_real_set,  test_fake_set,  phase="test")

    print("train set and test and set get")
    train_iter = Data.DataLoader(train_dataset, BATCH_SIZE, shuffle=False, drop_last=True)
    test_iter = Data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=True)
    
    g1 = TI2Net(feature_dim=1024, rnn_unit=RNN_UNIT, dropout_rate=DROPOUT_RATE)
    optimizer = optim.Adam(g1.parameters(), lr=LEARNING_RATE)
    model_save = model_save+".pth"
    add_weights_file = join(add_weights, model_save)
    log_g1 = train_loop(g1, train_iter, test_iter, optimizer, EPOCHS, 
                        device, add_weights_file, contrastive)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--fake_selection', '-fakesele', type=str)
    parse.add_argument("--train_contrastive", "-contrastive", type=int, default=0)
    parse.add_argument('--branch', '-branch', type=str)
    parse.add_argument('--model', '-model', type=str)   
    parse.add_argument('--real_path', '-realids', type=str)
    parse.add_argument('--real_len_path', '-reallens', type=str)
    parse.add_argument('--fake_path', '-dfids', type=str)
    parse.add_argument('--fake_len_path', '-dflens', type=str)
    main()


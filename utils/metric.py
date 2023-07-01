import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch import concat, squeeze, isnan
from sklearn.metrics import accuracy_score, roc_auc_score


def calculate_accuracy(predict, target):
    return (predict.argmax(dim=1) == target).float().mean().item()

def calculate_auc(prediction, target):
    prediction = prediction[:,1]
    
    #prediction = prediction.cpu().detach().numpy()
    #target = target.cpu().detach().numpy()
    fpr, tpr, threshold = roc_curve(target, prediction)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def video_score_cal(score_list, strategy):
    if strategy=="maj_vote":
        score_count = len(score_list)
        score_list = np.array([np.argmax(predict) for predict in score_list])
        pos_count = len(np.where(score_list==1)[0])
        if pos_count<(score_count/2):
            return 0
        else:
            return 1
    elif strategy=="avg":
        score = np.mean([predict[1] for predict in score_list])
        return score
        
def video_metrics(video_score_dict, strategy="maj_vote"):
    #strategy ["maj_vote", "avg"]
    stra = strategy
    prediction_list = []
    label_list = []
    for video, info in video_score_dict.items():
        video_score = video_score_cal(info["scores"], strategy=stra)
        prediction_list.append(video_score)
        label_list.append(info["label"])
    video_auc = roc_auc_score(label_list, prediction_list)
    prediction_list = [0 if i<0.5 else 1 for i in prediction_list ]   
    video_acc = accuracy_score(label_list, prediction_list)
    return video_acc, video_auc
    
    
def evaluate(model, data_iter, device, video_strategy="maj_vote"):
    acc_sum, samples_sum = 0.0, 0
    fpr_sum, tpr_sum, auc_sum = 0.0,  0.0, 0.0
    model.to(device)
    model.eval()
    ground_truth_list = None
    prediction_list = None
    video_scores = {}
    real_embeddings = []
    fake_embeddings = []
    for X_pos, X_neg, Y_pos, Y_neg, Source_pos, Source_neg  in data_iter:
        X = concat((X_pos, X_neg), axis=0)
        y = concat((Y_pos, Y_neg), axis=0)
        sources = concat((Source_pos, Source_neg))        
        X = X.to(device)
        y = squeeze(y, -1)
        embeddings, predictions = model(X)
        embeddings = embeddings.cpu().data.numpy()
        predictions = predictions.cpu().data.numpy()
        for idx, pre in enumerate(predictions):
            #print("y[idx].cpu().data.numpy()", y[idx].cpu().data.numpy())
            if y[idx].cpu().data.numpy()==1:
                fake_embeddings.append(embeddings[idx])
            elif y[idx].cpu().data.numpy()==0:
                real_embeddings.append(embeddings[idx])
            else:
                print("Wrong label parsing")
            if not sources[idx] in video_scores.keys():
                video_scores.update({sources[idx]:{"scores":[pre], "label":y[idx]}})
            else:
                video_scores[sources[idx]]["scores"].append(pre)
        if ground_truth_list is None:
            ground_truth_list = y
        else:
            ground_truth_list = concat((ground_truth_list, y), 0)
        
        if prediction_list is None:
            prediction_list = predictions
        else:
            prediction_list = np.concatenate((prediction_list, predictions), axis=0)
        samples_sum += X.shape[0]
    model.train()
    predictions_bin = [i.argmax() for i in prediction_list]
    acc = accuracy_score(predictions_bin, ground_truth_list)
    try:
        fpr, tpr, auc = calculate_auc(prediction_list, ground_truth_list)
        video_acc1, video_auc1 = video_metrics( video_scores, strategy="maj_vote")
        video_acc2, video_auc2 = video_metrics( video_scores, strategy="avg" )
        video_acc = max(video_acc1, video_acc2)
        video_auc = max(video_auc1, video_auc2)
    except Exception as e:
        auc = 0.0
        video_acc = 0.0
        video_auc = 0.0
    return acc, auc, video_acc, video_auc, {"real_embeddings":real_embeddings, "fake_embeddings":fake_embeddings}
    #return acc, 0.0, 0.0, 0.0, {"real_embeddings":real_embeddings, "fake_embeddings":fake_embeddings}

def predict(model, data_iter, device):
    predictions = []
    model.to(device)
    model.eval()
    for X, _ in data_iter:
        X = X.to(device)
        output = model(X)
        prediction_batch = output.cpu().detach().numpy()
        predictions.append(prediction_batch)
    model.train()
    prediction_all = np.concatenate(predictions, axis=0)
    return prediction_all


def merge_video_prediction(mix_prediction, s2v, vc):
    """
    :param mix_prediction: The mixed prediction of 2 branches. (of each sample)
    :param s2v: Sample-to-video. Refer to the 'sample_to_video' in function get_data_for_test()
    :param vc: Video-Count. Refer to the 'count_y' in function get_data_for_test()
    :return: prediction_video: The prediction of each video.
    """
    prediction_video = []
    pre_count = {}
    for p, v_label in zip(mix_prediction, s2v):
        p_bi = 0
        if p >= 0.5:
            p_bi = 1
        if v_label in pre_count:
            pre_count[v_label] += p_bi
        else:
            pre_count[v_label] = p_bi
    for key in pre_count.keys():
        prediction_video.append(pre_count[key] / vc[key])
    return prediction_video

def plot_ROC(fpr, tpr, roc_auc, name):#, roc_auc
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label="ROC curve(area = %0.2f)" % roc_auc ) 
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig("./figs/ROC/"+name+".png")
    plt.show()

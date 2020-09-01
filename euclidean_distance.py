# TODO: balanced test data
import torch
import torchvision.models as models
from torch import nn
import numpy as np
import os
import sys
from utils import h5_read, h5_save, image_pair_generator, image_pair_length, data_transforms
from preprocess import generate_feature
import config
import argparse
from sklearn.metrics import roc_auc_score, roc_curve


# ------------ Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='suzhou960x720', choices=['suzhou960x720', 'swiss960x720'], help='choose which data subset to be tested')
parser.add_argument('--test', default='balanced', choices=['balanced', 'unbalanced'], help='choose how to generate negative samples: equal to positive samples or more than them')
opt = parser.parse_args()

if opt.data == 'suzhou960x720':
    img_path = config.SUZHOU_960x720
elif opt.data == 'swiss960x720':
    img_path = config.SWISS_960x720
else:
    sys.exit("wrong '--data' option")

feature_map_file_name = os.path.join(config.MID_PRODUCT, 'features_'+opt.data+'_res34.h5')
dist_file_name = os.path.join(config.MID_PRODUCT, 'dist_'+opt.data+'_res34_eval.npy')
dist_fig_name = os.path.join(config.RESULT_DIR, 'model1_'+opt.data+'_'+opt.test+'_dist.png')
pred_fig_name = os.path.join(config.RESULT_DIR, 'model1_'+opt.data+'_'+opt.test+'_prediction.png')
ROC_fig_name = os.path.join(config.RESULT_DIR, 'model1_'+opt.data+'_'+opt.test+'_ROC.png')

feature_shape = config.RES34_960x720_SHAPE

# ------------ Initialization
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
normalize = data_transforms['norm']

# ------------ Generate Feature Maps


n = image_pair_length(img_path)
if not os.path.exists(feature_map_file_name):
    generate_feature(data_dir=img_path,
                     feature_map_file_name=feature_map_file_name,
                     feature_shape=feature_shape)
else:
    print("Feature maps file already exists, we just read it.")
f_a, f_b = h5_read(feature_map_file_name)

# ------------ Compute distance
# print("A domain feature maps size:", f_a.shape)
if opt.test == 'unbalanced':
    dst=np.zeros((n,n))
    for shift in range(n):
        for idx in range(n):
            a = f_a[idx:idx+1]
            b = f_b[(idx+shift)%n:(idx+shift)%n+1]
            dst[idx,shift] = np.linalg.norm(a - b)
            print('dst(idx,shift)(%d,%d)=%f' % (idx,shift,dst[idx,shift]))
    np.save(dist_file_name, dst)

# ------------ Visualize
import seaborn as sns
import matplotlib.pyplot as plt

fig_size = [10,10]
plt.rcParams["figure.figsize"] = fig_size
plt.axis('equal')
ax = sns.heatmap(dst,
                 xticklabels=2,
                 yticklabels=2)
ax.set_xlabel('Shift')
ax.set_ylabel('Image Index')
ax.set_title('Distance Matrix')
# plt.show()
plt.savefig(dist_fig_name, bbox_inches='tight')
plt.close()

# ------------ Analysis
print('Min, Max and Mean of Distances:')
print(np.min(dst), np.max(dst), np.average(dst))
lower, higher = np.min(dst), np.max(dst)

dst_t, dst_f = dst[:,0], dst[:,1:]
print('Min, Max and Mean of paired images:', np.min(dst_t), np.max(dst_t), np.mean(dst_t))
print('Min, Max and Mean of unpaired images:', np.min(dst_f), np.max(dst_f), np.mean(dst_f))

print('Sorted distance of paired images:')
print(np.sort(dst_t))

print('First n sorted distance of unpaired images:')
print(np.sort(dst_f.flatten())[:n])

# ------------ Evaluate
def predict(dst, threshold):
    return (dst <= threshold).astype(np.int)
def ground_truth(dst):
    n = dst.shape[0]
    gt = np.zeros((n,n)).astype(np.int)
    gt[:,0] = 1
    return gt
def confusion_matrix(pred, gt):
    n = gt.shape[0]
    TP = np.sum(gt[:,0] == pred[:,0])
    FN = np.sum(gt[:,0] != pred[:,0])
    TN = np.sum(gt[:,1:] == pred[:,1:])
    FP = np.sum(gt[:,1:] != pred[:,1:])
    TPR=TP/(TP+FN)
    FPR=FP/(FP+TN)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    return TP,FP,TN,FN, TPR,FPR, ACC,precision,recall,F1

# threshold = int(input("Please input an integer as the threshold:"))
percentile = 0.95
idx_percent = int(n*percentile)
threshold = (np.sort(dst_t)[idx_percent] + np.sort(dst_t)[idx_percent+1]) // 2
print("threshold =", threshold)
pred = predict(dst, threshold)
gt = ground_truth(dst)

prompt = "TP, FP, TN, FN, TPR, FPR, Accuracy, Precision, Recall, F1".split(', ')
print(list(zip(prompt,confusion_matrix(pred, gt))))

plt.axis('equal')
ax = sns.heatmap(pred,
                 xticklabels=2,
                 yticklabels=2)
ax.set_xlabel('Shift')
ax.set_ylabel('Image Index')
ax.set_title('Prediction Matrix')
# plt.show()
plt.savefig(pred_fig_name, bbox_inches='tight')
plt.close()

# ------------ Draw ROC curve
np.seterr(divide='ignore',invalid='ignore')
ROC_x, ROC_y = [0], [0]
for threshold in range(int(lower),int(higher)+1):
    pred = predict(dst, threshold)
    gt = ground_truth(dst)
    conf_mat = confusion_matrix(pred, gt)
    x, y = conf_mat[5], conf_mat[4] # x: FPR, y: TPR
    ROC_x.append(x)
    ROC_y.append(y)
ROC_x.append(1)
ROC_y.append(1)

print("ROC:", "-"*80)
print(ROC_x)
print(ROC_y)

plt.plot(ROC_x, ROC_y)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis('equal')
# plt.show()
plt.savefig(ROC_fig_name, bbox_inches='tight')
plt.close()

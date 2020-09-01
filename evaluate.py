# TODO: refactor using sklearn.metrics.roc_curve and calculate auc using metrics.auc
# TODO: refactor to reduce code redundancy
import torch.nn.functional as F
import net
from Dataset import *
import time
import argparse
import config
import sys
from utils import data_transforms
from sklearn.metrics import roc_auc_score, roc_curve

def confusion_matrix(pred, gt):
    assert pred.shape == gt.shape

    P = 0 # 0 is positive, paired data
    N = 1

    TP = np.sum(np.logical_and(gt == pred, pred == P))
    TN = np.sum(np.logical_and(gt == pred, pred == N))
    FP = np.sum(np.logical_and(gt != pred, pred == P))
    FN = np.sum(np.logical_and(gt != pred, pred == N))
    
    TPR=TP/(TP+FN)
    FPR=FP/(FP+TN)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TPR
    F1 = 2*precision*recall/(precision+recall)
    return TP, FP, TN, FN, TPR, FPR, ACC, precision, recall, F1

def evaluate_Siamese(model, device, margin, data):
    print("Test starts.")

    # Load datasets
    print("Loading data...")
    if data == 'raw':
        image_datasets = {
            x: SatUAVDataset(csv_meta='raw.csv', # evaluate does not run on augmented data
                             csv_file=f'{x}.csv',
                             root_dir=config.DATA_DIR,
                             transform=data_transforms['norm']) for x in ['train', 'val']
        }
    elif data == 'england':
        image_datasets = {
            x: SatUAVDataset(csv_meta='england.csv', # evaluate does not run on augmented data
                             csv_file=f'england_{x}.csv',
                             root_dir=config.DATA_DIR,
                             transform=data_transforms['norm']) for x in ['train', 'val']
        }
        pass
    else:
        print("To evaluate Siamese based networks, data must be raw or england!")
    batch_size = 1
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    n = dataset_sizes['train'] + dataset_sizes['val']
    print("Data Loading: Done.")

    # Predict and get prediction matrix
    print("Predicting...")
    output_matrix = {x: np.zeros(dataset_sizes[x]) for x in ['train', 'val']}
    label_matrix = {x: np.zeros(dataset_sizes[x]) for x in ['train', 'val']}
    since = time.time()
    for phase in ['train', 'val']:
        print(phase, "phase:")
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            print(i_batch+1, '/', dataset_sizes[phase], end='\r')
            A = sample_batched['A'].to(device)
            B = sample_batched['B'].to(device)
            # labels = sample_batched['label'].to(device)
            with torch.set_grad_enabled(False):
                outputs = model(A, B)
                dist = F.pairwise_distance(outputs[0], outputs[1])
                output_matrix[phase][i_batch] = dist.cpu().data.numpy()[0]
                label_matrix[phase][i_batch] = sample_batched['label'].numpy()[0]
        print()
    print((time.time()-since)/n, 'seconds/pair')

    # Draw ROC curve for test data
    fpr, tpr, thresholds = roc_curve(label_matrix['val'], output_matrix['val'])
    auc_score = roc_auc_score(label_matrix['val'], output_matrix['val'])
    print(f"AUC score is {auc_score}.")
    print("↓↓↓↓↓↓↓↓↓ ROC data ↓↓↓↓↓↓↓↓↓↓")
    print(fpr.tolist(), ',', tpr.tolist())
    print("↑↑↑↑↑↑↑↑↑ ROC end  ↑↑↑↑↑↑↑↑↑↑")

    # generate prediction result based on threshold = margin/2
    print("Prediction result based on threshold = margin/2 ")
    pred_matrix = {x: (output_matrix[x] > margin/2)
                   * 1 for x in ['train', 'val']}
    for x in ['train', 'val']:
        result = confusion_matrix(pred_matrix[x], label_matrix[x])
        print(f'  {x} data:', ':')
        for k,v in zip('TP,FP,TN,FN,TPR,FPR,ACC,precision,recall,F1'.split(','), result):
            print("    ", k, ':', v)

def evaluate_Siamese_Error_Tolerance(model, device, margin):
    print("Test starts.")
    # Predict and get prediction matrix
    resize_norm = data_transforms['resize_norm']
    print("Predicting...")

    for i in range(1, 6):
        print('-'*80)
        print(i)
        A = Image.open(os.path.join(config.ERROR_TOLERANCE, f'{i}/raw/raw.jpg'))
        A = resize_norm(A).unsqueeze(0).to(device)
        for shift in [15, 30, 45]:
            print(f"  {shift}meters:")
            for direction in ['E', 'W', 'S', 'N']:
                print(f"    {direction}{shift}.jpg:", end='')
                B = Image.open(os.path.join(config.ERROR_TOLERANCE, f"{i}/{shift}meters/{direction}{shift}.jpg"))
                B = resize_norm(B).unsqueeze(0).to(device)
                with torch.set_grad_enabled(False):
                    outputs = model(A, B)
                    dist = F.pairwise_distance(outputs[0], outputs[1])
                    d = dist.cpu().data.numpy()
                    print(d, ", matched!" if d[0] <= margin/2 else ", not matched.")


def evaluate_FCNet(model, device):
    print("Test starts.")
    feature_file = config.FULL_960x720_FEATURE_RES34

    # Load datasets
    print("Loading data...")
    image_datasets = {x: SatUAVH5Dataset(csv_file=os.path.join(config.MID_PRODUCT, f'{x}.csv'),
                                         feature_file=feature_file) for x in ['train', 'val']}
    batch_size = 1
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    n = dataset_sizes['train'] + dataset_sizes['val']
    print("Data Loading: Done.")

    # Predict and get prediction matrix
    print("Predicting...")
    output_matrix = {x: np.zeros(dataset_sizes[x]) for x in ['train', 'val']}
    label_matrix = {x: np.zeros(dataset_sizes[x]) for x in ['train', 'val']}
    since = time.time()
    for phase in ['train', 'val']:
        print(phase, "phase:")
        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            print(i_batch+1, '/', dataset_sizes[phase], end='\r')
            A = sample_batched['A'].to(device)
            B = sample_batched['B'].to(device)
            # labels = sample_batched['label'].to(device)
            with torch.set_grad_enabled(False):
                outputs = model(A, B)
                output_matrix[phase][i_batch] = outputs.cpu().data.numpy()[0]
                label_matrix[phase][i_batch] = sample_batched['label'].numpy()[0]
        print()
    print((time.time()-since)/n, 'seconds/pair')

    # Draw ROC curve for test data
    # fpr, tpr, thresholds = roc_curve(label_matrix['val'], output_matrix['val'])
    # auc_score = roc_auc_score(label_matrix['val'], output_matrix['val'])
    # print(f"AUC score is {auc_score}.")
    # print("↓↓↓↓↓↓↓↓↓ ROC data ↓↓↓↓↓↓↓↓↓↓")
    # print(fpr, ',', tpr)
    # print("↑↑↑↑↑↑↑↑↑ ROC end  ↑↑↑↑↑↑↑↑↑↑")

    # generate prediction result based on threshold = 0.5
    print("Prediction result based on threshold = 0.5 ")
    pred_matrix = {x: (output_matrix[x] > 0.5)
                   * 1 for x in ['train', 'val']}
    for x in ['train', 'val']:
        result = confusion_matrix(pred_matrix[x], label_matrix[x])
        print(x, 'data:', ':')
        for k,v in zip('TP,FP,TN,FN,TPR,FPR,ACC,precision,recall,F1'.split(','), result):
            print(k, ':', v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_names = sorted(name for name in net.__dict__
                         if name.endswith("Net") and callable(net.__dict__[name]))
    parser.add_argument('--model', default='SemiSiameseNet', choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: SemiSiameseNet)')
    parser.add_argument('--weight', type=str, help='weight file')
    parser.add_argument('--data', default='raw', choices=['raw', 'err', 'england'])
    parser.add_argument('--margin', type=float, default=4,
                        help='margin of Contrastive Loss, only useful in Siamese Network')

    opt = parser.parse_args()
    print(opt)
    sys.stdout.flush()

    if opt.model not in ['SiameseResNet', 'SiameseSqueezeNet', 'FCNet']:
        quit(f"Evaluation for {opt.model} is not implemented yet.")
        # raise NotImplementedError()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sys.stdout.flush()

    model = getattr(net, opt.model)()
    model.to(device)
    model_path = os.path.join(config.MODEL_DIR, opt.weight)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded.")
    sys.stdout.flush()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(model)
    sys.stdout.flush()

    if opt.model in ['SiameseResNet', 'SiameseSqueezeNet']:
        if opt.data == 'raw' or 'england':
            evaluate_Siamese(model, device, opt.margin, opt.data)
        elif opt.data == 'err':
            evaluate_Siamese_Error_Tolerance(model, device, opt.margin)
    elif opt.model in ['FCNet']:
        evaluate_FCNet(model, device)
    else:
        quit(f"{opt.model} is not supported.")



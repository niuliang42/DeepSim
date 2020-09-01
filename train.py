import gc
import argparse
import os
import time
import datetime
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import net
import config
from Dataset import SatUAVH5Dataset, SatUAVDataset
from utils import data_transforms

model_names = sorted(name for name in net.__dict__
                     if name.endswith("Net")
                     and callable(net.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=25,  help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16,  help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--step', type=int, default=10, help='learning rate step size')
parser.add_argument('--margin', type=float, default=4,
                    help='margin of ContrastiveLoss, only useful in Siamese Network')
parser.add_argument('--data', default='raw', choices=['raw', 'aug', ], help='only raw data or with augmented data')
parser.add_argument('--model', default='FCNet', choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: FCNet)')
opt = parser.parse_args()
print(opt)

def train_model(model, dataloaders, device,
                criterion, optimizer, scheduler, time_str, num_epochs=25,):
    print( model.__class__.__name__, 'starts to train.')# TODO: check correctness
    train_start_time = time.time()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # epoch_loss = 1000 # only effective using ReduceLROnPlateau
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                # scheduler.step(epoch_loss) # only effective using ReduceLROnPlateau
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            Siamese_acc = {'TP':0, 'TN':0, 'FP':0, 'FN':0}

            # Iterate over data.
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                A = sample_batched['A'].to(device)
                B = sample_batched['B'].to(device)
                labels = sample_batched['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(A, B)
                    loss = criterion(outputs, labels)
                    if model.__class__.__name__ in ['SiameseResNet', 'SiamesePiNet', 'SiameseSqueezeNet', 'FCSiameseNet']:
                        dist = F.pairwise_distance(outputs[0], outputs[1])
                        preds = (dist.cpu().data.numpy()[:, np.newaxis] > (opt.margin/2))*1
                        Siamese_acc['TP'] += np.sum(np.logical_and(labels.cpu().data.numpy()==preds, preds==1))
                        Siamese_acc['TN'] += np.sum(np.logical_and(labels.cpu().data.numpy()==preds, preds==0))
                        Siamese_acc['FP'] += np.sum(np.logical_and(labels.cpu().data.numpy()!=preds, preds==1))
                        Siamese_acc['FN'] += np.sum(np.logical_and(labels.cpu().data.numpy()!=preds, preds==0))
                    else:
                        preds = (outputs.cpu().data.numpy() > 0.5) * 1


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * sample_batched['A'].size(0)
                running_corrects += torch.sum(torch.from_numpy(preds) == labels.cpu().long())

                # For DEBUG
                print("batch:%d/%d, loss:%.4f" %
                      (i_batch, len(dataloaders[phase]), loss.item() * sample_batched['A'].size(0)),
                      end='  |  ', flush=True)
                def rs(s):
                    return " ".join(str(s).replace('\n', ' ').split())
                if (1+epoch) % 5 == 2:
                    if model.__class__.__name__ in ['SiameseResNet', 'SiamesePiNet', 'SiameseSqueezeNet', 'FCSiameseNet']:
                        data_str=('%s, %s, %s, %s' %
                                  ( rs(dist.cpu().data), rs(preds),
                                    rs(labels.cpu().data), torch.sum(torch.from_numpy(preds) == labels.cpu().long())
                                    )
                                  )
                    else:
                        data_str = ('%s, %s, %s' % (rs(outputs.cpu().data), rs(preds), rs(labels.cpu().data)))
                    print(data_str)

                # save memory to avoid memory usage exceeds limitation on Dalma
                del A, B, outputs, loss, labels
                if config.ENV == "Dalma":
                    torch.cuda.empty_cache()
                gc.collect()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if model.__class__.__name__ in ['SiameseResNet', 'SiamesePiNet', 'SiameseSqueezeNet', 'FCSiameseNet']:
                print("\n%s, TPR(Paired acc):%.2f, TNR(Unpaired acc):%.2f" %
                    (phase, Siamese_acc['TP']/(Siamese_acc['TP']+Siamese_acc['FN']),
                    Siamese_acc['TN']/(Siamese_acc['TN']+Siamese_acc['FP']),), end=' | ')
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format( phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
            epoch_time_elapsed // 60, epoch_time_elapsed % 60))

        print()

    train_time_elapsed = time.time() - train_start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        train_time_elapsed // 60, train_time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save model
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, '%s_%s_final.pth'%(opt.model, time_str)))
    torch.save(best_model_wts, os.path.join(config.MODEL_DIR, '%s_%s_best.pth'%(opt.model, time_str)))

    # load best model weights and return
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    # Initilization models and data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if opt.model in ['SiameseResNet', 'SiameseSqueezeNet']:
        num_workers = 0
        model = net.SiameseResNet() if opt.model == 'SiameseResNet' else net.SiameseSqueezeNet()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1) # Decay LR by a factor of 0.1 every opt.step epochs
        criterion = net.ContrastiveLoss(margin=opt.margin)
        # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        image_datasets = {
            x: SatUAVDataset(csv_meta=f'{opt.data}.csv' if x=='train' else 'raw.csv',
                             csv_file=f'{x}.csv',
                             root_dir=config.DATA_DIR,
                             transform=data_transforms['norm']) for x in ['train', 'val']
        }
    elif opt.model == 'FCNet':
        num_workers = 1
        feature_file = config.FULL_960x720_FEATURE_RES34
        model = net.FCNet()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1) # Decay LR by a factor of 0.1 every opt.step epochs
        criterion = nn.BCELoss()
        image_datasets = {x: SatUAVH5Dataset(csv_file=os.path.join(config.MID_PRODUCT, f'{x}.csv'),
                                             feature_file=feature_file) for x in ['train', 'val']}
    elif opt.model == 'FCSiameseNet':
        num_workers = 1
        feature_file = config.FULL_960x720_FEATURE_RES34
        model = net.FCSiameseNet()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1) # Decay LR by a factor of 0.1 every opt.step epochs
        criterion = net.ContrastiveLoss(margin=opt.margin)
        image_datasets = {x: SatUAVH5Dataset(csv_file=os.path.join(config.MID_PRODUCT, f'{x}.csv'),
                                             feature_file=feature_file) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,
                                                  shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    print(model.__class__.__name__, 'is created at:', time_str)

    # Training
    print(model)
    model = train_model(model, dataloaders, device,
                        criterion, optimizer, lr_scheduler, time_str,
                        num_epochs=opt.nepoch)

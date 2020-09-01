import os
import sys
import shutil
import numpy as np
import config
import utils
import csv
import torch
import torchvision.models as models
from torch import nn
from time import sleep
from tqdm import tqdm, trange
from utils import h5_save, data_transforms, image_pair_generator, image_pair_length, get_prefix


def rename(beg, end, src, dst, target_beg=1, DEBUG=True, prefix=''):
    '''
    raw data format: each paired in a single folder named from 1 to n.
    src: source path
    dst: destination path
    '''
    print(f"Renaming : {src} -> {dst}")
    print('')
    for i in trange(beg, end+1):
        p = os.path.join(src, str(i))
        l = os.listdir(p)
        l = [x for x in l if 'jpg' in x.lower()]

        src_a_name = str(i) + '.jpg'
        dst_a_name = '%04d_A.jpg' % (i+target_beg-beg)
        src_a = os.path.join(p, src_a_name)
        dst_a = os.path.join(dst, dst_a_name)

        src_b_name = [x for x in l if x.lower() != str(i) + '.jpg']
        assert len(src_b_name) == 1
        src_b_name = src_b_name[0]
        dst_b_name = '%04d_B.jpg' % (i+target_beg-beg)
        src_b = os.path.join(p, src_b_name)
        dst_b = os.path.join(dst, dst_b_name)
        infostr = f"{src_a_name} -> {dst_a_name} ; {src_b_name} -> {dst_b_name}"

        tqdm.write(prefix + infostr)
        if DEBUG:
            sleep(0.05)
            pass
        else:
            shutil.copyfile(src_a, dst_a)
            shutil.copyfile(src_b, dst_b)

def rename_many(param_list, dst, DEBUG=True, prefix=''):
    starts_from = 1
    for beg, end, src in param_list:
        rename(beg, end, src, dst, target_beg=starts_from, DEBUG=DEBUG, prefix=prefix)
        starts_from += end - beg + 1
        print('')

def resize(src, dst, x, y, DEBUG=True, prefix=''):
    '''
    resize images in src to dst as resolution x,y.
    Notice: images in domain A is 16:13, images in domain B is 4:3.
    We may need to work on this a little bit.
    '''
    print(f"Resizing : {src} -> {dst}")
    print('')
    l = sorted(os.listdir(src))
    l = [x for x in l if 'jpg' in x.lower()]
    for f in tqdm(l):
        f_src = os.path.join(src, f)
        f_dst = os.path.join(dst, f)
        # convert myfigure.png -resize 200x100 myfigure.jpg
        cmd = f'convert {f_src} -resize {x}x{y}! {f_dst}'
        if DEBUG:
            tqdm.write(prefix + cmd)
            sleep(0.05)
        else:
            tqdm.write(prefix + f)
            os.system(cmd)

def train_val_divide(path, train_percentage=0.80, seed=42):
    assert train_percentage<=1 and train_percentage >= 0
    l = sorted(os.listdir(path))
    l = [x for x in l if 'jpg' in x.lower()]
    assert len(l) % 2 == 0
    n = len(l) // 2
    A_list, B_list = utils.name_list(n, path='')
    # A_list, B_list = np.array(A_list), np.array(B_list)
    np.random.seed(seed)
    permut = np.random.permutation(n)
    permut_train, permut_val = permut[:int(n*train_percentage)], permut[int(n*train_percentage):]
    with open(os.path.join(path, 'train.csv'), 'w') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow(['A', 'B'])
        for idx in permut_train:
            csvWriter.writerow([A_list[idx], B_list[idx]])
    with open(os.path.join(path, 'val.csv'), 'w') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerow(['A', 'B'])
        for idx in permut_val:
            csvWriter.writerow([A_list[idx], B_list[idx]])
    return permut_train, permut_val

def generate_feature(data_dir, feature_map_file_name, feature_shape, eval=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sys.stdout.flush()
    normalize = data_transforms['norm']

    # Pretrained Model
    pretrained_model = models.resnet34(pretrained=True)
    if config.RESNET_POOLING == 'fixed' and str(pretrained_model.avgpool)[:8] == 'Adaptive':
        pretrained_model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
    del pretrained_model
    feature_extractor.to(device)
    if eval:
        feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    print(feature_extractor)
    sys.stdout.flush()

    # Generating
    n = image_pair_length(data_dir)
    f_a = np.zeros((n,)+feature_shape)
    f_b = np.zeros((n,)+feature_shape)
    for i, (a_img, b_img) in enumerate(image_pair_generator(data_dir)):
        print("Generating feature maps of %d th pair." % (i))
        sys.stdout.flush()
        a_tensor = normalize(a_img).unsqueeze(0).to(device)
        b_tensor = normalize(b_img).unsqueeze(0).to(device)
        a = feature_extractor(a_tensor).squeeze()
        b = feature_extractor(b_tensor).squeeze()
        f_a[i] = a.cpu().detach().numpy()
        f_b[i] = b.cpu().detach().numpy()
    h5_save(feature_map_file_name, f_a, f_b)

def _full_pipeline(DEBUG=True):
    print("Move and rename all images from different folders to a single folder using an unified naming.")
    prefix = get_prefix(DEBUG)
    rename_many(
        [
            (1  , 20 , config.KUNSHAN_1_RAW),
            (1  , 49 , config.PARIS_1_RAW),
            (1  , 9  , config.SHENNONGJIA_1_RAW),
            (1  , 100, config.SUZHOU_1_RAW),
            (101, 200, config.SUZHOU_2_RAW),
            (201, 300, config.SUZHOU_3_RAW),
            (301, 343, config.SUZHOU_4_RAW),
            (1  , 160, config.SWISS_1_RAW),
            (0  , 39 , config.SWISS_2_RAW),
            (1  , 113, config.SWISS_3_RAW),
            (1  , 57 , config.WEIHAI_1_RAW),
            (1  , 69 , config.WUXI_1_RAW),
        ],
        dst=config.FULL_DATA,
        DEBUG=DEBUG,
        prefix=prefix
    )
    resize(config.FULL_DATA, config.FULL_960x720, x=960, y=720, DEBUG=DEBUG, prefix=prefix)
    print(train_val_divide(config.FULL_960x720))

def _swiss_pipeline(DEBUG=True):
    print("Move and rename Swiss images from different folders to a single folder using an unified naming.")
    prefix = get_prefix(DEBUG)
    rename_many(
        [
            (1  , 160, config.SWISS_1_RAW),
            (0  , 39 , config.SWISS_2_RAW),
            (1  , 113, config.SWISS_3_RAW),
        ],
        dst=config.SWISS_DATA,
        DEBUG=DEBUG,
        prefix=prefix
    )
    resize(config.SWISS_DATA, config.SWISS_960x720, x=960, y=720, DEBUG=DEBUG, prefix=prefix)
    # was 1280x720 for swiss
    print(train_val_divide(config.SWISS_960x720))

def _suzhou_pipeline(DEBUG=True):
    print("Move and rename Suzhou images from different folders to a single folder using an unified naming.")
    prefix = get_prefix(DEBUG)
    rename_many(
        [
            (1  , 100, config.SUZHOU_1_RAW),
            (101, 200, config.SUZHOU_2_RAW),
            (201, 300, config.SUZHOU_3_RAW),
            (301, 343, config.SUZHOU_4_RAW),
        ],
        dst=config.SUZHOU_DATA,
        DEBUG=DEBUG,
        prefix=prefix
    )
    resize(config.SUZHOU_DATA, config.SUZHOU_960x720, x=960, y=720, DEBUG=DEBUG, prefix=prefix)
    # was 1280x720 for swiss
    print(train_val_divide(config.SUZHOU_960x720))

def _england_pipeline(DEBUG = True):
    prefix = get_prefix(DEBUG)
    def rename_subfolders(path):
        for folder_name in os.listdir(path):
            cmd = f"mv {os.path.join(path, folder_name)} {os.path.join(path, folder_name[2:])}"
            print(cmd)
            if not DEBUG:
                os.system(cmd)
    rename_subfolders(config.ENGLAND_BIRMINGHAM_RAW)
    rename_subfolders(config.ENGLAND_COVENTRY_RAW)
    rename_subfolders(config.ENGLAND_LIVERPOOL_RAW)
    rename_subfolders(config.ENGLAND_PEAK_RAW)

    rename_many(
        [
            (1, 37, config.ENGLAND_BIRMINGHAM_RAW),
            (2,  3, config.ENGLAND_COVENTRY_RAW),
            (5, 13, config.ENGLAND_COVENTRY_RAW),
            (15,18, config.ENGLAND_COVENTRY_RAW),
            (1, 41, config.ENGLAND_LIVERPOOL_RAW),
            (1, 14, config.ENGLAND_PEAK_RAW),
        ],
        dst = config.ENGLAND_DATA,
        DEBUG=DEBUG,
        prefix=''
    )
    resize(
        src = config.ENGLAND_DATA,
        dst = config.ENGLAND_960x720,
        x = 960,
        y = 720,
        DEBUG = DEBUG,
        prefix= prefix
    )
    print(train_val_divide(config.ENGLAND_960x720, train_percentage=0))

if __name__ == '__main__':
    # _full_pipeline(DEBUG=False)
    # _swiss_pipeline(DEBUG=False)
    # _suzhou_pipeline(DEBUG=False)
    # generate_feature(data_dir=config.FULL_960x720,
    #                  feature_map_file_name=config.FULL_960x720_FEATURE_RES34,
    #                  feature_shape=config.RES34_960x720_SHAPE, )

    _england_pipeline(DEBUG=False)
    print('Done.')


# This is file contains the functions needed to load GPS Spoofing Dataset.
import os
import config
import numpy as np
import h5py
from torchvision import transforms
from matplotlib.pyplot import imread
from PIL import Image
from sklearn import metrics
from tqdm._utils import _term_move_up

def name_list(n, path = ''):
    '''
    given n, return file name lists A and B
    if path is given, return 
    '''
    if path == '':
        A = ['%04d_A.jpg'%(i) for i in range(1,n+1)]
        B = ['%04d_B.jpg'%(i) for i in range(1,n+1)]
    else:
        A = [os.path.join(path, '%04d_A.jpg'%(i)) for i in range(1,n+1)]
        B = [os.path.join(path, '%04d_B.jpg'%(i)) for i in range(1,n+1)]
    return A, B

def load_image_pairs(path):
    l = sorted(os.listdir(path))
    l = [x for x in l if 'jpg' in x.lower()]
    assert len(l) % 2 == 0
    A_list, B_list = name_list( len(l) // 2, path=path)

    A = np.array([imread(fname) for fname in A_list])
    B = np.array([imread(fname) for fname in B_list])

    return A, B

def image_pair_length(path):
    l = sorted(os.listdir(path))
    l = [x for x in l if 'jpg' in x.lower()]
    assert len(l) % 2 == 0
    return len(l)//2

def image_pair_generator(path):
    l = sorted(os.listdir(path))
    l = [x for x in l if 'jpg' in x.lower()]
    assert len(l) % 2 == 0
    A_list, B_list = name_list( len(l) // 2, path=path)

    for A_name, B_name in zip(A_list, B_list):
        yield Image.open(A_name), Image.open(B_name)

def norm_brightness(img):
    '''
    img: ndarray (H,W,C)
    http://www.voidcn.com/article/p-qshwwtte-qv.html
    '''
    mean = np.mean(img)
    img = img - mean
    img = img*1.5 + mean*0.7 #修对比度和亮度
    return img

def h5_save(fname, f_a, f_b):
    '''save f_a and f_b as fname'''
    with h5py.File(fname, 'w') as f:
        f.create_dataset('f_a', data=f_a)
        f.create_dataset('f_b', data=f_b)

def h5_read(fname):
    '''read fname and return f_a and f_b'''
    with h5py.File(fname, 'r') as f:
        return f['f_a'][:], f['f_b'][:]

def auc(x, y):
    area = 0
    last_x = 0
    for x,y in zip(x,y):
        area += y*(x-last_x)
        last_x = x
    return area

def auc2(x, y):
    assert sorted(x) == x and sorted(y) == y
    x.reverse()
    y.reverse()
    x.append(0)
    y.append(0)
    area = 0
    last_x = 1
    last_y = 1
    for x,y in zip(x,y):
        if x != last_x:
            area += last_y*(last_x-x)
            last_x = x
            last_y = y
    return area

def auc3(x,y):
    return metrics.auc(x,y)

data_transforms = {
    'norm': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'resize_norm': transforms.Compose([
        transforms.Resize((720, 960)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

tqdm_prefix = _term_move_up() + '\r'
def get_prefix(DEBUG):
    return tqdm_prefix if not DEBUG else ''

if __name__ == '__main__':
    # print(name_list(60, path='/niu/'))
    A, B = load_image_pairs(path=config.SWISS_1280x720)
    print(A.shape, B.shape)
    # print(load_image_pairs())
    import cv2
    for i in range(A.shape[0]):
        a = norm_brightness(A[i])
        b = norm_brightness(B[i])
        cv2.imshow('A[%d]'%i,a/255.0)
        cv2.waitKey(0)
        cv2.imshow('B[%d]'%i,b/255.0)
        cv2.waitKey(0)
    #print('Done.')

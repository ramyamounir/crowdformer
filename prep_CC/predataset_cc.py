import glob
import math
import os
import torch
import cv2
import h5py, json
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import random

'''set your data path'''
root = '/home/ramy/Documents/crowdformer/data/CraneCounting/'
november_train = os.path.join(root, 'November/train', 'images')
november_test = os.path.join(root, 'November/test', 'images')
novermber_annots = os.path.join(root, 'November', 'annotations_file.json')
path_sets = [november_train, november_test]

'''for November set'''
if not os.path.exists(november_train.replace('images', 'gt_density_map_crop')):
    os.makedirs(november_train.replace('images', 'gt_density_map_crop'))

if not os.path.exists(november_train.replace('images', 'images_crop')):
    os.makedirs(november_train.replace('images', 'images_crop'))

if not os.path.exists(november_test.replace('images', 'gt_density_map_crop')):
    os.makedirs(november_test.replace('images', 'gt_density_map_crop'))

if not os.path.exists(november_test.replace('images', 'images_crop')):
    os.makedirs(november_test.replace('images', 'images_crop'))



# get image paths
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.JPG')):
        img_paths.append(img_path)
img_paths.sort()

# get annotations
with open(novermber_annots, 'r') as f:
    annots = json.load(f)

annots_processed = {}
for k,img_vals in annots.items():
    if len(img_vals["regions"]):
        annots_processed[f'{img_vals["filename"]}'] = [[region['shape_attributes']['cx'], region['shape_attributes']['cy']] for region in img_vals["regions"]]



# actual preparation
np.random.seed(0)
random.seed(0)
for img_path in img_paths:

    img_path_base = os.path.basename(img_path)
    if img_path_base not in annots_processed.keys():
        continue


    Img_data = cv2.imread(img_path)
    Gt_data = np.array(annots_processed[os.path.basename(img_path)])

    rate = 1
    rate_1 = 1
    rate_2 = 1
    flag = 0
    if Img_data.shape[1] >= Img_data.shape[0]:  # 后面的大
        rate_1 = 1152.0 / Img_data.shape[1]
        rate_2 = 768 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_1
        Gt_data[:, 1] = Gt_data[:, 1] * rate_2

    elif Img_data.shape[0] > Img_data.shape[1]:  # 前面的大
        rate_1 = 1152.0 / Img_data.shape[0]
        rate_2 = 768.0 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_2
        Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        print(img_path)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

    height, width = Img_data.shape[0], Img_data.shape[1]

    m = int(width / 384)
    n = int(height / 384)
    fname = img_path.split('/')[-1]
    root_path = img_path.split('IMG_')[0].replace('images', 'images_crop')

    kpoint = kpoint.copy()
    if root_path.split('/')[-3] == 'train':

        for i in range(0, m):
            for j in range(0, n):
                crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384]
                gt_count = np.sum(crop_kpoint)

                save_fname = str(i) + str(j) + str('_') + fname
                save_path = root_path + save_fname

                h5_path = save_path.replace('.JPG', '.h5').replace('images', 'gt_density_map')
                if gt_count == 0:
                    print(save_path, h5_path)
                with h5py.File(h5_path, 'w') as hf:
                    hf['gt_count'] = gt_count

                cv2.imwrite(save_path, crop_img)

    else:
        img_path = img_path.replace('images', 'images_crop')

        cv2.imwrite(img_path, Img_data)

        gt_count = np.sum(kpoint)
        with h5py.File(img_path.replace('.JPG', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
            hf['gt_count'] = gt_count

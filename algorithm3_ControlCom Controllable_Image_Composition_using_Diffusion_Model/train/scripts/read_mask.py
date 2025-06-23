import os
import shutil
import cv2
import pandas as pd
import argparse
from regex import D
from tqdm import tqdm

os.chdir('/data')
# os.mkdir('dataset/open-images/bbox')
dir_list=os.listdir('dataset/open-images/images')
dir_list.sort()
cls_names = pd.read_csv('dataset/open-images/annotations/oidv7-class-descriptions-boxable.csv')
count = {}

for dir_name in ['train']:
    count[dir_name] = {'images':0, 'pairs':0}
    print(dir_name)
    if 'validation' in dir_name:
        csv_file_path = 'dataset/open-images/masks/validation-annotations-object-segmentation.csv'
    elif 'test' in dir_name:
        csv_file_path = 'dataset/open-images/masks/test-annotations-object-segmentation.csv'
    else:
        csv_file_path = 'dataset/open-images/masks/train-annotations-object-segmentation.csv'

    download_dir = os.path.join('dataset/open-images/images', dir_name)
    label_dir = os.path.join('dataset/open-images/bbox_mask', dir_name)
    os.makedirs(label_dir, exist_ok=True)

    downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
    images_label_list = list(set(downloaded_images_list))
    df_val = pd.read_csv(csv_file_path)
    groups = df_val.groupby(df_val.ImageID)
    for image in tqdm(images_label_list):
        file_name = str(image.split('.')[0]) + '.txt'
        file_path = os.path.join(label_dir, file_name)
        if os.path.exists(file_path):
            continue
        try:
            current_image_path = os.path.join(download_dir, image + '.jpg')
            dataset_image = cv2.imread(current_image_path)
            # print(image)
            img_info  = groups.get_group(image.split('.')[0])
            boxes     = img_info[['BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax']].values.tolist()
            ious      = img_info['PredictedIoU'].values.tolist()
            mask_path = img_info['MaskPath'].values.tolist()
            cls_id    = img_info['LabelName'].values.tolist()
    
            boxes_new=[]
            for i,box in enumerate(boxes):
                cls_name  = cls_names[cls_names['LabelName'] == cls_id[i]]['DisplayName'].values[0]
                bbox_area = (box[1]-box[0])*(box[3]-box[2])
                
                if dir_name == 'train' and ious[i] < 0.6:
                    iou = round(ious[i], 1)
                    subdir = f'composition/Ours/scripts/mask_w_differentious/{iou}'
                    os.makedirs(subdir, exist_ok=True)
                    if len(os.listdir(subdir)) < 50:
                        dst_path = os.path.join(subdir, image + '_' + cls_name + '.jpg')
                        src_path = os.path.join('dataset/open-images/masks', dir_name, mask_path[i])
                        shutil.copyfile(src_path, dst_path)
                
                if not(bbox_area > 0.8 or bbox_area < 0.02):
                    box[0] *= int(dataset_image.shape[1])
                    box[1] *= int(dataset_image.shape[1])
                    box[2] *= int(dataset_image.shape[0])
                    box[3] *= int(dataset_image.shape[0])
                    boxes_new.append([box[0],box[1],box[2],box[3],cls_name,mask_path[i],ious[i]])
                
            
            if len(boxes_new)>0:
                count[dir_name]['images'] += 1
                count[dir_name]['pairs']  += len(boxes_new)
                # print(file_path)
                if os.path.isfile(file_path):
                    f = open(file_path, 'a')
                else:
                    f = open(file_path, 'w')

                for box in boxes_new:
                        # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
                    print(box[0], box[2], box[1], box[3], box[4], box[5], box[6], file=f)
        except Exception as e:
            pass
    print(count)
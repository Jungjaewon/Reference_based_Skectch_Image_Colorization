import os.path as osp
import os
import glob
import cv2

from tqdm import tqdm
if __name__ == '__main__':

    for mode in ['train', 'val']:

        image_list = glob.glob(osp.join('data', mode , '*.png'))

        for image_path in tqdm(image_list):
            fid = image_path.split(os.sep)[-1].split('.')[0]
            img = cv2.imread(image_path)
            color = img[:,:512,:]
            sket = img[:,512:,:]
            cv2.imwrite(osp.join('data', mode + '_new', '{}_{}.png'.format(fid, 'color')), color)
            cv2.imwrite(osp.join('data', mode + '_new', '{}_{}.png'.format(fid, 'sketch')), sket)
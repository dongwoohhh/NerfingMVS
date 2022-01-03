

import os, sys
import glob
import argparse
import imageio
import cv2
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str,)
#parser.add_argument('--index_order', type=int,)
args = parser.parse_args()

datadir = args.datadir

height_resized = 480

new_scene_dir = os.path.join(datadir, 'images_resize')
if not os.path.exists(new_scene_dir):
    os.mkdir(new_scene_dir)
all_images = [x for x in glob.glob(os.path.join(datadir, "images/*")) if os.path.isfile(x)]
all_images.sort()
#print(all_images)

for image_name in all_images:
    image = imageio.imread(image_name)
    height, width = image.shape[:2]
    ratio = width / height
    width_resized = height_resized*ratio
    if width_resized % 16 != 0:
        width_resized = width_resized - width_resized % 16
    
    new_size = np.array([height_resized, width_resized], dtype=np.int)
    new_size = new_size.astype(np.int)
    #print(new_size)

    image_r = cv2.resize(image, (new_size[1], new_size[0]))
    
    new_image_name = image_name.replace('images', 'images_resize')
    new_image_name = new_image_name.replace('jpg', 'png')
    imageio.imwrite(new_image_name, image_r)
    
    #image_r = imageio.imresize

os.system("mv {}/images {}/images_original".format(datadir, datadir))
os.system("mv {}/images_resize {}/images".format(datadir, datadir))

#print(all_images)

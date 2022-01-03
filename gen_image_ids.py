

import os, sys
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str,)
#parser.add_argument('--index_order', type=int,)
args = parser.parse_args()

datadir = args.datadir


all_images = [x for x in glob.glob(os.path.join(datadir, "images/*")) if os.path.isfile(x)]
all_images.sort()
#print(all_images)

file = open(os.path.join(datadir, 'train.txt'), 'w')


for i, name in enumerate(all_images):
    name = name.split('/')[-1]
    if i < len(all_images) - 1:
        file.write(name+'\n')
    else:
        file.write(name)
file.close()

file = open(os.path.join(datadir, 'test.txt'), 'w')
file.close()


#print(all_images)

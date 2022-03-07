

import os, sys
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str,)
#parser.add_argument('--index_order', type=int,)
args = parser.parse_args()

datadir = args.datadir

all_dirs = [x for x in glob.glob(os.path.join(datadir,"*")) if os.path.isdir(x)]
all_dirs.sort()

print(all_dirs)
file = open(os.path.join(datadir, 'new_train.lst'), 'w')


for i, name in enumerate(all_dirs):
    name = name.split('/')[-1]
    if i < len(all_dirs) - 1:
        file.write(name+'\n')
    else:
        file.write(name)
file.close()



#print(all_images)

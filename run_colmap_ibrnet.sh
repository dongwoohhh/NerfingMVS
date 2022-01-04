datadir=/media/hdd1/Datasets/ibrnet_NerfingMVS/data/ibrnet_collected_1

echo $datadir

python gen_scene_name.py --datadir datadir
for name in ${datadir}/* ; do
    echo $name
    echo ${name##*/}
    #rm -rf ${name}/depth ${name}/dense ${name}/sparse ${name}/poses_bounds.npy
    #rm ${name}/images/train.txt ${name}/images/test.txt
    #python resize_image.py --datadir=${name}
    #python gen_image_ids.py --datadir=${name}
    #rm -rf ${name}/images_1
    #sh colmap.sh ${name}
    python run_depth_priors.py --config configs/per_scene.txt --datadir=${name} --no_ndc --spherify --lindisp --expname=${name##*/}

done

datadir=/media/hdd1/Datasets/ibrnet_NerfingMVS/data/ibrnet_collected_2

echo $datadir

python gen_scene_name.py --datadir datadir
for name in ${datadir}/* ; do
    echo $name
    echo ${name##*/}
    
    #rm -rf ${name}/depth ${name}/dense ${name}/sparse ${name}/poses_bounds.npy
    #rm ${name}/images/train.txt ${name}/images/test.txt
    #python resize_image.py --datadir=${name}
    #python gen_image_ids.py --datadir=${name}
    #rm -rf ${name}/images_1
    #sh colmap.sh ${name}
    
    python run_depth_priors.py --config configs/per_scene.txt --datadir=${name} --no_ndc --spherify --lindisp --expname=${name##*/}

done

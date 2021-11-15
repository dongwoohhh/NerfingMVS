#SCENE=$1
#sh colmap.sh data/$SCENE

datadir=/media/hdd1/Datasets/ibrnet_NerfingMVS/data/ibrnet_collected_1
#outdir=/media/hdd1/Datasets/DTU_NerfingMVS
resultdir=/media/hdd1/results_nerf/NerfingMVS/per_scene
echo $datadir


for name in ${datadir}/* ; do
    echo ${name##*/}


    #mkdir $datadir/scan${i}_train/images
    #cp $datadir/scan${i}_train/* $datadir/scan${i}_train/images/
    #mkdir /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train
    #mv $datadir/scan${i}_train/images /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/
    #cp /home/dongwool/Projects/NeRFs/NerfingMVS/data/dtu_scene21/train.txt /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/
    #rm /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/images/rect_*_0_r5000.png
    #rm /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/images/rect_*_1_r5000.png
    #rm /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/images/rect_*_2_r5000.png
    #rm /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/images/rect_*_4_r5000.png
    #rm /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/images/rect_*_5_r5000.png
    #rm /media/hdd1/Datasets/DTU_NerfingMVS/scan${i}_train/images/rect_*_6_r5000.png
    #echo ${outdir}/scan${i}_train
    #sh colmap.sh ${outdir}/scan${i}_train
    python run_depth_priors.py --config configs/per_scene.txt --datadir=${name} --no_ndc --spherify --lindisp --expname=${name##*/}
done

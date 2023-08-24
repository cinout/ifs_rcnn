#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5


#SBATCH --job-name="debsurg"
#SBATCH --account=punim1623
#SBATCH --time=0-00:20:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=50G

module purge

eval "$(conda shell.bash hook)"
conda activate fct

dataset=coco
network=mask_rcnn
arch=50
suffix='' # TODO: which ?

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m tools.ckpt_surgery --$dataset \
        --src1 checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_base${suffix}/model_final.pth \
        --method remove \
        --save-dir checkpoints/${dataset}/${network}/${network}_R_${arch}_FPN_all${suffix}

##Log this job's resource usage stats###
my-job-stats -a -n -s
##
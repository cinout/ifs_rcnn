#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5


#SBATCH --job-name="debtrain"
#SBATCH --account=punim1623
#SBATCH --time=0-00:20:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --cpus-per-task=1
#SBATCH --mem=50G

module purge

eval "$(conda shell.bash hook)"
conda activate fsdet

dataset=coco
detection_folder=${dataset^^}-detection
network=mask_rcnn
arch=50
suffix='_sigmoid_classifier_box_iou_uncertainty'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m tools.train_net --num-gpus 1 --config-file configs/${detection_folder}/${network}/${network}_R_${arch}_FPN_base${suffix}.yaml

##Log this job's resource usage stats###
my-job-stats -a -n -s
##
#!/bin/bash

export PROJECTPATH=$1
export PROJECTNAME=$2
export MODELPATH=$3



rm -rf /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH/$PROJECTNAME # TODO: parametrize
mkdir -p /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH/$PROJECTNAME
cd tools/TrainLib_Deployer
if [ -z "$MODELPATH" ]
then
	python TrainLib_Deployer.py --project_path /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH --project_name $PROJECTNAME
else
	python TrainLib_Deployer.py --project_path /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH --project_name $PROJECTNAME --model_path $MODELPATH
fi
cd /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH/$PROJECTNAME/utils/
python GM.py
cp init-defines.h ../
cp io_data.c ../
cp io_data.h ../

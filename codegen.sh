#!/bin/bash

export PATH=/usr/pack/gcc-9.2.0-af/linux-x64/bin:$PATH 
export LD_LIBRARY_PATH=/usr/pack/gcc-9.2.0-af/linux-x64/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/scratch/wetterhorn/cioflanc/miniconda3/pkgs/mpfr-4.0.2-hb69a4c5_1/lib/

export PROJECTPATH=$1
export PROJECTNAME=$2
export MODELPATH=$3



rm -rf /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH/$PROJECTNAME # TODO: parametrize
mkdir -p /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH/$PROJECTNAME
cp resources/* /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/$PROJECTPATH/$PROJECTNAME
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
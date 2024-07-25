#!/bin/bash

export PATH=/usr/pack/gcc-9.2.0-af/linux-x64/bin:$PATH 
export LD_LIBRARY_PATH=/usr/pack/gcc-9.2.0-af/linux-x64/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/scratch/wetterhorn/cioflanc/miniconda3/pkgs/mpfr-4.0.2-hb69a4c5_1/lib/

export PROJECTPATH=$1
export PROJECTNAME=$2
export MODELPATH=$3
export START=$4

export CUR_DIR=$PWD
export BASE_DIR=$CUR_DIR/../


rm -rf $BASE_DIR/$PROJECTPATH/$PROJECTNAME # TODO: parametrize

cd tools/TrainLib_Deployer
if [ -z "$MODELPATH" ]
then
	python TrainLib_Deployer.py --project_path $BASE_DIR/$PROJECTPATH --project_name $PROJECTNAME
else
	python TrainLib_Deployer.py --project_path $BASE_DIR/$PROJECTPATH --project_name $PROJECTNAME --model_path $BASE_DIR/$MODELPATH --start_at $START
fi
cp ../../resources/* $BASE_DIR/$PROJECTPATH/$PROJECTNAME
cd $BASE_DIR/$PROJECTPATH/$PROJECTNAME/utils/
python GM.py
cp init-defines.h ../
cp io_data.h ../
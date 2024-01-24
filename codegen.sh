#!/bin/bash

export PROJECTPATH=$1
export PROJECTNAME=$2



rm -rf $PROJECTPATH/$PROJECTNAME # TODO: parametrize
mkdir $PROJECTPATH
cd tools/TrainLib_Deployer
python TrainLib_Deployer.py --project_path $PROJECTPATH --project_name $PROJECTNAME 
cd $PROJECTPATH/$PROJECTNAME/utils/
python GM.py
cp initdefines.h ../
cp iodata.c ../
cp iodata.h ../

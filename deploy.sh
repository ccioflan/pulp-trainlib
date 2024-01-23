#!/bin/bash

# Copyright (C) 2021-2023 ETH Zurich

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)


# Set up constants

if [ "$1" == "-h" ] ; then
    echo "SDK: pulp_sdk, gap_sdk"
    echo "MEMORY: (L)2, (L)3"
    echo "PLATFORM: gvsoc, fpga, rtl"
    echo "MFCC computation: 0 (offline), 1 (online)"
    exit 0
fi

export SDK=$1
export MEMORY=$2
export PLATFORM=$3
export MFCC=$4

# export PATH=/usr/pack/gcc-4.9.1-af/x86_64-rhe6-linux/bin:$PATH
# export LD_LIBRARY_PATH=/usr/pack/gcc-4.9.1-af/x86_64-rhe6-linux/lib64/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/pack/gcc-4.9.1-af/x86_64-rhe6-linux/lib/:$LD_LIBRARY_PATH

# export PATH=/usr/pack/gcc-9.2.0-af/x86_64-rhe6-linux/bin:$PATH
# export LD_LIBRARY_PATH=/usr/pack/gcc-9.2.0-af/x86_64-rhe6-linux/lib64/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/pack/gcc-9.2.0-af/x86_64-rhe6-linux/lib/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/scratch/wetterhorn/cioflanc/miniconda3/pkgs/mpfr-4.0.2-hb69a4c5_1/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/scratch/wetterhorn/cioflanc/teaching/classes/mlonmcu/mlonmcu_exercise6/exercise6/local_libs/

export CC=gcc-9.2.1
export CXX=g++-9.2.1

if [[ $SDK == "pulp_sdk" ]]
then
  export PULP_RISCV_GCC_TOOLCHAIN=/usr/scratch/wetterhorn/cioflanc/tools/pulp_riscv_toolchain/v1.0.16-pulp-riscv-gcc-centos-7/
  # Select target
  if [[ $PLATFORM == "gvsoc" ]]
  then
    source /usr/scratch/wetterhorn/cioflanc/tools/pulp-sdk/configs/pulp-open.sh # PULP-SDK
    # Error
    # Launching training procedure...
    # 9560452886: 636551: [/sys/board/chip/cluster/pe0/warning                 ] Invalid access (pc: 0x1c00b012, offset: 0x57575757, size: 0x1, is_write: 0)

  elif [[ $PLATFORM == "fpga" ]]
  then
    source /usr/scratch/wetterhorn/cioflanc/tools/pulp_sdk_fpga/pulp-sdk/configs/pulp-open.sh
  elif [[ $PLATFORM == "rtl" ]]
  then
    source /usr/scratch/wetterhorn/cioflanc/tools/pulp_sdk_fpga/pulp-sdk/configs/pulp-open.sh
  fi
else
  export GAP_RISCV_GCC_TOOLCHAIN=/usr/scratch/wetterhorn/cioflanc/tools/gap_riscv_toolchain/
  # Select target
  
  # source /usr/scratch/wetterhorn/cioflanc/tools/gap_sdk/sourceme.sh # GAP8 - GAPUINO
  # Error
  # gvsoc_launcher_debug: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /usr/scratch/wetterhorn/cioflanc/tools/gap_sdk/install/workstation/lib/libpulpvp-debug.so)
  
  # source /usr/scratch/wetterhorn/cioflanc/tools/gap_sdk_mar23/gap_sdk/sourceme.sh # GAP8 - GAPOC_B_V2
  # Error:
  # Launching training procedure...
  # 6023176965: 400609: [/sys/board/chip/cluster/pe0/warning                              ] Invalid access (pc: 0x1c00399e, offset: 0x57575757, size: 0x1, is_write: 0)

  source /usr/scratch/wetterhorn/cioflanc/tools/gap_sdk_private/sourceme.sh # GAP9
  # Error:
  # 2392121649: 72001: [/chip/cluster/pe8/warning                                            ] Invalid access (pc: 0x1c014844, offset: 0x57575757, size: 0x1, is_write: 0)
  # make: *** [/usr/scratch/wetterhorn/cioflanc/tools/gap_sdk_private/rtos/pmsis/os/freeRTOS/vendors/gwt/rules/freeRTOS_rules.mk:491: run] Error 1

fi

cd /usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/trainlib_example_dscnn/DSCNNS/ # TODO: parametrize
python utils/GM.py

make clean get_golden all run platform=$PLATFORM


make clean all run platform=$PLATFORM CORE=8 # sample=$AUDIO_SAMPLE sdk=$SDK memory=$MEMORY platform=$PLATFORM mfcc=$MFCC CORE=8


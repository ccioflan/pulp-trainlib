'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini
'''

"""
TrainLib Deployer: a tool to deploy DNN on-device training on MCUs

Available DNN layer names:
'linear'    -> fully-connected layer
'conv2d'    -> 2d convolution layer
'PW'        -> pointwise convolution
'DW'        -> depthwise convolution
'ReLU'      -> ReLU activation
'MaxPool'   -> max pooling layer
'AvgPool'   -> average pooling layer
'Skipnode'  -> node at which data is taken and passes forward, to add an additional layer after the skip derivation simply substitute 'Skipnode' with any kind of layer
'Sumnode'   -> node at which data from Skipnode is summed 
'InstNorm'  -> instance Normalization layer

Available losses:
'MSELoss'           -> Mean Square Error loss
'CrossEntropyLoss'  -> CrossEntropy loss

Available optimizers:
'SGD'       -> Stochastic Gradient Descent
"""

import deployer_utils.DNN_Reader     as reader
import deployer_utils.DNN_Composer   as composer

import os
import argparse
import onnx
from onnx import shape_inference, numpy_helper
import numpy as np

# ---------------------
# --- USER SETTINGS ---
# ---------------------


parser = argparse.ArgumentParser(
                    prog='Deployer',
                    description='Generating C code for on-device training')

parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--project_name', type=str, default="examplenet/", help='Project name', required=True)
required.add_argument('--project_path', type=str, default="/usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/trainlib_example_dscnn/", help='Project path', required=True)
optional.add_argument('--model_path', type=str, default=None, help='Pretrained model path')
args = parser.parse_args()

# GENERAL PROPERTIES
project_name    = args.project_name
project_path    = args.project_path
proj_folder     = project_path + project_name + '/'

# TRAINING PROPERTIES
epochs          = 10
batch_size      = 1                   # BATCHING NOT IMPLEMENTED!!
learning_rate   = 0.01
optimizer       = "SGD"                # Name of PyTorch's optimizer
loss_fn         = "CrossEntropyLoss"            # Name of PyTorch's loss function

# # EXAMPLE
# # ------- NETWORK GRAPH --------
# # Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
# layer_list          = [ 'DW', 'PW', 'ReLU', 'DW', 'PW', 'ReLU', 'DW', 'PW', 'ReLU', 'DW', 'PW', 'ReLU', 'linear']
# # Layer properties
# sumnode_connections = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ]            # For Skipnode and Sumnode only, for each Skipnode-Sumnode couple choose a value and assign it to both, all other layer MUST HAVE 0

# in_ch_list          = [ 3,  3,  4,  4,  4,  8,  8,  8, 16, 16, 16, 24,  1536 ]         # Linear: size of input vector
# out_ch_list         = [ 3,  4,  4,  4,  8,  8,  8, 16, 16, 16, 24, 24,  2 ]            # Linear: size of output vector
# hk_list             = [ 9,  1,  1,  7,  1,  1,  3,  1,  1,  9,  1,  1,  1 ]            # Linear: = 1
# wk_list             = [ 9,  1,  1,  7,  1,  1,  3,  1,  1,  9,  1,  1,  1 ]            # Linear: = 1
# # Input activations' properties
# hin_list            = [ 32, 24, 24, 24, 18, 18, 18, 16, 16, 16, 8,  8, 1 ]            # Linear: = 1
# win_list            = [ 32, 24, 24, 24, 18, 18, 18, 16, 16, 16, 8,  8, 1 ]            # Linear: = 1
# # Convolutional strides
# h_str_list          = [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ]            # Only for conv2d, maxpool, avgpool (NOT IMPLEMENTED FOR CONV2D)
# w_str_list          = [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ]            # Only for conv2d, maxpool, avgpool (NOT IMPLEMENTED FOR CONV2D)
# # Padding (bilateral, adds the specified padding to both image sides)
# h_pad_list          = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]                            # Only for conv2d, DW (NOT IMPLEMENTED)
# w_pad_list          = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]                            # Only for conv2d, DW (NOT IMPLEMENTED)
# # Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
# opt_mm_fw_list      = [ 1, 12, 12, 12, 12, 12, 12, 12, 12, 1, 12, 1, 10 ]
# opt_mm_wg_list      = [ 1, 12, 12, 12, 12, 12, 12, 12, 12, 1, 12, 1, 10 ]
# opt_mm_ig_list      = [ 1, 12, 12, 12, 12, 12, 12, 12, 12, 1, 12, 1, 10 ]
# # Data type list for layer-by-layer deployment (mixed precision)
# data_type_list      = ['FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16']
# #data_type_list     = ['FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32']
# # Data layout list (CHW or HWC) 
# data_layout_list    = ['CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW']   # TO DO
# # ----- END OF NETWORK GRAPH -----

# ------- DS-CNN NETWORK GRAPH --------
layer_list = ['linear']
# Layer properties
sumnode_connections = [ 0]
in_ch_list = [172]                # Linear: size of input vector
out_ch_list = [12]                # Linear: size of input vector
hk_list = [1]                     # Linear: = 1
wk_list = [1]                     # Linear: = 1

# Input activations' properties
hin_list        = [1]             # Linear: = 1
win_list        = [1]             # Linear: = 1

# Convolutional strides
h_str_list      = [1]             ## Only for conv2d, maxpool, avgpool (NOT IMPLEMENTED FOR CONV2D)
w_str_list      = [1]             ## Only for conv2d, maxpool, avgpool (NOT IMPLEMENTED FOR CONV2D)

# Padding (bilateral, adds the specified padding to both image sides)
h_pad_list      = [0 ]             # Only for conv2d, DW
w_pad_list      = [0 ]             # Only for conv2d, DW

# # TODO Understand
# Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
opt_mm_fw_list  = [0 ]
opt_mm_wg_list  = [0 ]
opt_mm_ig_list  = [0 ]

# Data type list for layer-by-layer deployment (mixed precision)
data_type_list   = ['FP32']
# Data layout list (CHW or HWC) 
data_layout_list = ['CHW']   # TO DO
# Pretrained parameters
data_list = []
# ----- END OF DS-CNN NETWORK GRAPH -----

# EXECUTION PROPERTIES
NUM_CORES       = 1
L1_SIZE_BYTES   = 60*(2**10)
USE_DMA = 'NO'                          # choose whether to load all structures in L1 ('NO') or in L2 and use Single Buffer mode ('SB') or Double Buffer mode ('DB') 
# BACKWARD SETTINGS
SEPARATE_BACKWARD_STEPS = False          # If True, writes separate weight and input gradient in backward step
# PROFILING OPTIONS
PROFILE_SINGLE_LAYERS = False           # If True, profiles forward and backward layer-by-layer

# OTHER PROPERTIES
# Select if to read the network from an external source
READ_MODEL_ARCH = args.model_path

# ---------------------------
# --- END OF USER SETTING ---
# ---------------------------


"""
BACKEND
"""

# Call the DNN Reader and then the DNN Composer 
if READ_MODEL_ARCH :

    layer_list = []
    in_ch_list = []
    out_ch_list = []
    hk_list = []
    wk_list = []
    hin_list = []
    win_list = []
    h_pad_list = []
    w_pad_list = []
    opt_mm_fw_list = []
    opt_mm_wg_list = []
    opt_mm_ig_list = []
    data_type_list = []
    data_layout_list = []
    
    if (args.model_path.split('.')[-1] == "onnx"):
        onnx_model = onnx.load(args.model_path)
        onnx.checker.check_model(onnx_model)
        onnx_graph = onnx_model.graph
        m_onnx_graph = shape_inference.infer_shapes(onnx_model)
        graph_offset = int(m_onnx_graph.graph.value_info[0].name)
        graph_len = len(m_onnx_graph.graph.value_info)
        
        for onnx_node in m_onnx_graph.graph.node:

            if (onnx_node.op_type == 'Gemm') or (onnx_node.op_type == 'MatMul'):
                in_ch_list.append(m_onnx_graph.graph.value_info[int(onnx_node.input[0])-graph_offset].type.tensor_type.shape.dim[1].dim_value) # ch x w
                if (int(onnx_node.input[0])-graph_offset == graph_len-1):
                    out_ch_list.append(m_onnx_graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value) # ch x w
                else:
                    out_ch_list.append(m_onnx_graph.graph.value_info[int(onnx_node.output[0])-graph_offset].type.tensor_type.shape.dim[1].dim_value) # ch x w
                layer_list.append('linear') 
                hk_list.append(1)
                wk_list.append(1)
                hin_list.append(1)
                win_list.append(1)
                h_pad_list.append(0)
                w_pad_list.append(0)
                opt_mm_fw_list.append(0)
                opt_mm_wg_list.append(0)
                opt_mm_ig_list.append(0)
                # TODO: Read from file
                data_type_list.append('FP32')
                # TODO: Read from file
                # Note that this also determines the read position for in_ch_list and out_ch_list
                data_layout_list.append('CHW')

                for init in onnx_graph.initializer:
                    if init.name == onnx_node.input[1]: # weights
                        data_list.append(numpy_helper.to_array(init))
                    # if init.name == onnx_node.input[2]: # bias
                        # TODO: Add bias
    else:
        raise NotImplementedError("Model format not supported.")

    data_dir = proj_folder+'data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for layer in layer_list:
        np.save(data_dir+"l"+str(layer_list.index(layer))+"w.npy", data_list[layer_list.index(layer)])

    print("Generating project at location "+proj_folder)

    # Check if Residual Connections are valid
    sumnode_connections = composer.AdjustResConnList(sumnode_connections)

    composer.CheckResConn(layer_list, in_ch_list, out_ch_list, hin_list, win_list, sumnode_connections) 

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, L1_SIZE_BYTES, USE_DMA)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list, sumnode_connections,
                            USE_DMA, PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, data_list)

    print("PULP project generation successful!")


else:

    print("Generating project at location "+proj_folder)

    # Check if Residual Connections are valid
    sumnode_connections = composer.AdjustResConnList(sumnode_connections)

    composer.CheckResConn(layer_list, in_ch_list, out_ch_list, hin_list, win_list, sumnode_connections) 

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, L1_SIZE_BYTES, USE_DMA)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list, sumnode_connections, 
                            USE_DMA, PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, data_list)

    print("PULP project generation successful!")

    pass

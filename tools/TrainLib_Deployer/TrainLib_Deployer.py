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
"""

import utils.DNN_Reader     as reader
import utils.DNN_Composer   as composer

import argparse
import onnx
from onnx import shape_inference

# ---------------------
# --- USER SETTINGS ---
# ---------------------


parser = argparse.ArgumentParser(
                    prog='Deployer',
                    description='Generating C code for on-device training')

parser.add_argument('--model_path', type=str, default="/usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/kws-on-pulp/application_dscnnl_gap8/model.onnx")
parser.add_argument('--project_name', type=str, default="examplenet")
parser.add_argument('--project_path', type=str, default="/usr/scratch/wetterhorn/cioflanc/kws_on_gap9/tiny_denoiser/trainlib_example_dscnn/")
args = parser.parse_args()

# GENERAL PROPERTIES
project_name    = args.project_name
project_path    = args.project_path
proj_folder     = project_path + project_name + '/'

# TRAINING PROPERTIES
epochs          = 1
batch_size      = 1                    # BATCHING NOT IMPLEMENTED!!
learning_rate   = 0.5
optimizer       = "SGD"                # Name of PyTorch's optimizer
loss_fn         = "MSELoss"            # Name of PyTorch's loss function

# # Example
# # ------- NETWORK GRAPH --------
# # Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
# layer_list      = [ 'conv2d', 'ReLU', 'DW', 'PW', 'ReLU', 'linear' ]
# # Layer properties
# in_ch_list      = [ 1, 4, 4, 4, 4, 4*2*2 ]          # Linear: size of input vector
# out_ch_list     = [ 4, 4, 4, 4, 4, 2 ]            # Linear: size of output vector
# hk_list         = [ 3, 1, 3, 1, 1, 1 ]             # Linear: = 1
# wk_list         = [ 3, 1, 3, 1, 1, 1 ]             # Linear: = 1
# # Input activations' properties
# hin_list        = [ 6, 4, 4, 2, 2, 1 ]             # Linear: = 1
# win_list        = [ 6, 4, 4, 2, 2, 1 ]             # Linear: = 1
# # Convolutional strides
# h_str_list      = [ 1, 1, 1, 1, 1, 1 ]             # Only for conv2d, maxpool, avgpool
# w_str_list      = [ 1, 1, 1, 1, 1, 1 ]             # Only for conv2d, maxpool, avgpool
# # Padding (bilateral, adds the specified padding to both image sides)
# h_pad_list      = [ 0, 0, 0, 0, 0, 0 ]             # Only for conv2d, DW
# w_pad_list      = [ 0, 0, 0, 0, 0, 0 ]             # Only for conv2d, DW
# # Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
# opt_mm_fw_list  = [ 0, 0, 0, 0, 0, 0 ]
# opt_mm_wg_list  = [ 0, 0, 0, 0, 0, 0 ]
# opt_mm_ig_list  = [ 0, 0, 0, 0, 0, 0 ]
# Data type list for layer-by-layer deployment (mixed precision)
# data_type_list   = ['FP32', 'FP32', 'FP32', 'FP16', 'FP16', 'FP16']
# Data layout list (CHW or HWC) 
# data_layout_list = ['CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW']   # TO DO
# # ----- END OF NETWORK GRAPH -----


# ------- DS-CNN NETWORK GRAPH --------
layer_list = ['linear']
# Layer properties
in_ch_list = [276]
out_ch_list = [12]
hk_list = [1]
wk_list = [1]

# # TODO Understand
# Input activations' properties
hin_list        = [1]             # Linear: = 1
win_list        = [1]             # Linear: = 1

# Convolutional strides
h_str_list      = [1]             # Only for conv2d, maxpool, avgpool
w_str_list      = [1]             # Only for conv2d, maxpool, avgpool

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
# ----- END OF DS-CNN NETWORK GRAPH -----

# EXECUTION PROPERTIES
NUM_CORES       = 8
L1_SIZE_BYTES   = 120*(2**10)

# OTHER PROPERTIES
# Select if to read the network from an external source
READ_MODEL_ARCH = True                # NOT IMPLEMENTED!!

# ---------------------------
# --- END OF USER SETTING ---
# ---------------------------


"""
BACKEND
"""

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


# Call the DNN Reader and then the DNN Composer 
if READ_MODEL_ARCH :

    # # Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
    # layer_list      = [ 'conv2d', 'ReLU', 'DW', 'PW', 'ReLU', 'linear' ]
    # # Layer properties
    # in_ch_list      = [ 1, 4, 4, 4, 4, 4*2*2 ]          # Linear: size of input vector
    # out_ch_list     = [ 4, 4, 4, 4, 4, 2 ]            # Linear: size of output vector
    # hk_list         = [ 3, 1, 3, 1, 1, 1 ]             # Linear: = 1
    # wk_list         = [ 3, 1, 3, 1, 1, 1 ]             # Linear: = 1
    # # Input activations' properties
    # hin_list        = [ 6, 4, 4, 2, 2, 1 ]             # Linear: = 1
    # win_list        = [ 6, 4, 4, 2, 2, 1 ]             # Linear: = 1
    # # Convolutional strides
    # h_str_list      = [ 1, 1, 1, 1, 1, 1 ]             # Only for conv2d, maxpool, avgpool
    # w_str_list      = [ 1, 1, 1, 1, 1, 1 ]             # Only for conv2d, maxpool, avgpool
    # # Padding (bilateral, adds the specified padding to both image sides)
    # h_pad_list      = [ 0, 0, 0, 0, 0, 0 ]             # Only for conv2d, DW
    # w_pad_list      = [ 0, 0, 0, 0, 0, 0 ]             # Only for conv2d, DW
    # # Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
    # opt_mm_fw_list  = [ 0, 0, 0, 0, 0, 0 ]
    # opt_mm_wg_list  = [ 0, 0, 0, 0, 0, 0 ]
    # opt_mm_ig_list  = [ 0, 0, 0, 0, 0, 0 ]
    # Data type list for layer-by-layer deployment (mixed precision)
    # data_type_list   = ['FP32', 'FP32', 'FP32', 'FP16', 'FP16', 'FP16']
    # Data layout list (CHW or HWC) 
    # data_layout_list = ['CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW']   # TO DO

    onnx_model = onnx.load(args.model_path)
    onnx.checker.check_model(onnx_model)
    onnx_graph = onnx_model.graph
    m_onnx_graph = shape_inference.infer_shapes(onnx_model)
    graph_offset = int(m_onnx_graph.graph.value_info[0].name)
    graph_len = len(m_onnx_graph.graph.value_info)

    print (onnx_graph.initializer[3])
    # print (graph_offset)
    # print (graph_len)

    # print (m_onnx_graph.graph.value_info[0].type.tensor_type.shape.dim) # returns the activation sizes
    # print (m_onnx_graph.graph.output[0].type.tensor_type.shape.dim) # returns the activation sizes
    # print (m_onnx_graph.graph.input[0].type.tensor_type.shape.dim) # returns the activation sizes

    
    for onnx_node in m_onnx_graph.graph.node:

        print (onnx_node)
        print (onnx_node.attribute)

        print ("-------------------------")
        if (onnx_node.op_type == 'Gemm'):
            in_ch_list.append(m_onnx_graph.graph.value_info[int(onnx_node.input[0])-graph_offset].type.tensor_type.shape.dim[0].dim_value)
            if (int(onnx_node.input[0])-graph_offset == graph_len-1):
                out_ch_list.append(m_onnx_graph.graph.output[0].type.tensor_type.shape.dim[0].dim_value)
            else:
                out_ch_list.append(m_onnx_graph.graph.value_info[int(onnx_node.output[0])-graph_offset].type.tensor_type.shape.dim[0].dim_value)
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
                if init.name == onnx_node.input[1]:
                    # print(init.raw_data)
                    # TODO: Save data in data_type_list format
                if init.name == onnx_node.input[2]:
                    # print(init.raw_data)
                    # TODO: Save data in data_type_list format
            

            
    # for attribute in node_iterating.attribute:
    #         if attribute.name not in ['kernel_shape', 'dilations', 'group', 'strides', 'pads'] or self.name == "Pad":
    #             if bool(attribute.i):
    #                 self.__dict__[attribute.name] = int(attribute.i)
    #             elif bool(attribute.f):
    #                 self.__dict__[attribute.name] = int(attribute.f)
    #             elif bool(attribute.ints):
    #                 self.__dict__[attribute.name] = list(attribute.ints)
    #             elif attribute.i == 0:
    #                 self.__dict__[attribute.name] = 0
    #             else:
    #                 sys.exit("DORY FRONTEND error. DORY does not find any values for the attribute {}".format(attribute.name))
    # graph inputs
    # for input_name in onnx_graph.input:
    #     print(input_name)
    # # graph parameters
    # for init in onnx_graph.initializer:
    #     print(init.name)
    #     print(len(init))
    # # graph outputs
    # for output_name in onnx_graph.output:
    #     print(output_name)
    # # iterate over nodes
    # for node in onnx_graph.node:
    #     # node inputs
    #     for idx, node_input_name in enumerate(node.input):
    #         print(idx, node_input_name)
    #     # node outputs
    #     for idx, node_output_name in enumerate(node.output):
    #         print(idx, node_output_name)

    # print ("-------")
    # print (len(onnx_graph.initializer[0].raw_data))


    # for node_iterating in (self.graph.graph.node):
    #         ### check if the node is supported
    #         assert (node_iterating.op_type in self.layers_accepted), f"{node_iterating.op_type} not supported by DORY"
    #         ### Neglecting some nodes since they are not translated to any operation on any backend
    #         if node_iterating.op_type in self.layers_neglected:
    #             for node in self.DORY_Graph[::-1]:
    #                 if int(node_iterating.output[0]) > int(node.get_parameter('output_index')) and node.get_parameter("name") != "Constant":
    #                     node.add_existing_parameter('output_index', node_iterating.output[0]) 
    #                     break
    #         # Adding a new layer
    #         elif node_iterating.op_type in self.layers_accepted:
    #             new_node = self.create_node(node_iterating, self.graph)
    #             self.DORY_Graph.append(new_node)
    #         else:
    #             sys.exit("DORY Frontend. Node not parsed.")


    print("Automatically generating project at location "+proj_folder)

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, L1_SIZE_BYTES)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list)

    print("PULP project generation successful!")


else:

    print("Generating project at location "+proj_folder)

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, L1_SIZE_BYTES)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list)

    print("PULP project generation successful!")

    pass
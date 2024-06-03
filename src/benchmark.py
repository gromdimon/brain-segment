# This script requires having compiled 2 different IR models named as follows
# model.bin, model.xml (as the benchmark)
# opt_model.bin, opt_model.xml (as the optimized version)

import openvino as ov
import torch
import numpy as np
import nibabel as nib

import time

core = ov.Core()

devices = core.available_devices

for device in devices:
    device_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

opt_ir_model_name_xml = 'src/onnx/opt_model.xml'
opt_ir_model_bin = './onnx/opt_model.bin'
base_ir_name_xml = 'src/onnx/model.xml'
opt_ir_model_bin = './onnx/model.bin'

opt_model = core.read_model(model=opt_ir_model_name_xml)
compiled_opt_model = core.compile_model(model=opt_model, device_name="CPU")

base_model = core.read_model(model=base_ir_name_xml)
compiled_base_model = core.compile_model(model=base_model, device_name="CPU")

print(f"Input type: {opt_model.inputs}")

input_layer_opt = compiled_opt_model.input(0)
output_layer_opt = compiled_opt_model.output(0)

input_layer_base = compiled_base_model.input(0)
output_layer_base = compiled_base_model.output(0)



# Inference
# Here we've chosen a random example for time savings, ideally we'd want to loop through the whole dataset to get proper data
image_path = "src/data/Task01_BrainTumour/imagesTr/BRATS_035.nii.gz"
image = nib.load(image_path)

print(f"Type of image: {type(image)}")

# Convert to numpy array
input_data = image.get_fdata()

# Expand dimensions and transpose to fit input type

input_data = np.expand_dims(input_data, axis=0)

input_data = input_data.transpose(0, 4, 1, 2, 3)

print(f"Input reshaped to: {input_data.shape}")

start_time = time.time()
result = compiled_opt_model([input_data])[output_layer_opt]
end_time = time.time()
inference_time_opt = end_time - start_time

start_time = time.time()
result = compiled_base_model([input_data])[output_layer_base]
end_time = time.time()
inference_time_base = end_time - start_time


print(f"Total inference time of optimized model: {inference_time_opt}s")
print(f"Total inference time of base model: {inference_time_base}s")
print(f"Difference: {inference_time_base - inference_time_opt}s")
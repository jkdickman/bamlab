# tf_to_tflite.py
import tensorflow as tf
import os
import numpy as np
from PIL import Image

# Path to the TensorFlow SavedModel
saved_model_path = "/home/jdickman/mmg/bamlab/mm_gnc/yolov5/runs/train/exp2/weights/best_saved_model"  # Update this path
tflite_path = "/home/jdickman/mmg/bamlab/mm_gnc/yolov5/runs/train/exp2/weights/fully_quantized_model.tflite"  # Path to save the TFLite model

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# Define a representative dataset generator
def representative_data_gen():
    image_folder = "/home/jdickman/mmg/bamlab/mm_gnc/dataset/train/images"
    images_paths = os.listdir(image_folder)[:100]  # Take the first 100 images

    for img_path in images_paths:
        # Construct the full image path
        img_full_path = os.path.join(image_folder, img_path)

        # Open the image, resize it to 640x640, and ensure it's in RGB
        img = Image.open(img_full_path).resize((640, 640)).convert('RGB')

        # Convert the image to a numpy array, normalize, and add batch dimension
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        yield [img_array]

# Point the converter to the representative dataset generator
converter.representative_dataset = representative_data_gen

# Restricting supported target ops to TFLITE_BUILTINS_INT8 ensures full integer quantization
# Before the conversion
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TFLite builtins ops
                                       tf.lite.OpsSet.SELECT_TF_OPS]  # Enable TensorFlow ops (Flex ops)

 Set the input and output tensors to uint8 (change to int8 if your edge device requires it)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Perform the conversion
tflite_quant_model = converter.convert()

# Save the TFLite model
with open(tflite_path, "wb") as f:
    f.write(tflite_quant_model)


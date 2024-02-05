# onnx_to_tf.py
import onnx
from onnx_tf.backend import prepare

# Path to the ONNX file
onnx_path = "/home/jdickman/mmg/bamlab/mm_gnc/yolov5/runs/train/exp2/weights/best.onnx"  # Update this path
tf_path = "/home/jdickman/mmg/bamlab/mm_gnc/yolov5/runs/train/exp2/weights/best_saved_model"  # Update this path

# Load the ONNX model
model_onnx = onnx.load(onnx_path)

# Convert to TensorFlow
tf_rep = prepare(model_onnx)

# Export the model to TensorFlow SavedModel format
tf_rep.export_graph(tf_path)


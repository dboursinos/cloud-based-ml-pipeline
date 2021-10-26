import engine as eng
import argparse
import numpy as np
from onnx import ModelProto
import tensorrt as trt
from minio import Minio
import pickle
import mlflow
import pycuda.driver as cuda
import pycuda.autoinit

engine_name = "model.plan"
onnx_path = "model.onnx"
batch_size = 1 

model = ModelProto()
with open(onnx_path, "rb") as f:
   model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
shape = [batch_size , d0, d1 ,d2]
engine = eng.build_engine(onnx_path)
eng.save_engine(engine, engine_name) 
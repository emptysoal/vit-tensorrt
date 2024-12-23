# -*- coding:utf-8 -*-

"""
    onnx 模型转 tensorrt 模型，并使用 tensorrt runtime 推理
"""

import os
import time

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

# import calibrator

classes_num = 1000
input_size = (224, 224)  # (rows, cols)
onnx_file = "./model.onnx"
trt_file = "./model.plan"

# for FP16 mode
use_fp16_mode = False
# for INT8 model
use_int8_mode = False
n_calibration = 20
cache_file = "./int8.cache"
# calibration_data_path = val_data_path

np.set_printoptions(precision=3, linewidth=160, suppress=True)


def get_engine():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.exists(trt_file):
        with open(trt_file, "rb") as f:  # read .plan file if exists
            engine_string = f.read()
        if engine_string is None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # set workspace for TensorRT
        if use_fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if use_int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator.MyCalibrator(calibration_data_path, n_calibration,
                                                             (16, 3, input_size[0], input_size[1]), cache_file)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnx_file):
            print("Failed finding ONNX file!")
            return
        print("Succeeded finding ONNX file!")
        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
            print("Succeeded parsing .onnx file!")

        input_tensor = network.get_input(0)
        profile.set_shape(input_tensor.name, [1, 3, input_size[0], input_size[1]], [4, 3, input_size[0], input_size[1]],
                          [8, 3, input_size[0], input_size[1]])
        config.add_optimization_profile(profile)

        engine_string = builder.build_serialized_network(network, config)
        if engine_string is None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trt_file, "wb") as f:
            f.write(engine_string)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)

    return engine


def image_preprocess(np_img):
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)  # bgr to rgb
    # resize
    img = cv2.resize(img, (int(input_size[1] * 1.143), int(input_size[0] * 1.143)), interpolation=cv2.INTER_LINEAR)
    # crop
    crop_top = (img.shape[0] - input_size[0]) // 2
    crop_left = (img.shape[1] - input_size[1]) // 2
    img = img[crop_top:crop_top + input_size[0], crop_left:crop_left + input_size[1], :]
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data = img.astype(np.float32)
    data = (data / 255. - np.array(mean)) / np.array(std)
    # transpose
    data = data.transpose((2, 0, 1)).astype(np.float32)  # HWC to CHW

    return data


def inference_one(data_input, context, buffer_h, buffer_d):
    """
        使用tensorrt runtime 做一次推理
    :param data_input: 经过预处理（缩放、裁剪、归一化等）后的图像数据（ndarray类型）
    """
    buffer_h[0] = np.ascontiguousarray(data_input)
    cudart.cudaMemcpy(buffer_d[0], buffer_h[0].ctypes.data, buffer_h[0].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(buffer_d)  # inference

    cudart.cudaMemcpy(buffer_h[1].ctypes.data, buffer_d[1], buffer_h[1].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outs = buffer_h[-1].reshape(-1)
    predict = np.exp(outs)
    predict = predict / np.sum(predict)
    cls = int(np.argmax(predict))
    score = predict[cls]

    return cls, score


if __name__ == '__main__':
    engine = get_engine()

    n_io = engine.num_bindings
    l_tensor_name = [engine.get_binding_name(i) for i in range(n_io)]
    n_input = np.sum([engine.binding_is_input(i) for i in range(n_io)])

    context = engine.create_execution_context()
    context.set_binding_shape(0, [1, 3, input_size[0], input_size[1]])
    for i in range(n_io):
        print("[%2d]%s->" % (i, "Input " if i < n_input else "Output"), engine.get_binding_dtype(i),
              engine.get_binding_shape(i), context.get_binding_shape(i), l_tensor_name[i])

    buffer_h = []
    for i in range(n_io):
        buffer_h.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    buffer_d = []
    for i in range(n_io):
        buffer_d.append(cudart.cudaMalloc(buffer_h[i].nbytes)[1])

    image_path = "../banana.jpeg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # read image

    start = time.time()
    input_data = image_preprocess(image)
    input_data = np.expand_dims(input_data, axis=0)  # add batch size dimension
    cate, prob = inference_one(input_data, context, buffer_h, buffer_d)
    print("Classify: %10s, prob: %.2f" % (cate, prob))
    end = time.time()

    print("Inference total cost is: %.3f" % (end - start))

    for b in buffer_d:
        cudart.cudaFree(b)

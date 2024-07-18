# -*- coding:utf-8 -*-

"""
    PyTorch 模型转 onnx 模型，并使用 onnx runtime 模型推理
"""

import os
import time

import cv2
import numpy as np
import torch
import onnxruntime

from vit_model import vit_base_patch16_224 as create_model

classes_num = 1000
input_size = (224, 224)  # (rows, cols)
pth_file = "../vision_transformer/vit_base_patch16_224.pth"
onnx_file = "./model.onnx"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pth2onnx():
    """
        PyTorch模型转onnx模型
    """
    model = create_model()
    model.load_state_dict(torch.load(pth_file, map_location="cpu"))
    model.to(device)

    torch.onnx.export(
        model,
        torch.randn((1, 3, input_size[0], input_size[1]), device=device),
        onnx_file,
        input_names=["data"],
        output_names=["prob"],
        do_constant_folding=True,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=12,
        dynamic_axes={"data": {0: "nBatchSize"}, "prob": {0: "nBatchSize"}}
    )
    print("Succeeded converting model into ONNX!")


def image_preprocess(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr to rgb
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


def inference_one(data_input, onnx_session):
    inputs = {onnx_session.get_inputs()[0].name: data_input}
    outs = onnx_session.run(None, inputs)[0]
    predict = np.exp(outs)
    predict = predict / np.sum(predict, axis=1)
    cls = np.argmax(predict, axis=1)[0]
    score = predict[0][cls]

    return cls, score


if __name__ == '__main__':
    if not os.path.exists(onnx_file):  # 不存在onnx模型就使用pth模型导出
        pth2onnx()

    # onnx inference
    session = onnxruntime.InferenceSession(
        onnx_file,
        providers=[
            # 'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            # 'CPUExecutionProvider'
        ]
    )

    start = time.time()

    image_path = "../banana.jpeg"
    input_data = image_preprocess(image_path)
    input_data = np.expand_dims(input_data, axis=0)  # add batch size dimension
    cate, prob = inference_one(input_data, session)
    print("Classify: %10s, prob: %.2f" % (cate, prob))

    end = time.time()
    print("Inference total cost is: %.3f" % ( end - start))

# TensorRT 部署 ViT

1. 这里有 2 种方式构建 `ViT` 网络模型，分别为：

- `PyTorch` 构建 `ViT` 网络
- 使用 `transformers` 库中的 `ViT`

2. 两种方式下，分别做 `.pth` -> `.onnx` -> `.plan` 的转换

## 一. PyTorch 构建 ViT 网络

- 即本项目的 `vision_transformer` 目录，进入该目录

- 代码来源：[vision_transformer](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)，对其中的 `predict.py` 做了少许修改
- `pytorch` 推理：

```bash
这里使用 vit_base_patch16_224 模型
根据 vit_model.py 的 295 行的链接，下载模型文件，放到当前目录
python predict.py
```

- 导出 `onnx`：

```bash
cd ../onnx_parser
python onnx_infer.py  # 运行成功后，会在当前目录生成 model.onnx
```

- 转换 `TensorRT`：

```bash
python trt_infer.py  # 运行成功后，会在当前目录生成 model.plan
```

- 使用 `C++` 代码转换 `TensorRT`：

```bash
cd ../cpp_onnx_parser
cp ../onnx_parser/model.onnx .
make
./trt_infer
```

备注：`onnx` 和 `TensorRT` 的转换，参考本人这个项目：[tensorrt-experiment](https://github.com/emptysoal/tensorrt-experiment) 中的分类任务，环境也可根据这个项目中的环境构建部分来配置

## 二. 使用 transformers 库中的 ViT

- 即本项目的 `vision_transformer_2` 目录，进入该目录
- 参考 https://huggingface.co/google/vit-base-patch16-224 ，也做了改动（下载相关文件到本地，然后加载模型）
- 在安装有 `torch` 的环境的基础上，安装必要的库：

```bash
pip install transformers
pip install optimum[exporters]
```

- 根据上方链接下载 `model.safetensors` 模型文件，`config.json` 和 `preprocessor_config.json`，放入 `local-pt-checkpoint` 目录下；
- `pytorch` 推理：

```bash
python vit.py
```

- 导出 `onnx`：

```bash
python export.py  # 运行成功后，model.onnx 会生成到当前目录的 onnx 目录下
```

- 转换 `TensorRT`：

```bash
cd ../onnx_parser
cp ../vision_transformer_2/onnx/model.onnx .
python trt_infer.py  # 运行成功后，会在当前目录生成 model.plan
```

- 使用 `C++` 代码转换 `TensorRT`：

```bash
cd ../cpp_onnx_parser
cp ../vision_transformer_2/onnx/model.onnx .
make
./trt_infer
```

备注：导出 `onnx` 的代码的参考链接：https://blog.csdn.net/engchina/article/details/136792093

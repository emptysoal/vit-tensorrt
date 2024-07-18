from optimum.onnxruntime import ORTModelForImageClassification

model_checkpoint = "local-pt-checkpoint"
save_directory = "onnx/"

ort_model = ORTModelForImageClassification.from_pretrained(model_checkpoint, export=True)

ort_model.save_pretrained(save_directory)


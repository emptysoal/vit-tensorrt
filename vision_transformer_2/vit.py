from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

image = Image.open("../banana.jpeg")

# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# processor = ViTImageProcessor.from_pretrained('./vit/preprocessor_config.json')
processor = ViTImageProcessor.from_pretrained('./local-pt-checkpoint')

# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('./local-pt-checkpoint')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])


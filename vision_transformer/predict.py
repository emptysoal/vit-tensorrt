import os
import time
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from vit_model import vit_base_patch16_224_in21k as create_model
from vit_model import vit_base_patch16_224 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "../banana.jpeg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # create model
    model = create_model().to(device)
    # load model weights
    model_weight_path = "./vit_base_patch16_224.pth"
    # weights_dict = torch.load(model_weight_path, map_location=device)

    # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
    #         else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    # for k in del_keys:
    #     del weights_dict[k]
    # print(model.load_state_dict(weights_dict, strict=False))


    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    start = time.time()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(time.time() - start)

    print(predict_cla)
    print(predict[predict_cla])


if __name__ == '__main__':
    main()

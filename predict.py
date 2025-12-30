
import os
import pandas as pd
import torch
import torchvision
import LUNA
from PIL import Image
import numpy as np
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# 输入图片文件夹路径
folder_path = './input_image/'

# 初始化模型
model_X = LUNA.IQANetwork(num_blocks=1).cuda()
model_X.train(False)



model_X.load_state_dict(
    torch.load('./model_pth/LUNA.pth'))

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])



def save_scores_to_excel(folder_path, excel_path):
    data = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
            continue

        pred_scores = []
        for i in range(10):
            img = pil_loader(file_path)
            img = transforms(img)
            img = img.cuda().clone().detach().unsqueeze(0)
            pred = model_X(img)
            pred_scores.append(float(pred.item()))
        score = np.mean(pred_scores)

        print(f'Image: {filename}, Predicted quality score: {score:.2f}')

        data.append([filename, score])

    df = pd.DataFrame(data, columns=['Filename', 'Score'])
    df.to_excel(excel_path, index=False, engine='openpyxl')


excel_path = "./luna_test.xlsx"
save_scores_to_excel(folder_path, excel_path)
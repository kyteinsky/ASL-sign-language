import torch
import torchvision.transforms as transforms
import os
import cv2
from time import time

test_transforms = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor()])

model=torch.load('ckpt/sign_lang_lr_0.001_epo_25_0.pth')
model.eval()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    output = model(input)
    index = output.data.numpy().argmax()
    return index


images = []
for  in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
folder = 'test_images/'

for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is None: continue
    st = time()
    pred = classes[predict_image(img)]
    print(f'{filename} is for "{pred}"')
    print(f'Inference took {time()-st} s')

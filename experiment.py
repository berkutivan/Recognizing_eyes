from nn_to_seg import  *
from camera import *
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

device  = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ExtremeC3Net(classes=3, p=1, q=5)  #левый глаз правый глаз
print(device)


import numpy as np




def load_image(pixels):
    image = Image.fromarray(pixels.astype('uint8'),'RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Добавляем batch dimension
    return image

def get_mask(image):
    global  model
    image_t = load_image(image)
    masks = model(image_t)[0].permute(1,2,0).detach().numpy()
    masks = cv2.resize(masks, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)

    image[masks > 0.1] = masks[masks > 0.1]
    return image

try:
    checkpoint = torch.load("ExtremeC3_last.pt",map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
except Exception as e:
    print(e)

cam = Camera(frame_width= 512, frame_height= 512)
cam.start(func = get_mask)
#cam.start(func = lambda image:cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) )
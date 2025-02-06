import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision
import random
from torchvision import transforms, datasets, models
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from nn_to_seg import ExtremeC3Net, get_face_mask, load_image, just_eyes

'''
===============Проверка_данных_обучения=====================================================================================================
'''
data_link = "data_to_seg"
val_data_link = "val_data_to_seg"
DATA_MODES = ['train', 'val', 'test']
device  = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

class Segmentation_frame(Dataset):
    def __init__(self, links, mode):
      super().__init__()

      self.links = links
      self.get_link = lambda x: self.links + '/' + x
      self.files = os.listdir(self.links)
      self.data = np.array([self.files[x:3 + x] for x in range(0,len(self.files),3)])
      self.len_ = self.data.size//3
      self.index = np.arange(self.len_)
      self.mode = mode

      if self.mode not in DATA_MODES:
          print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
          raise NameError

    def __len__(self):
        return self.len_

    def shuffle(self):
        np.random.shuffle(self.index)

    def get_points(self, links):
        file = open(self.get_link(links), 'r')
        points_text = file.read().split("\n")
        points_text = [x for x in points_text if x != '']
        points = np.array([[float(point.split(' ')[0]), float(point.split(' ')[1])] for point in points_text])
        return points

    def load_masks(self, image_path):
        masks = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
        ])
        masks = transform(masks).unsqueeze(0)  # Добавляем batch dimension
        return masks


    def __getitem__(self, index):

        if self.mode == 'test':
            data_point = self.data[self.index[index]]
            points = self.get_points(data_point[1]) #набор точек numpy
            masks = self.load_masks(self.get_link(data_point[2])) #синий левый глаз, красный - правый
            return points,  just_eyes(masks[0].permute(1, 2, 0))

        elif self.mode == 'train':
            data_point = self.data[self.index[index]]
            real = load_image(self.get_link(data_point[0])) # первичное изображение -> torch
            points = self.get_points(data_point[1]) # набор точек numpy
            masks = self.load_masks(self.get_link(data_point[2])) # синий левый глаз, красный - правый

            return real[0], points, just_eyes(masks[0].permute(1, 2, 0))

df = Segmentation_frame(data_link, 'train')
real, points, masks = df[2]
'''
print(points[0][0], points[0][1])
image_path = "LpnbOBGSHrI.jpg"
image_tensor = load_image(image_path)
model = ExtremeC3Net(classes=2, p=1, q=5)
face_mask = get_face_mask(image_tensor, model)
face_mask = face_mask.cpu().numpy()
plt.imshow(face_mask, cmap='gray')
plt.show()
'''

plt.scatter(points[0,0], points[0,1])
plt.imshow(masks)
plt.show()

'''
===============Функции_для_обучения=========================================================================================================
'''
def My_Loss(pred, check):
    #pred = torch.sigmoid(model(pred))
    pred = pred.permute(0,2,3,1)
    metrik = (pred-check)**2

    return (metrik + torch.mul(metrik,check)*100).mean()

def predict(model, test_loader, DEVICE):
    with torch.no_grad():
        logits = []

        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            model.eval()
            outputs = model(inputs.unsqueeze(0)).cpu()
            logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1)
    return probs

def eval_epoch(model, val_loader, criterion, DEVICE):
    model.eval()
    running_loss = 0.0
    processed_size = 0
    count = 0
    for inputs,points,masks in val_loader:
        count +=1
        inputs = inputs.to(DEVICE)
        masks = masks.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, masks)

        running_loss += loss.item() * inputs.size(0)
        #running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
        if count%10==0:
          print(count)

    val_loss = running_loss / processed_size
    val_acc = 0
    return val_loss, val_acc

def fit_epoch(model, train_loader, criterion, optimizer, DEVICE):
    running_loss = 0.0
    processed_data = 0


    count = 0

    for inputs, points, masks in train_loader:
        #print(type(inputs), type(points), type(masks))
        count +=1
        inputs = inputs.to(DEVICE)
        #print(masks.size(),"in the train")
        masks = masks.to(DEVICE)
        optimizer.zero_grad()
        #print(inputs.size(),"in the train")
        outputs = model(inputs)

        loss = criterion(outputs, masks)
        print(type(loss), loss)
        #print(loss.backward)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        processed_data += inputs.size(0)
        if count%10==0:
          print(count)

    train_loss = running_loss / processed_data
    train_acc = 0
    return train_loss, train_acc

def train(train_files,val_files, model, epochs, batch_size, device, MyLoss):

    best_val_loss = None
    train_loader = DataLoader(train_files, batch_size=batch_size, shuffle= True)
    val_loader = DataLoader(val_files, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr= 0.001)

    try:
        pt_file = torch.load("ExtremeC3_last.pt")
        optimizer.load_state_dict(pt_file['optimizer_state_dict'])
    except:
       print("Warning : no file")

    history = []

    criterion = MyLoss

    for epoch in range(epochs):
        print("EPOCH")
        model.train()
        train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer, device)
        print("loss", train_loss)
        val_loss, val_acc =  eval_epoch(model, val_loader, criterion, device)
        history.append((train_loss, train_acc, val_loss, val_acc))
        print("validation loss:", val_loss)

        f = open("ExtremeC3_info.txt", "a")
        point = str(train_loss) + ", " + str(val_loss)
        f.write(point)
        f.close()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        },"ExtremeC3_last.pt")

        flag = False
        if best_val_loss == None:
            flag = True
        elif val_loss < best_val_loss:
            flag = True

        if flag:
            best_val_loss = val_loss
            torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                            }, "ExtremeC3_best.pt")



    print("end")
    return history

'''
===============Загрузка_модели==============================================================================================================
'''

model = ExtremeC3Net(classes=3, p=1, q=5)  #левый глаз правый глаз

try:
    checkpoint = torch.load("ExtremeC3_last.pt",map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
except Exception as e:
    print(e)

print(real.size(), "out of train")
answer = (model(real.unsqueeze(0)))
print(answer.max())
plt.imshow(answer[0].permute(1,2,0).detach().numpy())
plt.show()

model.to(device)
'''
===============Обучение_модели==============================================================================================================
'''
train_df = Segmentation_frame(data_link, 'train')
val_df = Segmentation_frame(val_data_link, 'train')


history = train(train_df, val_df, model=model, epochs= 50, batch_size = 10, device= device, MyLoss = My_Loss)
train_loss, train_acc, val_loss, val_acc = zip(*history)

print(answer.detach().numpy().max())
plt.imshow(answer[0].permute(1,2,0).detach().numpy())
plt.show()

'''
===============Тест_модели==================================================================================================================
'''
import torch
import os
from  PIL import  Image
import numpy as np
print(torch.cuda.is_available())

a = torch.tensor([ [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]]
])

b = torch.tensor([ [[0,1,0],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,1,0]]
])
a = np.array([ [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]]
])

b = np.array([ [[0,1,0],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,0,1]],
                   [[0,0,1],  [0,1,0]]
])

print(a==b)
path = "data_to_seg/000010.png"
image = Image.open(path)

w, h = image.size

print(w,h)
print(0.1 < float(torch.isinf(0.1)))
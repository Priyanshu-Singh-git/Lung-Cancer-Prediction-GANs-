#%% SEEING THE DATA ON PLOTS
import cv2,os
import numpy as np 
import matplotlib.pyplot as plt


directory_1 = "D:/pythonProject/lung cancer GAN/Data"
q = input("enter the file type  ")
l = input("enter type of carcinoma")

directory = os.path.join(directory_1,q,l)
files = os.listdir(directory)

num_col = 10
num_rows = (len(files) + num_col - 1) // num_col
fig, axes = plt.subplots(num_rows, num_col, figsize=(100,100))
axes=axes.flatten()
for i,list in enumerate(os.listdir(directory)):
    file_path = os.path.join(directory,list)
    img = cv2.resize(cv2.imread(file_path),(224,224))
    
    if img is None:
        
        print('check for file path error')
        
    else:
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(list)
        axes[i].axis("off")
        
for j in range(len(os.listdir(directory)), len(axes)):
    axes[j].axis('off')  
plt.tight_layout()
plt.show()

#%% PREPROCESSING DATA

base_dir = "D:/pythonProject/lung cancer GAN/Data"
directories = ['train']
subdirectories = ['remaining']

target_path1 = "D:/pythonProject/lung cancer GAN/Data/train/remaining_inter_yes"


def normalize(array):
    return array.astype(np.float32)/255.0
    
    
def resize(array,target = (128,128),pad_color=(0,0,0)):
    h,w = array.shape[:2]
    target_h,target_w = target

    if h > w:
        new_h = target_h
        new_w = int(w * (target_h / h))
    else:
        new_w = target_w
        new_h = int(h * (target_w / w))

    img_resize = cv2.resize(array,(new_w,new_h),interpolation = cv2.INTER_AREA)
    
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    
    new_img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    return new_img
    
        
    
var = 1
for i,directory in enumerate(directories):
    for j,subdirectory in enumerate(subdirectories):
        current_dir = os.path.join(base_dir, directory, subdirectory)
        print(current_dir)

        for filename in os.listdir(current_dir):
            file_path = os.path.join(current_dir, filename)
            
            
            img = cv2.imread(file_path)
            img_r = resize(img)
            
            
            print("max ",np.max(img_r),"min",np.min(img_r))
            
            file_name = str(var)+"interarea.png"
            target_path3 = os.path.join(target_path1,file_name)
            cv2.imwrite(target_path3, (img_r).astype(np.float32))
            var+=1
            
            
#%% TARGET DATA ITERABLE CREATION
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,random_split
data_direc = "D:/pythonProject/lung cancer GAN/Data/train"
lung_cancer_direc = os.path.join(data_direc,"lung_cancer_new")
normal_direc = os.path.join(data_direc,"normal_new")

data_transforms =  transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        
        ])
    
image_dataset = datasets.ImageFolder(data_direc,transform = data_transforms)

dataloader = DataLoader(image_dataset,batch_size=32,shuffle=True)
class_names = image_dataset.classes
dataset_size = len(image_dataset)
print(class_names)

#%%
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg19(weights = True)

for param in model.parameters():
    param.requires_grad = False
    


num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

for param in model.classifier[6].parameters():
    param.requires_grad = True
print(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier[6].parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
loss_l=[]
acc_l =[]
def train_model(model,criterion,optimizer,scheduler,num_epochs = 25):
    for epoch in range(num_epochs):
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs,labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _,preds = torch.max(outputs,1)
            loss = criterion(outputs,labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()*inputs.size(0)
            running_corrects +=torch.sum(preds == labels.data)
            
        scheduler.step()
        
        epoch_loss = running_loss/dataset_size
        
        epoch_acc = running_corrects.double()/dataset_size
        print(f'Train loss : {epoch_loss:.4f}  Acc : {epoch_acc:.4f}')
        print()
        loss_l.append(epoch_loss)
        acc_l.append(epoch_acc.item())
    return model

model = train_model(model,criterion,optimizer,scheduler,num_epochs=25)

plt.plot(loss_l,label="loss")
plt.plot(acc_l,label = "accuracy")
plt.legend()
plt.show()

torch.save(model.state_dict(),"D:/pythonProject/lung cancer GAN/vgg19_transfer_learning_updated.pth")


        





    

    
    
    




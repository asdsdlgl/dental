import pandas as pd
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
import PIL
import json
from sklearn.utils import shuffle
def getData(mode, Istestori = False):
    
    if Istestori == True:
        filename = 'teeth_clahe_hflip_without_rotation.json'
        with open(filename,'r') as dicctionary:   

            concate_dict = pd.DataFrame()    
            df = json.loads(dicctionary.read())

            data = pd.DataFrame({key:value for key,value in df.items()if key in df},index = [0]).T
            print('data length =',len(data))
            data = data[6000:8000]
            concate_dict= pd.concat([concate_dict,data],sort=False,axis=0)
            concate_dict[0] = concate_dict[0].astype(int)
            for i in range(1,33):
                print('class {}:\tlength :{}'.format(i,len(concate_dict[concate_dict[0]==i])))
            img = concate_dict[0].keys().values
            label = concate_dict[0].values - 1
            return np.squeeze(img),np.squeeze(label)        
    
    
    
    filename = 'teeth_clahe_hflip.json'
    with open(filename,'r') as dicctionary:   

        concate_dict = pd.DataFrame()    
        df = json.loads(dicctionary.read())
        
        data = pd.DataFrame({key:value for key,value in df.items()if key in df},index = [0]).T
        print('data length =',len(data))
#         if mode == 'train':
#             data = data[:60000].sample(15000)#14000 240000 274000
#         elif mode == 'test':
#             data = data[60000:80000].sample(5000)
#         elif mode == 'val':
#             data = data[80000:].sample(4000)

#         if mode == 'train':
#             data = data[:6556]
#         elif mode == 'test':
#             data = data[6556:8556]
#         elif mode == 'val':
#             data = data[8556:]

        if mode == 'train':
            data = data[:66000].sample(16500)#14000 240000 274000
        elif mode == 'test':
            #data = data
            data = data[66000:88000]#.sample(10000)
        elif mode == 'val':
            data = data[88000:].sample(8000)
        
        concate_dict= pd.concat([concate_dict,data],sort=False,axis=0)
        
        if mode != 'test':
            concate_dict = shuffle(concate_dict)
        concate_dict[0] = concate_dict[0].astype(int)
        
        for i in range(1,33):
            print('class {}:\tlength :{}'.format(i,len(concate_dict[concate_dict[0]==i])))
        
        #print(concate_dict)
        img = concate_dict[0].keys().values
        label = concate_dict[0].values - 1
        
        return np.squeeze(img),np.squeeze(label)


class GetDataset(data.Dataset):
    def __init__(self, root, mode, Istestori = False):

        self.root = root
        self.mode = mode
        self.img_name, self.label = getData(mode, Istestori)
#         new_size = (150,80)
        new_size = (200,150)
        trans_method = [
            transforms.Resize(new_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
        trans_flipback = [
            transforms.Resize(new_size),
            transforms.RandomVerticalFlip(p=1),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
        self.trans = transforms.Compose(trans_method)
        self.trans_back = transforms.Compose(trans_flipback)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        path = os.path.join(self.root, self.img_name[index])
        label = self.label[index]
        img = PIL.Image.open(path).convert('RGB')
        
        if label < 16:
            img = self.trans_back(img)
        else:
            img = self.trans(img)
            
        if self.mode == 'test':
            return img, label, path
        
        return img, label
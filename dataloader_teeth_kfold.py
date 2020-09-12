import pandas as pd
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
import PIL
import json
from sklearn.utils import shuffle
def getData(mode, Istestori = False, folds = [0,1,2,3,4]):
    
    kfold_section = [[0,38200],[38200,76400],[76400,114600],[114600,152800],[152800,191040]]
    #folds = [0,1,2,3,4] # train train train val test
    
    if Istestori == True:
        filename = 'teeth_clahe_hflip_without_rotation.json'
        with open(filename,'r') as dicctionary:   

            concate_dict = pd.DataFrame()    
            df = json.loads(dicctionary.read())

            data = pd.DataFrame({key:value for key,value in df.items()if key in df},index = [0]).T        
            data = data[kfold_section[folds[4]][0]//20:kfold_section[folds[4]][1]//20] #origin test 6000-8000 0-2000 -2000
            print('data length =',len(data))
            concate_dict= pd.concat([concate_dict,data],sort=False,axis=0)
            concate_dict[0] = concate_dict[0].astype(int)
#             for i in range(1,33):
#                 print('class {}:\tlength :{}'.format(i,len(concate_dict[concate_dict[0]==i])))
            img = concate_dict[0].keys().values
            label = concate_dict[0].values - 1
            return np.squeeze(img),np.squeeze(label)        
    
    
    
    filename = 'teeth_clahe_hflip_10degree.json'
    with open(filename,'r') as dicctionary:   

        concate_dict = pd.DataFrame()    
        df = json.loads(dicctionary.read())

        data = pd.DataFrame({key:value for key,value in df.items()if key in df},index = [0]).T
        print('data length =',len(data))

        if mode == 'train':
            for fold in folds[:3]:
                datum = data[kfold_section[fold][0]:kfold_section[fold][1]]
                concate_dict= pd.concat([concate_dict,datum],sort=False,axis=0)
            concate_dict = concate_dict.sample(30000)
        elif mode == 'val':
            for fold in folds[3:4]:
                datum = data[kfold_section[fold][0]:kfold_section[fold][1]]
                concate_dict= pd.concat([concate_dict,datum],sort=False,axis=0)
            concate_dict = concate_dict.sample(10000)
        elif mode == 'test':
            for fold in folds[4:]:
                datum = data[kfold_section[fold][0]:kfold_section[fold][1]]
                concate_dict= pd.concat([concate_dict,datum],sort=False,axis=0)
            concate_dict = concate_dict.sample(10000)

        if mode != 'test':
            concate_dict = shuffle(concate_dict)
        concate_dict[0] = concate_dict[0].astype(int)

#         for i in range(1,33):
#             print('class {}:\tlength :{}'.format(i,len(concate_dict[concate_dict[0]==i])))

        img = concate_dict[0].keys().values
        label = concate_dict[0].values - 1
        
        return np.squeeze(img),np.squeeze(label)


class GetDataset(data.Dataset):
    def __init__(self, root, mode, Istestori = False, folds = [0,1,2,3,4]):

        self.root = root
        self.mode = mode
        self.img_name, self.label = getData(mode, Istestori, folds)
#         new_size = (200,150)
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
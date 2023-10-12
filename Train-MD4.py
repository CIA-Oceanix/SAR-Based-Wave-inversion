import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.colors import ListedColormap

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data =torch.load('../data/TD_2020_V1.pt')


class Spectrum_dataset(Dataset):
    
    def __init__(self,data,initial,l):
        self.l = l
        self.initial = initial
        sar = data['CS_tot'][self.initial:self.initial+self.l]
        ww3 = data['ww3_spec'][self.initial:self.initial+self.l]
        #sar_real = data['sar_real_smooth'][self.initial:self.initial+self.l]

        self.data = sar,ww3
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self,idx):
     
        return self.data[0][idx],self.data[1][idx]
        
Dataset_train = Spectrum_dataset(data,1000,100000)
Dataset_val = Spectrum_dataset(data,101000,10000)
data_loader_train = DataLoader(dataset=Dataset_train, batch_size=100,num_workers=2,shuffle =True)
data_loader_val= DataLoader(dataset=Dataset_val,batch_size=100,num_workers=2)


class conv_block(pl.LightningModule):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)         
        self.relu = nn.ReLU()     
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
        
 class encoder_block(pl.LightningModule):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))     
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
        
class decoder_block(pl.LightningModule):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)     
    def forward(self, inputs, skip):
        x = self.up(inputs)

        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class build_unet(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.lr =config['lr']
        self.e1 = encoder_block(2, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.b = conv_block(256, 512)        
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)        
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)   
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        #inputs = torch.log(self.relu(inputs) +1)
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        b = self.b(p3)    
        d2 = self.d2(b, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)         
        outputs = self.outputs(d4)   
        outputs= self.relu(outputs)
        
        return outputs
    
    def configure_optimizers(self):       
        optim = torch.optim.AdamW(self.parameters(),lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode ='min', patience=3,threshold=5e-2, factor=0.9)
        return  {"optimizer" : optim ,"lr_scheduler" : scheduler,  'monitor' : 'val_loss'}
    
    def loss(self,pred_spec,true_spec):     
        
        l = 1000*F.mse_loss(pred_spec.squeeze(),true_spec)

        return l
    
    def training_step(self,batch,batch_idx):
        sar,ww3=batch

        l=len(sar)
        for i in range(l):
            if torch.max(sar[i][0])>1:
                sar[i][0] = sar[i][0]/torch.max(sar[i][0])
                sar[i][1] = sar[i][1]/torch.max(sar[i][1])
            ww3[i] = ww3[i]/torch.max(ww3[i])
        
        pred_spec=self(sar.to(torch.float))
        loss= self.loss(pred_spec,ww3.to(torch.float))
        self.log('loss',loss)
        return {'loss' : loss}
    def validation_step(self,batch,batch_idx):
        sar,ww3=batch
        l=len(sar)
        for i in range(l):
            if torch.max(sar[i][0])>1:
                sar[i][0] = sar[i][0]/torch.max(sar[i][0])
                sar[i][1] = sar[i][1]/torch.max(sar[i][1])
            ww3[i] = ww3[i]/torch.max(ww3[i])
        pred_spec=self(sar.to(torch.float))
        loss= self.loss(pred_spec,ww3.to(torch.float))
        self.log('val_loss',loss)
        return {'val_loss' : loss}
    def predict_step(self,batch,batch_idx):
        sar,ww3=batch
        real_ww3=ww3.clone()

        l=len(sar)

        for i in range(l):
            if torch.max(sar[i][0])>1:
                sar[i][0] = sar[i][0]/torch.max(sar[i][0])
                sar[i][1] = sar[i][1]/torch.max(sar[i][1])


            ww3[i] = ww3[i]/torch.max(ww3[i])
        pred_spec_real =self(sar.to(torch.float))
        #pred_spec_real = torch.exp(pred_spec_real)-1

        return pred_spec_real
        
config={'lr':5e-4}


logger = pl.loggers.TensorBoardLogger(name=f'MD4-simu-lr={config["lr"]}', save_dir='inversion_logs')
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(gpus=[6],progress_bar_refresh_rate=5,max_epochs=15,logger=logger,callbacks=[lr_monitor],detect_anomaly=False)
model = build_unet(config)
trainer.fit(model,data_loader_train,data_loader_val)


"""
Training script
"""

import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from dataset.dataset import Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss

from net.ResUNet import net

import parameter as para

import matplotlib.pyplot as plt


import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 


cf.go_offline()

step_list = [0]

# Set graphics card related
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# Define the network
net = torch.nn.DataParallel(net).cuda()
net.train()

# Define Dateset
train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))

# Define data loading
train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)

# loss function
loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
loss_func = loss_func_list[5]

# Define optimizer
opt = torch.optim.Adam(net.parameters(), lr=0.01)
#opt = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

# Learning rate decay
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# In-depth supervision attenuation coefficient
alpha = para.alpha

# Training network
start = time()
for epoch in range(para.Epoch):

    lr_decay.step()

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4

        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 is 0:
            
            step_list.append(step_list[-1] + 1)
            
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)

    # Save model
    if epoch % 50 is 0 and epoch is not 0:

        # The naming method of the network model is: the number of epoch rounds + the loss of the current minibatch + the average loss of the current epoch
		#torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))
        print('50%%%%%%%%%%%%%%%%%%%%%%')


#  depth supervision coefficient
    if epoch % 40 is 0 and epoch is not 0:
        alpha *= 0.8
        print('40%%%%%%%%%%%%%%%%%%%%%%')



init_notebook_mode(connected=False)

plt.plot(epoch, mean_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

#
#plt.show()
		
    
	
    


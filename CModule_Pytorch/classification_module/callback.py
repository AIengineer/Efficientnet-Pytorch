import os
import torch
import time
import torch.nn as nn
from utils.utils import CustomDataParallel
# from .utils import CustomDataParallel

def SaveModelCheckpoint(model ,PATH, epoch, value=0., save_best_opt=False):
    os.makedirs(PATH,exist_ok=True)
    if save_best_opt :
        if isinstance(model, nn.DataParallel) or isinstance(model, CustomDataParallel):
            print("Saving multi-gpus model...")
            torch.save(model.module, os.path.join(PATH,'weights-improvement-epoch-%04d-val_loss-%04f.pth' % (epoch, value)))
        else:
            print("Saving model...")
            torch.save(model, os.path.join(PATH,'weights-improvement-epoch-%04d-val_loss-%04f.pth' % (epoch, value)))
    else:
        if isinstance(model, nn.DataParallel) or isinstance(model, CustomDataParallel):
            print("Saving multi-gpus model...")
            torch.save(model.module, os.path.join(PATH,'%s_%04d.pth' % (time.strftime('%Y%m%d', time.localtime()), epoch)))
        else:
            print("Saving model...")
            torch.save(model, os.path.join(PATH,'%s_%04d.pth' % (time.strftime('%Y%m%d', time.localtime()), epoch)))
import sys
sys.path.append('/home/whao/pose_eatimate/Feed-forward_iGuassion-weight-matching-7-pose')
import torch
from torch.utils.data import DataLoader

from models.util.save_model import save_model_dp
from utils.config_util import load_config_module
from utils.calculate_pose_loss import PoseLossCalculator

import pdb

class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model

        self.loss_total = PoseLossCalculator()

    def forward(self, target_imgs, source_imgs, source_poses, target_pose, train=True):
        pred_pose = self.model(target_imgs, source_imgs, source_poses)

        loss = self.loss_total(pred_pose, target_pose)

        return loss


def train_epoch(model, dataset, optimizer, configs, epoch):
    # Last iter as mean loss of whole epoch
    model.train()
    train_loss = 0
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (target_imgs, source_imgs, source_poses, target_pose) in enumerate(dataset):
        target_imgs = target_imgs.cuda()
        source_imgs = source_imgs.cuda()
        source_poses = source_poses.cuda()
        target_pose = target_pose.cuda()

        loss = model(target_imgs, source_imgs, source_poses, target_pose)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        train_loss += loss.item()
        # if idx % 8 == 0:
        #     print(idx, loss.item(), '*' * 10)
    print(epoch, train_loss/len(dataset), '*' * 10)
        

def val_epoch(model, dataset):
    # Last iter as mean loss of whole epoch
    # model.eval()
    # model.train()
    val_loss = 0
    with torch.no_grad():
        for idx, (target_imgs, source_imgs, source_poses, target_pose) in enumerate(dataset):
            target_imgs = target_imgs.cuda()
            source_imgs = source_imgs.cuda()
            source_poses = source_poses.cuda()
            target_pose = target_pose.cuda()

            loss = model(target_imgs, source_imgs, source_poses, target_pose)
            val_loss += loss.item()

    val_loss /= len(dataset)

    return val_loss


def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('use gpu ids is'+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)

    ''' models and optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.DataParallel(model)

    # optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    # scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    
    # epoch = 200
    # steps_per_epoch = dataset size / dataset size = 3200 / 1 = 3200
    pct_warmup = 50000 / 640000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        1e-5, 640000, pct_start=pct_warmup, div_factor=25, cycle_momentum=False)

    ''' dataset '''
    Train_Dataset = getattr(configs, "train_dataset", None)
    if Train_Dataset is None:
        Train_Dataset = configs.training_dataset
    train_loader = DataLoader(Train_Dataset(), **configs.train_loader_args, pin_memory=True)

    Val_Dataset = getattr(configs, "val_dataset", None)
    if Val_Dataset is None:
        Val_Dataset = configs.val_dataset
    val_loader = DataLoader(Val_Dataset(), **configs.val_loader_args, pin_memory=True)

    ''' get validation '''
    best_val_loss = 200
    trigger_times = 0
    for epoch in range(configs.epochs):
        print('*' * 100, epoch)
        # best_val_loss = float('inf')
        patience = 4

        # train and val
        train_epoch(model, train_loader, optimizer, configs, epoch)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            save_model_dp(model, None, configs.model_save_path, 'ep%03d.pth' % epoch)

        if (epoch + 1) % 20 == 0:
        # save_model_dp(model, None, configs.model_save_path, 'ep%03d.pth' % epoch)
            val_loss = val_epoch(model, val_loader)
            # scheduler.step(val_loss)
            if val_loss < best_val_loss:
                trigger_times = 0
                print("val_loss", val_loss)
                print("best_val_loss", best_val_loss)
                best_val_loss = val_loss
                trigger_times = 0
                # torch.save(model.state_dict(), 'best_model.pth')
                save_model_dp(model, None, configs.model_save_path, 'best_model.pth')
                print(f'Saved Best Model with Validation Loss: {val_loss:.4f}')
            else:
                trigger_times += 1
                print("trigger_times", trigger_times)  
                if trigger_times >= patience:
                    print('Early stopping triggered!')
                    break

# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    seed_value=114514
    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

    worker_function('./config_blender.py', gpu_id=[0])

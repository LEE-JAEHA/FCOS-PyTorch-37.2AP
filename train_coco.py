from model.fcos import FCOSDetector
import torch
from dataset.COCO_dataset import COCODataset
import math,time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from coco_eval import COCOGenerator
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=48, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu
print(os.environ["CUDA_VISIBLE_DEVICES"])
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()

resize_ = [600,600]
train_dataset=COCODataset("/HDD/jaeha/dataset/COCO/2014/train2014",
                          '/HDD/jaeha/dataset/COCO/2014/annotations/instances_train2014.json',transform=transform,resize_size=resize_)


val_dataset=COCOGenerator("/HDD/jaeha/dataset/COCO/2017/train2017",
                          '/HDD/jaeha/dataset/COCO/2017/annotations/changed500_instances_val2017.json',resize_size=resize_)

# changed_val_dataset=COCOGenerator("/HDD/jaeha/dataset/COCO/2017/val2017",
#                           '/HDD/jaeha/dataset/COCO/2017/annotations/changed_instances_train2014.json',resize_size=resize_)


# train_dataset=COCODataset("/HDD/jaeha/dataset/COCO/2017/val2017",
#                           '/HDD/jaeha/dataset/COCO/2017/annotations/instances_val2017.json',transform=transform,resize_size=[600,600])
model = FCOSDetector(mode="training").cuda()
print(model)

model = torch.nn.DataParallel(model)
BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMUP_STEPS = 500
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS = 0
# LR_INIT = 0.01 #origin
LR_INIT = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=0.0001)
lr_schedule = [120000, 160000]


def lr_func(step):
    lr = LR_INIT
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr * warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)


# model.train()

best_acc = -1
best_ep = -1
for epoch in range(EPOCHS):
    print(
        "{2} Epoch Train Finish. Img size : {0} Batch : {1} / train with dilation mode".format(resize_, opt.batch_size,
                                                                                               epoch + 1))
    model.train()
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        lr = lr_func(GLOBAL_STEPS)
        for param in optimizer.param_groups:
            param['lr'] = lr

        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 3)
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)
        if epoch_step % 100 == 0:
            print(
                "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                 losses[2].mean(), cost_time, lr, loss.mean()))
        GLOBAL_STEPS += 1
    torch.save(model.state_dict(), "./checkpoint/model_{}.pth".format(epoch + 1))

    if epoch + 1 > 20:
        model2 = FCOSDetector(mode="inference")
        model2 = torch.nn.DataParallel(model2)
        model2 = model2.cuda().eval()
        model2.load_state_dict(
            torch.load("./checkpoint/model_{}.pth".format(epoch + 1), map_location=torch.device('cuda:1')),
            strict=False)
        tt = coco_eval.evaluate_coco(val_dataset, model2)
        m_acc = tt[4].astype(float)
        if m_acc > best_acc:
            best_acc = m_acc
            best_ep = epoch + 1
        print("Best Acc of Medium : {0}, Best Ep of Medium : {1}".format(best_acc, best_ep))







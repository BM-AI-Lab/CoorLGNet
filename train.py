import argparse
import os
import sys
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from CoorLGNet import coorlgnet
import xlwt
import sklearn.metrics as sm
from my_dataset import MyDataSet
from utils import read_split_data

import torch.nn.functional as F
import csv
import random
import numpy as np


book = xlwt.Workbook(encoding='utf-8')  # 创建Workbook，相当于创建Excel
# 创建sheet，Sheet1为表的名字，cell_overwrite_ok为是否覆盖单元格
sheet1 = book.add_sheet(u'Train_data', cell_overwrite_ok=True)
# 向表中添加数据
sheet1.write(0, 0, 'epoch')
sheet1.write(0, 1, 'Train_Loss')
sheet1.write(0, 2, 'Train_Acc')
sheet1.write(0, 3, 'Val_Loss')
sheet1.write(0, 4, 'Val_Acc')
sheet1.write(0, 5, 'lr')
sheet1.write(0, 6, 'Best val Acc')


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    warnings.filterwarnings("ignore")
    # loss_function = torch.nn.CrossEntropyLoss()
    #
    # model.eval()
    #
    # accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    # accu_loss = torch.zeros(1).to(device)  # 累计损失
    #
    # sample_num = 0
    # data_loader = tqdm(data_loader, file=sys.stdout)
    # for step, data in enumerate(data_loader):
    #     images, labels = data
    #     sample_num += images.shape[0]
    #
    #     pred = model(images.to(device))
    #     pred_classes = torch.max(pred, dim=1)[1]
    #     accu_num += torch.eq(pred_classes, labels.to(device)).sum()
    #
    #     loss = loss_function(pred, labels.to(device))
    #     accu_loss += loss
    #
    #     data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
    #                                                                            accu_loss.item() / (step + 1),
    #                                                                            accu_num.item() / sample_num)
    #
    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num

    '''
       model.eval()

       # 验证样本总个数
       total_num = len(data_loader.dataset)

       # 用于存储预测正确的样本个数
       sum_num = torch.zeros(1).to(device)

       data_loader = tqdm(data_loader, file=sys.stdout)

       for step, data in enumerate(data_loader):
           images, labels = data
           pred = model(images.to(device))
           pred = torch.max(pred, dim=1)[1]
           sum_num += torch.eq(pred, labels.to(device)).sum()

       return sum_num.item() / total_num
       '''
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    p = 0
    r = 0
    f1 = 0
    k = 0


    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    out = open('CoorLGNet-old.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(
        ['acc', 'precision', 'recall', 'f1-score', 'true label', 'class0', 'class1', 'pred label'])
    all_label = []
    all_pre = []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        label = labels.numpy().tolist()  # +
        pred = model(images.to(device))
        lb_pred = F.softmax(pred).cpu().numpy()  # +

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        all_label.extend(i for i in labels.tolist())
        all_pre.extend(i for i in pred_classes.cpu().numpy().tolist())


        label = labels.to(device)
        # ap = sm.average_precision_score(all_label, all_pre, average='macro')  # 2分类ap，多分类没有ap概念  average='macro'
        # mAP = compute_mAP(all_label, all_pre)
        precision_ = sm.precision_score(all_label, all_pre, average='macro')
        recall_ = sm.recall_score(all_label, all_pre, average='macro')
        f1_score_ = sm.f1_score(all_label, all_pre, average='macro')
        # acc = sm.accuracy_score(all_label, all_pre)
        p += precision_
        r += recall_
        f1 += f1_score_

        for j in range(len(lb_pred)):
            # print(lb_pred[j][0],)
            csv_write.writerow(['', '', '', '', label.cpu().numpy().tolist()[j], lb_pred[j][0], lb_pred[j][1],
                                pred_classes[j].cpu().numpy().tolist()])
        k += 1

    # 每一个epoch的平均
    print("Accuracy:", accu_num.item() / sample_num)
    acc = accu_num.item() / sample_num
    precision = p / k
    recall = r / k
    f1_score = f1 / k
    print("Precision: {:.5f}, Recall: {:.5f}, F1_score: {:.5f}".format(precision, recall, f1_score))

    # 保存指标
    out = open('CoorLGNet-old.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow([acc, precision, recall, f1_score, '', '', '', ''])


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.RandomRotation((-90, 90)),
                                     # transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "D:\pyCharmdata\Vit_myself_bu\datasets") #   数据集地址
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                      transform=data_transform["train"])
    # train_num = len(train_dataset)
    #
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)
    #
    # batch_size = 16
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn
                                             )




    # model = eca_resnet50(num_classes=args.num_classes)
    # # load pretrain weights
    # # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = args.weights
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    # # for param in net.parameters()d:
    # #     param.requires_grad = False  #是否冻结网络
    #
    # # change fc layer structure
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, args.num_classes) #数据集种类
    # model.to(device)




    model = coorlgnet(num_classes=args.num_classes).to(device)

    print("batch_size:", args.batch_size)
    print("lr:", args.lr)
    print("weight_decay:", args.weight_decay)
    print(model)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "fc" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))


    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))




    images = torch.zeros(1, 3, 224, 224).to(device)       # 要求大小与输入图片的大小一致
    tb_writer.add_graph(model, images, verbose=False)

    epochs = args.epochs       # 训练轮数
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    best_acc = 0.0
    best_acc_epoch = 0
    save_path = './weight/best.pth'
    train_steps = len(train_loader)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    for epoch in range(epochs):

        epoch_list.append(epoch)

        sheet1.write(epoch + 1, 0, epoch + 1)
        sheet1.write(epoch + 1, 5, str(optimizer.state_dict()['param_groups'][0]['lr']))

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()   # 更新学习率

        sheet1.write(epoch + 1, 1, str(train_loss))
        sheet1.write(epoch + 1, 2, str(train_acc))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        sheet1.write(epoch + 1, 3, str(val_loss))
        sheet1.write(epoch + 1, 4, str(val_acc))

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch
            # torch.save(model.state_dict(), save_path)

        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig("./loss.png")
        plt.close(1)
        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig("./acc.png")
        plt.close(2)

    sheet1.write(1, 6, str(best_acc))
    # book.save('.\Train_data.xlsx')
    print("The Best Acc = : {:.4f}".format(best_acc))
    print("The Best_acc_epoch:", best_acc_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1E-3)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./datasetold")

    # # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./best.pth',
    #                     help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()




    main(opt)

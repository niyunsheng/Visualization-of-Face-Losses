import argparse
import datetime
import json
import logging
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms

from models import LeNet_pp, MiniResnet18, AngleLinear
# from trainer import Trainer

colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999',
         '#000000']

def main():
    parser = argparse.ArgumentParser(description='face loss visualization')
    parser.add_argument('--gpus', type=str, default='0', help='avialable gpus')
    parser.add_argument('--lossname', type=str, default='softmax', help='avialable gpus')
    parser.add_argument('--save_weights', type=bool, default=True, help='if you need to save weights')
    parser.add_argument('--basemodel', type=str, default="MiniResnet18", help='basemodel architecture', choices = ['LeNet_pp', 'MiniResnet18'])
    parser.add_argument('--visual_dimension', type=int, default=2, help='encoder a image to a visual-dimensional space')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config
    with open("config.json") as f:
        cnfs = json.load(f)

    # train_loader
    train_dataset = datasets.MNIST(cnfs["data"]["src_dir"], train=True, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ]),download=True)
    trainloader = DataLoader(train_dataset,batch_size = cnfs["data"]["batch_size"],
        shuffle = cnfs["data"]["shuffle"],
        num_workers = cnfs["data"]["num_workers"]
    )
    # test_loader
    test_dataset = datasets.MNIST(cnfs["data"]["src_dir"], train=False, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ]),download=True)
    testloader = DataLoader(test_dataset,batch_size = cnfs["data"]["batch_size"],
        shuffle = cnfs["data"]["shuffle"],
        num_workers = cnfs["data"]["num_workers"]
    )
    # model
    loss_cnf = cnfs["loss"][args.lossname]
    if args.basemodel == 'LeNet_pp':
        encoder = LeNet_pp(out_features=args.visual_dimension)
    else:
        encoder = MiniResnet18(out_features=args.visual_dimension)
    encoder = encoder.to(device)
    angleLinear = AngleLinear(
                                in_features=args.visual_dimension, 
                                out_features=cnfs["data"]["num_classes"],
                                w_norm=loss_cnf["w_norm"],
                                x_norm=loss_cnf["x_norm"],
                                s=loss_cnf["s"],
                                m1=loss_cnf["m1"],
                                m2=loss_cnf["m2"],
                                m3=loss_cnf["m3"],
                                device=device
                                )
    angleLinear = angleLinear.to(device)
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam([
                {'params': encoder.parameters()},
                {'params': angleLinear.parameters()} ], 
                lr=cnfs["lr"])
    # make dirs
    logging_dir = cnfs["training"]["log_dir"]
    os.makedirs(logging_dir,exist_ok=True)
    checkpoints_dir = os.path.join(cnfs["training"]["checkpoints_dir"],args.lossname)
    os.makedirs(cnfs["training"]["checkpoints_dir"],exist_ok=True)
    os.makedirs(checkpoints_dir ,exist_ok=True)
    # logging
    now = datetime.datetime.now().strftime("%m-%d_%H:%M")
    logging_file = os.path.join(logging_dir, 
                        '{}_{}.txt'.format(args.lossname, now))
    logging.basicConfig(level=logging.INFO,
                    filename=logging_file,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)

    def L2_norm(x):
        assert len(x.shape)==2
        return x/np.sqrt(np.sum(x*x,axis=1)).reshape(x.shape[0],-1)

    def visualization(features, labels, w, title):
        features = features[labels>=0] # label=-1 means predtion error
        labels = labels[labels>=0]
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        if loss_cnf["x_norm"]:
            features = L2_norm(features)
        if loss_cnf["w_norm"]:
            w = L2_norm(w)
        for i in range(0,10):
            x_i = features[:,0][labels==i]
            y_i = features[:,1][labels==i]
            plt.scatter(x_i,y_i,s=1,c=colors[i])
        plt.legend(['num_'+str(i) for i in range(10)], loc = 'upper right')
        for i in range(10):
            plt.annotate(str(i),xy=(0,0),xytext=(w[i][0],w[i][1]),arrowprops=dict(arrowstyle="<-",connectionstyle="arc3"))
        plt.title(title)
        plt.savefig(os.path.join('weights', args.lossname, title + '.png'))

    # clip_gradient
    def clip_gradient(optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    # training
    def train_epoch(epoch_id):
        encoder.train()
        angleLinear.train()
        losses = []
        total_loss = 0
        correct = 0
        total = 0
        plot_features = []
        plot_labels = []
        for batch_idx, (data, labels) in enumerate(trainloader):
            data = data.to(device)
            labels = labels.to(device)

            features = encoder(data)
            # append features
            plot_features.append(features.cpu().detach().numpy())
            outputs = angleLinear(features, labels)
            _, predicted = outputs.max(1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            losses.append(loss.item())
            total_loss += loss.item()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            clip_gradient(optimizer, cnfs["grad_clip"])
            optimizer.step()

            # append except features
            labels = labels.cpu().detach().numpy()
            predicted = predicted.cpu().detach().numpy()
            labels[labels!=predicted] = -1 
            plot_labels.append(labels)

            # logging message
            if batch_idx % cnfs["training"]["log_interval"] == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tAccuracy: {:.6f}%\tLoss: {:.6f}'.format(
                        total, len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), 
                        100.*correct/total,
                        np.mean(losses))
                print(message)
                losses = []
        total_loss /= (batch_idx+1)
        message = 'Train: {}/{} Accuracy: {:.6f}\tLoss: {:.6f}'.format(
                        epoch_id,
                        cnfs["training"]["n_epoch"],
                        100.*correct/total,
                        total_loss)
        print(message)
        logger.info(message)

        # plot and save feature distribution
        title = '{}_train_epoch{:02}'.format(args.lossname, epoch_id)
        w = angleLinear.weight.cpu().detach().numpy()
        plot_features = np.concatenate(plot_features)
        plot_labels = np.concatenate(plot_labels)
        # visualization(plot_features, plot_labels, w, title)

    best_acc = 0
    def test_epoch(epoch_id,visual=False):
        nonlocal best_acc
        encoder.eval()
        angleLinear.eval()
        val_loss = 0
        correct = 0
        total = 0
        plot_features = []
        plot_labels = []
        with torch.no_grad():
            for batch_idx, (data, labels) in tqdm(enumerate(testloader)):
                data = data.to(device)
                labels = labels.to(device)
                features = encoder(data)
                # append features
                plot_features.append(features.cpu().detach().numpy())
                outputs = angleLinear(features)
                _, predicted = outputs.max(1)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

                val_loss += loss.item()

                # append except features
                labels = labels.cpu().detach().numpy()
                predicted = predicted.cpu().detach().numpy()
                labels[labels!=predicted] = -1 
                plot_labels.append(labels)
        message = 'Val: {}/{} Accuracy: {:.6f}\tLoss: {:.6f}'.format(
                        epoch_id,
                        cnfs["training"]["n_epoch"],
                        100.*correct/total,
                        val_loss/(batch_idx+1))
        print(message)
        logger.info(message)
        # Save checkpoint.
        acc = 100.*correct/total

        if args.save_weights and best_acc < acc or epoch_id==cnfs["training"]["n_epoch"]-1:
            print('Saving..')
            state = {
                'encoder': encoder.state_dict(),
                'angleLinear':angleLinear.state_dict(),
                'acc': acc,
                'epoch':epoch_id
            }
            torch.save(state, os.path.join(checkpoints_dir,'{}_epoch_{}_acc_{}.pth'.format(args.lossname,epoch_id,acc)))
            best_acc = acc

            # plot and save feature distribution
            title = '{}_test_epoch_{:02}'.format(args.lossname, epoch_id)
            print(title)
            w = angleLinear.weight.cpu().detach().numpy()
            plot_features = np.concatenate(plot_features)
            plot_labels = np.concatenate(plot_labels)
            visualization(plot_features, plot_labels, w, title)

    for epoch_id in range(0,cnfs["training"]["n_epoch"]):
        print('*'*10,' epoch ',epoch_id,'\t','*'*10,)
        train_epoch(epoch_id)
        test_epoch(epoch_id)

if __name__ == '__main__':
    main()
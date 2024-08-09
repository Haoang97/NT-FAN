import argparse
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
from dataloader import get_dataloader, sample_data, create_target_samples, sample_groups
from utils import seed_torch, AverageMeter, LabelSmoothingCrossEntropy
from models.main_models import Classifier, DCD, Encoder, resnet18, resnet50
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def exclude_bias_and_norm(p):
    return p.ndim == 1

parser = argparse.ArgumentParser(description='WFDA')
parser.add_argument('--n_epochs_1',type=int,default=1000)
parser.add_argument('--n_epochs_2',type=int,default=50)
parser.add_argument('--n_epochs_3',type=int,default=500)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--batch_size',type=int,default=256)
parser.add_argument('--task', type=str, default='visda')
parser.add_argument('--noise_type', type=str, default='pairflip')
parser.add_argument('--noise_rate', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=0)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--reload', type=bool, default=False)
parser.add_argument('--restart_epoch', type=int, default=1000)
parser.add_argument('--output_to_log', type=bool, default=False)
parser.add_argument('--shuffle', type=bool, default=True)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
seed_torch(args.seed)

if args.output_to_log == True:
    logfile = './log/' + args.task + '_' + args.noise_type + '_' + str(args.n_target_samples) + '.log'
    sys.stdout = open(logfile, mode='w', encoding='utf-8')

print('========= load data =========')
train_dataloader, test_dataloader = get_dataloader(args)

print('========= load model =========')
if args.task in ['m2s', 'm2u', 'u2s', 'u2m']:
    encoder = resnet18(channel=1)
    classifier_pre = Classifier(encoder.fc.in_features, args.num_classes)
    classifier = Classifier(encoder.fc.in_features, args.num_classes)
    discriminator = DCD(encoder.fc.in_features)
elif args.task in ['s2m', 's2u']:
    encoder = resnet18()
    classifier_pre = Classifier(encoder.fc.in_features, args.num_classes)
    classifier = Classifier(encoder.fc.in_features, args.num_classes)
    discriminator = DCD(encoder.fc.in_features)
elif args.task in ['i2p-ic', 'p2i-ic', 'i2c-ic', 'c2i-ic', 'c2p-ic', 'p2c-ic']:
    encoder = resnet18()
    classifier_pre = Classifier(512, args.num_classes)
    classifier = Classifier(512, args.num_classes)
    discriminator = DCD(512, 4)
elif args.task in ['visda', 'p2r-dn', 'r2p-dn', 'p2s-dn', 's2p-dn', 'r2s-dn', 's2r-dn']:
    encoder = resnet50()
    classifier_pre = Classifier(2048, args.num_classes)
    classifier = Classifier(2048, args.num_classes)
    discriminator = DCD(2048, 4)
else:
    raise NotImplementedError

classifier_pre = classifier_pre.to(device)
classifier = classifier.to(device)
encoder = encoder.to(device)
discriminator = discriminator.to(device)

loss_fn = torch.nn.CrossEntropyLoss().to(device) 
loss_ls = LabelSmoothingCrossEntropy(0.05).to(device)

#--------------prepare g and h for step 1---------------------------------
if args.reload == True:
    print('========= reload g and h for step 1 =========')
    if args.task[0] == "p":
        task = "p2i-ic"
    elif args.task[0] == "c":
        task = "c2i-ic"
    elif args.task[0] == "i":
        task = "i2c-ic"
    elif args.task[0] == "m":
        task = "m2s"
    elif args.task[0] == "s":
        task = "s2m"
    elif args.task[0] == "u":
        task = "u2m"
    else:
        task=args.task

    ckpt_encoder = torch.load('./pretrain/Encoder/ckpt_{}_{}_{}_{}.pt'.format(task, args.noise_type, args.noise_rate, args.restart_epoch), map_location=device)
    ckpt_classifier_pre = torch.load('./pretrain/Classifier/ckpt_{}_{}_{}_{}.pt'.format(task, args.noise_type, args.noise_rate, args.restart_epoch), map_location=device)
    encoder.load_state_dict(ckpt_encoder)
    classifier_pre.load_state_dict(ckpt_classifier_pre)

else:
    print('========= pretrain g and h for step 1 =========')
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(classifier_pre.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    encoder.train()
    classifier_pre.train()
    for epoch in range(args.n_epochs_1):
        loss_rec = AverageMeter()
        acc = 0
        for step, (data, labels, _) in enumerate(train_dataloader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            with autocast():
                feat = encoder(data)
                y_pred = classifier_pre(feat)
                loss = loss_fn(y_pred, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_rec.update(loss.item(), data.size(0))
            acc += (torch.max(y_pred,1)[1]==labels).float().mean().item()
            if step % 50 == 0:
                print(f"[Epoch:{epoch}/{args.n_epochs_1} [Step:{len(train_dataloader)}/{step}] [Loss:{loss.item()}]")

        accuracy = round(acc / float(len(train_dataloader)), 6)
        print('epoch {}, lr={:.7f}, loss={:.7f}, accuracy={:.4f}'
                    .format(epoch+1, optimizer.state_dict()['param_groups'][0]['lr'], loss_rec.avg, accuracy))
        if accuracy > 0.95:
            torch.save(encoder.state_dict(), './pretrain/Encoder/ckpt_{}_{}_{}_{}.pt'.format(args.task, args.noise_type, args.noise_rate, epoch+1))
            torch.save(classifier_pre.state_dict(), './pretrain/Classifier/ckpt_{}_{}_{}_{}.pt'.format(args.task, args.noise_type, args.noise_rate, epoch+1))
            break

        if (epoch + 1) % 100 == 0:
            torch.save(encoder.state_dict(), './pretrain/Encoder/ckpt_{}_{}_{}_{}.pt'.format(args.task, args.noise_type, args.noise_rate, epoch+1))
            torch.save(classifier_pre.state_dict(), './pretrain/Classifier/ckpt_{}_{}_{}_{}.pt'.format(args.task, args.noise_type, args.noise_rate, epoch+1))

    torch.save(encoder.state_dict(), './pretrain/Encoder/ckpt_{}_{}_{}_{}.pt'.format(args.task, args.noise_type, args.noise_rate, args.n_epochs_1))
    torch.save(classifier_pre.state_dict(), './pretrain/Classifier/ckpt_{}_{}_{}_{}.pt'.format(args.task, args.noise_type, args.noise_rate, args.n_epochs_1))

acc = 0
encoder.eval()
classifier_pre.eval()
with torch.no_grad():
    for data, labels, _ in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)
        y_test_pred = classifier_pre(encoder(data))
        acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
wa_acc = round(acc / float(len(test_dataloader)), 3)
print(f"Target Acc without adaptation: {wa_acc}")

#-------------------------------------------------------------------
X_s,Y_s = sample_data(args, train_dataloader.dataset)
X_t,Y_t = create_target_samples(args, test_dataloader.dataset)

#-----------------train DCD for step 2--------------------------------
optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=0.1)

for epoch in range(args.n_epochs_2):
    # data
    groups,aa = sample_groups(X_s,Y_s,X_t,Y_t,seed=epoch)

    n_iters = 4 * len(groups[1])
    index_list = torch.randperm(n_iters)
    mini_batch_size=40 #use mini_batch train can be more stable


    loss_mean=[]

    X1=[];X2=[];ground_truths=[]

    discriminator.train()
    encoder.eval()
    for index in range(n_iters):

        ground_truth=index_list[index]//len(groups[1])

        x1,x2=groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]

        X1.append(x1)
        X2.append(x2)
        ground_truths.append(ground_truth)

        #select data for a mini-batch to train
        if (index+1)%mini_batch_size==0:
            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths=torch.LongTensor(ground_truths)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths=ground_truths.to(device)

            optimizer_d.zero_grad()
            
            X_cat=torch.cat([encoder(X1),encoder(X2)],1)
            y_pred=discriminator(X_cat.detach())
            loss=loss_fn(y_pred,ground_truths)
            loss.backward()
            optimizer_d.step()
            loss_mean.append(loss.item())
            X1 = []
            X2 = []
            ground_truths = []

    print("step2----Epoch %d/%d loss:%.3f"%(epoch+1, args.n_epochs_2, np.mean(loss_mean)))

#----------------------------------------------------------------------

#-------------------training for step 3-------------------
optimizer_g_h = torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler_gh = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g_h, T_max=args.n_epochs_3, eta_min=5e-6)

for epoch in range(args.n_epochs_3):
    loss_gh_rec = AverageMeter()
    #---training g and h , DCD is frozen

    groups, groups_y = sample_groups(X_s,Y_s,X_t,Y_t,seed=args.n_epochs_2+epoch)
    G1, G2, G3, G4 = groups
    Y1, Y2, Y3, Y4 = groups_y
    groups_2 = [G2, G4]
    groups_y_2 = [Y2, Y4]

    n_iters = 2 * len(G2)
    index_list = torch.randperm(n_iters)

    n_iters_dcd = 4 * len(G2)
    index_list_dcd = torch.randperm(n_iters_dcd)

    mini_batch_size_g_h = 20 #data only contains G2 and G4 ,so decrease mini_batch
    mini_batch_size_dcd= 40 #data contains G1,G2,G3,G4 so use 40 as mini_batch
    X1 = []
    X2 = []
    ground_truths_y1 = []
    ground_truths_y2 = []
    dcd_labels=[]

    encoder.train()
    classifier.train()
    discriminator.eval()
    for index in range(n_iters):


        ground_truth=index_list[index]//len(G2)
        x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
        
        dcd_label=0 if ground_truth==0 else 2

        X1.append(x1)
        X2.append(x2)
        ground_truths_y1.append(y1)
        ground_truths_y2.append(y2)
        dcd_labels.append(dcd_label)

        if (index+1)%mini_batch_size_g_h==0:

            X1=torch.stack(X1)
            X2=torch.stack(X2)
            ground_truths_y1=torch.LongTensor(ground_truths_y1)
            ground_truths_y2 = torch.LongTensor(ground_truths_y2)
            dcd_labels=torch.LongTensor(dcd_labels)
            X1=X1.to(device)
            X2=X2.to(device)
            ground_truths_y1=ground_truths_y1.to(device)
            ground_truths_y2 = ground_truths_y2.to(device)
            dcd_labels=dcd_labels.to(device)

            optimizer_g_h.zero_grad()

            encoder_X1=encoder(X1)
            encoder_X2=encoder(X2)

            X_cat=torch.cat([encoder_X1,encoder_X2],1)
            y_pred_X1=classifier(encoder_X1)
            y_pred_X2=classifier(encoder_X2)
            y_pred_dcd=discriminator(X_cat)

            loss_X1=loss_fn(y_pred_X1,ground_truths_y1)
            loss_X2=loss_fn(y_pred_X2,ground_truths_y2)
            loss_dcd=loss_fn(y_pred_dcd,dcd_labels)

            loss_sum = loss_X1 + loss_X2 + 0.2 * loss_dcd

            loss_gh_rec.update(loss_sum.item(), X_cat.size(0))

            loss_sum.backward()
            optimizer_g_h.step()

            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []


    #----training dcd ,g and h frozen
    X1 = []
    X2 = []
    ground_truths = []

    discriminator.train()
    encoder.eval()
    classifier.eval()
    for index in range(n_iters_dcd):

        ground_truth=index_list_dcd[index]//len(groups[1])

        x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
        
        X2.append(x2)
        ground_truths.append(ground_truth)

        if (index + 1) % mini_batch_size_dcd == 0:
            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            ground_truths = torch.LongTensor(ground_truths)
            X1 = X1.to(device)
            X2 = X2.to(device)
            ground_truths = ground_truths.to(device)

            optimizer_d.zero_grad()
            X_cat = torch.cat([encoder(X1), encoder(X2)], 1)
            y_pred = discriminator(X_cat.detach())
            loss = loss_fn(y_pred, ground_truths)
            loss.backward()
            optimizer_d.step()
            X1 = []
            X2 = []
            ground_truths = []

    lr_scheduler_gh.step()

    #testing
    acc = 0
    with torch.no_grad():
        for data, labels, _ in test_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            y_test_pred = classifier(encoder(data))
            acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    accuracy = round(acc / float(len(test_dataloader)), 3)

    print("step3----Epoch %d/%d  lr: %.4f  loss: %.4f  accuracy: %.3f " % (epoch + 1, args.n_epochs_3, optimizer_g_h.state_dict()['param_groups'][0]['lr'], loss_gh_rec.avg, accuracy))
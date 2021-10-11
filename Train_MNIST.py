# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attacktrain', default='cent',type=str,choices=['cent','cw'])
    parser.add_argument('--epsilon', default=0.4, type=float)
    return parser.parse_args()



mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

class Interpolate(nn.Module):
    def forward(self, x,shape=(28,28)):
        return F.interpolate(x,size=shape)   

model_cnn_robust = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)

img_mask=nn.Sequential(nn.Conv2d(1,64,3,padding=1,stride=2),nn.LeakyReLU(0.2),
                    nn.Conv2d(64,128,3,padding=1,stride=1),nn.LeakyReLU(0.2),
                    nn.Conv2d(128,128,3,padding=1,stride=1),nn.LeakyReLU(0.2),
                    Interpolate(),
                    nn.Conv2d(128,1,3,padding=1),nn.Sigmoid()).to(device)

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter=20, randomize=True):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * 0.1 - 0.1
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        yp=model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
def pgd_linf_cw(model, X, y, epsilon=0.3, alpha=0.01, num_iter=20, randomize=True):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * 0.1 - 0.1
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        yp=model(X + delta)
        loss = cwloss(yp,y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach() 


def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def cwloss(output, target,confidence=150, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss
def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    mask_loss_total=0.0
    global opt1
    kl = nn.KLDivLoss(size_average=False)
    for X,y in loader:
        
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        batch_size=X.shape[0]
        input_mask=X+delta
        mask=img_mask(input_mask)
        delta_new=delta*mask

        yp_nat=model(X)
        yp_nat_prop=F.softmax(yp_nat,dim=1)
        true_probs=torch.gather(yp_nat_prop, 1, (y.unsqueeze(1)).long()).squeeze()

        if opt:
            yp = model(X+delta_new)
        else:
            yp=model(X+delta)
                
        loss_robust=(1.0 / batch_size)*kl(F.log_softmax(yp, dim=1),F.softmax(yp_nat, dim=1))
        
        if opt:
            loss=loss_robust+(1.0/batch_size)*torch.sum(nn.CrossEntropyLoss(reduction="none")(yp_nat,y)*(1.0000001 - true_probs))

            opt.zero_grad()
            loss.backward()
            opt.step()
        loss=loss_robust
        if opt:
            mask=img_mask(input_mask)
            delta_new=delta.clone()
            delta_new=delta*mask
            yp_adv=model(X+delta)
            yp=model(X+delta_new)
            adv_probs=F.softmax(yp_adv,dim=1)
            probs=F.softmax(yp,dim=1)
            temp_loss=(1.0 / batch_size)*kl(F.log_softmax(yp, dim=1),F.softmax(yp_adv, dim=1))
            temp_loss1=nn.CrossEntropyLoss()(yp,y)
            loss_robust=temp_loss+0.3*temp_loss1
            opt1.zero_grad()
            loss_robust.backward()
            opt1.step()
              
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


torch.cuda.manual_seed(0)
opt = optim.Adam(list(model_cnn_robust.parameters()), lr=1e-3)
opt1=optim.Adam(img_mask.parameters(),lr=1e-3)
args = get_args()
best=100
print("epoch    train_err    test_err    adv_err")
for t in range(40):
    if args.attacktrain=="cent":
        train_err, train_loss = epoch_adversarial(train_loader, model_cnn_robust, pgd_linf,opt,epsilon=args.epsilon,alpha=args.epsilon/20,num_iter=20,randomize=True)
    elif args.attacktrain=="cw":
        train_err,train_loss=epoch_adversarial(train_loader,model_cnn_robust,pgd_linf_cw,opt,epsilon=args.epsilon,alpha=args.epsilon/20,num_iter=20,randomize=True)
    else:
        print("wrong in training attack setting!")
   
    test_err, test_loss = epoch(test_loader, model_cnn_robust)
    adv_err, adv_loss = epoch_adversarial(test_loader, model_cnn_robust, pgd_linf,epsilon=0.3, alpha=0.015,randomize=False)
    if best>(test_err+adv_err):
      torch.save(model_cnn_robust.state_dict(), "./model_cnn_robust_e4_CW_new_repeat3_test.pt")
      torch.save(img_mask.state_dict(),"./img_mask_e4_CW_new_repeat3_test.pt")
      best=test_err+adv_err
    if t == 30:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-4
        for param_group in opt1.param_groups:
            param_group["lr"] = 1e-4
            
    print(*("{:.4f}".format(i) for i in (t,train_err, test_err, adv_err)), sep="\t",flush=True)



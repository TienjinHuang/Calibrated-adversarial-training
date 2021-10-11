import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms
from models import *
import attack_generator as attack
from wideresnet_trades import WideResNet_trades
parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="WRN", help="decide which network to use,choose from smallcnn,resnet18,WRN,WRN_trades")
parser.add_argument('--dataset', type=str, default="cifar10",choices=['cifar10','cifar100'], help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10,help='WRN width factor')
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--attack_method', type=str,default="dat", help = "choose form: dat and trades")
parser.add_argument('--model_path', default='./FAT_models/fat_for_trades_wrn34-10_eps0.031_beta1.0.pth.tar', help='model for white-box attack evaluation')
parser.add_argument('--preprocess', type=str, default='meanstd',
                    choices=['meanstd', '01', '+-1'], help='The preprocess for data')

                    
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

if args.preprocess == 'meanstd':
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    elif args.dataset == 'cifar100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
elif args.preprocess == '01':
    mean = (0, 0, 0)
    std = (1, 1, 1)
elif args.preprocess == '+-1':
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
else:
    raise ValueError('Please use valid parameters for normalization.')

print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
#if args.dataset == "svhn":
#    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
#    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset=="cifar100":
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('==> Load Model')
if args.net == "resnet18":
    if args.dataset=="cifar100":
        model = PreActResNet18(num_classes=100).cuda()
    else:
        model = PreActResNet18().cuda()
    net = "resnet18"

if args.net == "WRN":
    ## WRN-34-10
    if args.dataset=="cifar100":
        model=WideResNet(depth=args.depth, num_classes=100, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    else:
        model = WideResNet(depth=args.depth, num_classes=10, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    
    
    net = "WRN{}-{}-dropout{}".format(args.depth,args.width_factor,args.drop_rate)

#if args.load=="1":
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.model_path)['state_dict'])
model = nn.Sequential(Normalize(mean=mean, std=std), model).cuda()
'''
elif args.load=="2":
    if args.flag=="2":    
        model = torch.nn.DataParallel(model)
    s=torch.load(args.model_path)
    print(s.keys())
    print(model.state_dict().keys())
    model.load_state_dict(s)
    model = nn.Sequential(Normalize(mean=mean, std=std), model).cuda()
'''
print(net)



print('==> Evaluating Performance under White-box Adversarial Attack')

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))
loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=8./255, step_size=8./255,loss_fn="cent", category="Madry",rand_init=False)
print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
loss, pgd20_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=8./255, step_size=0.003,loss_fn="cent",category="Madry",rand_init=False)
print('PGD20 without Random Start Test Accuracy: {:.2f}%'.format(100. * pgd20_wori_acc))
loss, pgd100_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=100, epsilon=8./255, step_size=0.003,loss_fn="cent",category="Madry",rand_init=False)
print('PGD100 without Random Start Test Accuracy: {:.2f}%'.format(100. * pgd100_wori_acc))
loss, cw_wri_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=0.031, step_size=0.003,loss_fn="cw",category="Madry",rand_init=True)
print('CW with Random Start Test Accuracy: {:.2f}%'.format(100. * cw_wri_acc))

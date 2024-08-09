import time
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.mnist import MNIST
from data.svhn import SVHN
from data.usps import USPS
from data.utils import reform_Ctrain, reform_Strain, ResizeImage, PlaceCrop
from data_list import ImageList

def get_dataloader(args):
    if args.task == 'm2s':
        train_loader = DataLoader(
            dataset=MNIST('/data/chihaoang/datasets/MNIST',train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]), noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=args.random_state),
            batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_loader = DataLoader(
            dataset=SVHN('/data/chihaoang/datasets/SVHN', split='test', download=False,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,], [0.229,])
                        ]), noise_type=None),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.task == 'm2u':
        train_loader = train_loader = DataLoader(
            dataset=MNIST('/data/chihaoang/datasets/MNIST',train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]), noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=args.random_state),
            batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_loader = DataLoader(
            dataset=USPS('/data/chihaoang/datasets/USPS', train=False, download=False,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]), noise_type=None),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.task == 's2m':
        train_loader = DataLoader(
            dataset=SVHN('/data/chihaoang/datasets/SVHN', split='train', download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]), noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=args.random_state),
            batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_loader = DataLoader(
            dataset=MNIST('/data/chihaoang/datasets/MNIST',train=False, download=False,
                   transform=transforms.Compose([
                       transforms.Resize((32,32)),
                       transforms.Lambda(lambda x: x.convert("RGB")),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),noise_type=None),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.task == 's2u':
        train_loader = DataLoader(
            dataset=SVHN('/data/chihaoang/datasets/SVHN', split='train', download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]), noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=args.random_state),
            batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_loader = DataLoader(
            dataset=USPS('/data/chihaoang/datasets/USPS', train=False, download=False,
                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ]), noise_type=None),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.task == 'u2m':
        train_loader = DataLoader(
            dataset=USPS('/data/chihaoang/datasets/USPS', train=True, download=False,
                    transform=transforms.Compose([
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]), noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=args.random_state),
            batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_loader = DataLoader(
            dataset=MNIST('/data/chihaoang/datasets/MNIST',train=False, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]), noise_type=None),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.task == 'u2s':
        train_loader = DataLoader(
            dataset=USPS('/data/chihaoang/datasets/USPS', train=True, download=False,
                    transform=transforms.Compose([
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]), noise_type=args.noise_type, noise_rate=args.noise_rate, random_state=args.random_state),
            batch_size=args.batch_size, shuffle=args.shuffle
        )
        test_loader = DataLoader(
            dataset=SVHN('/data/chihaoang/datasets/SVHN', split='test', download=False,
                    transform=transforms.Compose([
                        transforms.Resize((28,28)),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ]), noise_type=None),
            batch_size=args.batch_size, shuffle=False
        )
    
    elif args.task == 'a2d':
        train_loader = DataLoader(
            ImageList(open("./data/office/amazon_list.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office/dslr_list.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.task == 'a2w':
        train_loader = DataLoader(
            ImageList(open("./data/office/amazon_list.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office/webcam_list.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.task == 'd2a':
        train_loader = DataLoader(
            ImageList(open("./data/office/dslr_list.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office/amazon_list.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.task == 'd2w':
        train_loader = DataLoader(
            ImageList(open("./data/office/dslr_list.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office/webcam_list.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.task == 'w2a':
        train_loader = DataLoader(
            ImageList(open("./data/office/webcam_list.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office/amazon_list.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.task == 'w2d':
        train_loader = DataLoader(
            ImageList(open("./data/office/webcam_list.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office/dslr_list.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

    elif args.task == 'a2c':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Art.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Clipart.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'a2p':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Art.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Product.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'a2r':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Art.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Real_World.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'c2a':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Clipart.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Art.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'c2p':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Clipart.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Product.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'c2r':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Clipart.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Real_World.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'p2a':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Product.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Art.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'p2c':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Product.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Clipart.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'p2r':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Product.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Real_World.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'r2a':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Real_World.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Art.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'r2c':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Real_World.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Clipart.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    
    elif args.task == 'r2p':
        train_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Real_World.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=65, \
                transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("./data/office-home-resized/Product.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
        
    elif args.task == 'visda':
        train_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/VisDA-C/train_list.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=12, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/VisDA-C/validation_list.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

    elif args.task == 'i2p-ic':
        train_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/iList.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=12, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/pList.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

    elif args.task == 'p2i-ic':
        train_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/pList.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=12, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/iList.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

    elif args.task == 'i2c-ic':
        train_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/iList.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=12, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/cList.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
        
    elif args.task == 'c2i-ic':
        train_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/cList.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=12, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/iList.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

    elif args.task == 'c2p-ic':
        train_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/cList.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=12, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/pList.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

    elif args.task == 'p2c-ic':
        train_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/pList.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=12, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=8, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/data/chihaoang/datasets/image-clef/list/cList.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.task == 'p2r-dn':
        train_loader = DataLoader(
            ImageList(open("/home/user/chihaoang/datasets/DomainNet/painting_train.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=345, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/home/user/chihaoang/datasets/DomainNet/real_test.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
    elif args.task == 'r2p-dn':
        train_loader = DataLoader(
            ImageList(open("/home/user/chihaoang/datasets/DomainNet/real_train.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=345, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/home/user/chihaoang/datasets/DomainNet/painting_test.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
    elif args.task == 's2p-dn':
        train_loader = DataLoader(
            ImageList(open("/home/user/chihaoang/datasets/DomainNet/sketch_train.txt").readlines(), noisy=True, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_state=args.random_state, num_class=345, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=args.shuffle, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            ImageList(open("/home/user/chihaoang/datasets/DomainNet/painting_test.txt").readlines(), noisy=False, \
                transform=transforms.Compose([
                            transforms.Resize(256),
                            PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
    return train_loader, test_loader

def sample_data(args, dataset):
    n = len(dataset)
    c, h, w = dataset[0][0].size()

    X = torch.Tensor(n, c, h, w)
    Y = torch.LongTensor(n)
    if args.noise_type is not None:
        noise_or_not = dataset.noise_or_not
    else:
        noise_or_not = torch.zeros([n], dtype=torch.int)
    inds = torch.randperm(len(dataset))
    #noise_or_not = np.arrange(noise_or_not, inds)
    #noise_or_not = noise_or_not[list(inds)]
    for i, index in enumerate(inds):
        x, y, idx = dataset[index]
        X[i] = x
        Y[i] = y
    return X, Y

def create_target_samples(args, dataset): 
    X,Y=[],[]
    classes=args.num_classes*[args.n_target_samples]

    i=0
    #sample_idx = np.random.randint(0,len(dataset),1000)
    while True:
        if len(X)==args.n_target_samples*args.num_classes:
            break
        x, y, _ =dataset[i]
        #x,y=dataset[sample_idx[i]]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
        i+=1

    assert (len(X)==args.n_target_samples*args.num_classes)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))

def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):


    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)

def create_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    n=X_t.shape[0] #class_num*shot

    #shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]

    class_num=classes.shape[0]
    shot=n//class_num
    
    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix=torch.stack(source_idxs)
    target_matrix=torch.stack(target_idxs)

    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]


    for i in range(class_num):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            if shot > 1:
                G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
                Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            else:
                G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i]]))
                Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i]]))
            G3.append((X_s[source_matrix[i%class_num][j]],X_s[source_matrix[(i+1)%class_num][j]]))
            Y3.append((Y_s[source_matrix[i % class_num][j]], Y_s[source_matrix[(i + 1) % class_num][j]]))
            if shot > 1:
                G4.append((X_s[source_matrix[i%class_num][j]],X_t[target_matrix[(i+1)%class_num][j]]))
                Y4.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num][j]]))
            else:
                G4.append((X_s[source_matrix[i%class_num][j]],X_t[target_matrix[(i+1)%class_num]]))
                Y4.append((Y_s[source_matrix[i % class_num][j]], Y_t[target_matrix[(i + 1) % class_num]]))
    

    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]

    #make sure we sampled enough samples
    for g in groups:
        assert(len(g)==n)
    return groups,groups_y
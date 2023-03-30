import logging

import torch
from PIL import Image

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import DatasetGrocery,DatasetGroceryCoarse,DatasetFreiburg,DatasetFreiburgSWAP,DatasetGrocerySWAP,DatasetGrocerySplit,CUB,DatasetProducts10k

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()




    if args.dataset == "grocery_store":
        # train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.RandomCrop((448, 448)),
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.CenterCrop((448, 448)),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # trainset=DatasetGrocerySplit(train=True,split=args.splitnum,transform=train_transform)
        # testset=DatasetGrocerySplit(train=False,split=args.splitnum,transform=test_transform)
        if args.swap:
            trainset=DatasetGrocerySWAP(csv_file_path="/train.txt",train=True,transform=train_transform)
            testset=DatasetGrocerySWAP(csv_file_path="/test.txt",train=False,transform=test_transform)
        else:
            trainset=DatasetGrocery(csv_file_path="/train.txt",transform=train_transform)
            testset=DatasetGrocery(csv_file_path="/test.txt",transform=test_transform)


    if args.dataset == "grocery_store_c":
        train_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset=DatasetGroceryCoarse(csv_file_path="/train.txt",transform=train_transform)
        testset=DatasetGroceryCoarse(csv_file_path="/test.txt",transform=test_transform)
        
    if args.dataset == "freiburg":
        train_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.RandomCrop((448, 448)),
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.CenterCrop((448, 448)),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # if args.swap:
        #     trainset=DatasetFreiburgSWAP(transform=train_transform,train=True,split=args.splitnum)
        #     testset=DatasetFreiburgSWAP(transform=test_transform,train=False,split=args.splitnum)
        # else:
        trainset=DatasetFreiburg(transform=train_transform,split=args.splitnum,train=True)
        testset=DatasetFreiburg(transform=test_transform,split=args.splitnum,train=False)

    if args.dataset == "meituan":
        train_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.RandomCrop((448, 448)),
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.CenterCrop((448, 448)),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset=datasets.ImageFolder(root="/root/datasets/Meituan/Train",transform=train_transform)
        testset=datasets.ImageFolder(root="/root/datasets/Meituan/Val",transform=test_transform)
    if args.dataset == "products10k":
        train_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.RandomCrop((448, 448)),
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
        #                             transforms.CenterCrop((448, 448)),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset=DatasetProducts10k(train=True,transform=train_transform)
        testset=DatasetProducts10k(train=False,transform=test_transform)

    if args.dataset == 'CUB_200_2011':
            # train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
            #                         transforms.RandomCrop((448, 448)),
            #                         transforms.RandomHorizontalFlip(),
            #                         transforms.ToTensor(),
            #                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
            #                         transforms.CenterCrop((448, 448)),
            #                         transforms.ToTensor(),
            #                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                transforms.RandomCrop((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                transforms.CenterCrop((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(is_train=True, transform=train_transform)
        testset = CUB(is_train=False, transform = test_transform)
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((args.img_size, args.img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    # if args.dataset == "cifar10":
    #     trainset = datasets.CIFAR10(root="./data",
    #                                 train=True,
    #                                 download=True,
    #                                 transform=transform_train)
    #     testset = datasets.CIFAR10(root="./data",
    #                                train=False,
    #                                download=True,
    #                                transform=transform_test) if args.local_rank in [-1, 0] else None

    # else:
    #     trainset = datasets.CIFAR100(root="./data",
    #                                  train=True,
    #                                  download=True,
    #                                  transform=transform_train)
    #     testset = datasets.CIFAR100(root="./data",
    #                                 train=False,
    #                                 download=True,
    #                                 transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

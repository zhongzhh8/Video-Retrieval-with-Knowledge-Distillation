import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
from data_loader import CustomDataset
from model import C3D_Hash_Model
from triplet_loss import TripletLoss
import time
import os
from utils import *




def load_data(opt,root_folder, fpath_label, batch_size, shuffle=True, num_workers=16, train=False,num_frames=32):
    if train:
        transform = transforms.Compose([
            # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),  # Center
            transforms.RandomHorizontalFlip(),  # 训练集才需要做这一步处理，获得更多的随机化数据
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])
    else:
        transform = transforms.Compose([
            # transforms.ToPILImage(),#Converts a Tensor or a numpy of shape H x W x C to a PIL Image C x H x W
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),  # Center
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.405, 0.378), std=(0.152, 0.149, 0.157))])

    data_ = CustomDataset(opt,
                          root_folder=root_folder,
                          fpath_label=fpath_label,
                          transform=transform,
                          num_frames=num_frames)

    # torch.utils.data.DataLoader
    loader_ = data.DataLoader(
        dataset=data_,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=shuffle,  # shuffle
        num_workers=num_workers)  # multi thread

    return loader_


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def Pure(opt,train_loader,test_loader,db_loader,checkpoint_path):
    print('===start setting network and optimizer===')
    device = torch.device("cuda:"+opt.cudaid if torch.cuda.is_available() else "cpu")  # device configuration
    triplet_loss = TripletLoss(opt.margin, device).to(device)

    net = C3D_Hash_Model(opt.hash_length)
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=opt.deviceid)  # for multi gpu
    if opt.load_model:
        net.load_state_dict(torch.load(opt.load_model_path))
        print('loaded model from ' + opt.load_model_path)
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_lr,gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    print('===finish setting network and optimizer===')

    print('===start pure training===')
    maxMAP = 0
    total_step = len(train_loader)  # batch数量
    train_loader_iter = iter(cycle(train_loader))  # iter(dataloader)返回的是一个迭代器，然后可以使用next访问
    for epoch in range(opt.num_epochs):
        net.train()
        start_time = time.time()
        # scheduler.step()
        epoch_loss = 0.
        for i in range(total_step):  # 逐个batch地遍历整个训练集
            frames, _, labels = next(train_loader_iter)
            frames = frames.to(device)
            labels = labels.to(device)
            hash_features = net(frames)
            loss = triplet_loss(hash_features, labels)
            print(f'[epoch{epoch}-batch{i}] loss:{loss:0.4}')
            if loss == 0:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / total_step
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'[{epoch}/{opt.num_epochs}] loss:{avg_loss:0.5f} '
              f' time:{elapsed_time:0.2f} s')

        if epoch % opt.checkpoint_step == 0:  # (epoch + 1) % 2 == 0:
            scheduler.step(maxMAP)
            map_start_time = time.time()
            print('getting binary code and label')
            db_binary, db_label = inference(db_loader, net, opt.hash_length, device)
            test_binary, test_label = inference(test_loader, net, opt.hash_length, device)
            print('calculating mAP')
            MAP_ = compute_MAP(db_binary, db_label, test_binary, test_label)
            print("MAP_: %s" % MAP_)

            f = open(os.path.join(checkpoint_path, "MAP.log"), "a+")
            f.write('epoch:' + str(epoch) + "  loss:" + str(avg_loss) + '  mAP:' + str(MAP_) + '\n')
            f.close()

            if MAP_ > maxMAP:
                maxMAP = MAP_
                save_pth_path = os.path.join(checkpoint_path, f'net_mAP{MAP_:04f}.pth')  # epoch{epoch}_
                torch.save(net.state_dict(), save_pth_path)

            map_end_time = time.time()
            print('calcualteing mAP used ', map_end_time - map_start_time, 's')

def KD(opt, train_loader, test_loader, db_loader, checkpoint_path):
    print('===start setting network and optimizer===')
    device = torch.device("cuda:" + opt.cudaid if torch.cuda.is_available() else "cpu")  # device configuration
    triplet_loss = TripletLoss(opt.margin, device).to(device)

    teacher_net = C3D_Hash_Model(opt.hash_length)
    teacher_net.to(device)
    teacher_net = torch.nn.DataParallel(teacher_net, device_ids=opt.deviceid)
    teacher_net.load_state_dict(torch.load(opt.teacher_model_path))

    net = C3D_Hash_Model(opt.hash_length)
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=opt.deviceid)  # for multi gpu
    if opt.load_model:
        net.load_state_dict(torch.load(opt.load_model_path))
        print('loaded model from ' + opt.load_model_path)
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.step_lr,gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    print('===finish setting network and optimizer===')

    print('===start pure training===')
    maxMAP = 0
    total_step = len(train_loader)  # batch数量
    train_loader_iter = iter(cycle(train_loader))  # iter(dataloader)返回的是一个迭代器，然后可以使用next访问
    for epoch in range(opt.num_epochs+1):
        net.train()
        start_time = time.time()
        # scheduler.step()
        epoch_loss = 0.
        for i in range(total_step):  # 逐个batch地遍历整个训练集
            frames,teacher_frames, labels= next(train_loader_iter)
            # frames = frames[:,:, 0:32:2, :, :]  放这里的时候已经不行了。不知道为什么，难道是batch的问题吗
            frames = frames.to(device)
            teacher_frames=teacher_frames.to(device)
            labels = labels.to(device)

            # frame_step = int(opt.teacher_num_frames / opt.num_frames)
            # end_index = int(frame_step * opt.num_frames)

            with torch.no_grad():  # 加上no grad能省超多显存
                teacher_hash_features = teacher_net(teacher_frames)

            student_hash_features = net(frames)
            Ltriplet = opt.L3weight*triplet_loss(student_hash_features, labels)
            mseloss = torch.nn.MSELoss()
            Lrep = opt.Lrepweight * mseloss(teacher_hash_features, student_hash_features)
            Loss = Ltriplet + Lrep

            print(f'[epoch{epoch}-batch{i}] Loss:{Loss:0.4} Ltriplet:{Ltriplet:0.4} Lrep:{Lrep:0.4}')
            if Loss == 0:
                continue
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            epoch_loss += Loss.item()

        avg_loss = epoch_loss / total_step
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'[{epoch}/{opt.num_epochs}] loss:{avg_loss:0.5f} '
              f' time:{elapsed_time:0.2f} s')

        if epoch % opt.checkpoint_step == 0:  # (epoch + 1) % 2 == 0:
            scheduler.step(maxMAP)
            map_start_time = time.time()
            print('getting binary code and label')
            db_binary, db_label = inference(db_loader, net, opt.hash_length, device)
            test_binary, test_label = inference(test_loader, net, opt.hash_length, device)
            print('calculating mAP')
            MAP_ = compute_MAP(db_binary, db_label, test_binary, test_label)
            print("MAP_: %s" % MAP_)

            f = open(os.path.join(checkpoint_path, "MAP.log"), "a+")
            f.write('epoch:' + str(epoch) + "  loss:" + str(avg_loss) + '  mAP:' + str(MAP_) + '\n')
            f.close()

            if MAP_ > maxMAP:
                maxMAP = MAP_
                save_pth_path = os.path.join(checkpoint_path, f'net_mAP{MAP_:04f}.pth')  # epoch{epoch}_
                torch.save(net.state_dict(), save_pth_path)

            map_end_time = time.time()
            print('calcualteing mAP used ', map_end_time - map_start_time, 's')

def prepare_data(opt,the_num_frames):
    print('===setting data loader===')
    if opt.dataset_name == 'UCF':
        root_folder = "/home/disk1/azzh/data/UCF-101/"
        train_fpath_label = "/home/disk1/azzh/data/UCF-101/TrainTestlist/train1.txt"
        test_fpath_label = "/home/disk1/azzh/data/UCF-101/TrainTestlist/test1.txt"
    elif opt.dataset_name == 'HMDB':
        root_folder = "/home/disk3/a_zhongzhanhui/data/HMDB-51/HMDB51/"
        train_fpath_label = "/home/disk3/a_zhongzhanhui/data/HMDB-51/TrainTestlist/labels/train1.txt"
        test_fpath_label = "/home/disk3/a_zhongzhanhui/data/HMDB-51/TrainTestlist/labels/test1.txt"
    elif opt.dataset_name == 'JHMDB':
        root_folder = "/home/disk1/azzh/data/JointHMDB/Frames"
        train_fpath_label = "/home/disk1/azzh/data/JointHMDB/Label_Split/train_10_210.txt"
        test_fpath_label = "/home/disk1/azzh/data/JointHMDB/Label_Split/test_10_210.txt"
        db_fpath_label = '/home/disk1/azzh/data/JointHMDB/Label_Split/db_20_420.txt'
    else:
        print('dataset_name error')
        exit(0)
    train_loader = load_data(opt,root_folder, train_fpath_label, opt.batch_size, shuffle=True, num_workers=16, train=False,
                             num_frames=the_num_frames)
    test_loader = load_data(opt,root_folder, test_fpath_label, opt.batch_size, shuffle=False, num_workers=8, train=False,
                            num_frames=the_num_frames)
    if opt.dataset_name == 'UCF' or opt.dataset_name == 'HMDB':
        db_loader = train_loader
    elif opt.dataset_name == 'JHMDB':
        db_loader = load_data(opt,root_folder, db_fpath_label, opt.batch_size, shuffle=True, num_workers=16, train=False,
                              num_frames=the_num_frames)


    return train_loader,db_loader,test_loader

def get_parser():
    parser = argparse.ArgumentParser(description='train C3DHash')

    parser.add_argument('--dataset_name', default='UCF', help='HMDB or UCF or JHMDB')
    parser.add_argument('--mode', type=str, default='KD', help='pure or KD')
    parser.add_argument('--num_frames', type=int, default=16, help='number of frames taken form a video')
    parser.add_argument('--hash_length', type=int, default=48, help='length of hashing binary')
    parser.add_argument('--margin', type=float, default=14, help='取bit的四分之一多一点，margin影响很大')

    parser.add_argument('--L3weight', type=float, default=0.8, help='取bit的四分之一多一点，margin影响很大')
    parser.add_argument('--Lrepweight', type=float, default=1, help='取bit的四分之一多一点，margin影响很大')

    parser.add_argument('--postfix', type=str, default='0.8+1', help='postfix of checkpoint file')

    parser.add_argument('--cudaid', type=str, default='0', help='')
    parser.add_argument('--deviceid', type=list, default=[0,2], help='')

    parser.add_argument('--lr', type=float, default=0.0001, help='UCF建议用 0.0001， JHMDB用 0.001')
    parser.add_argument('--batch_size', type=int, default=120, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--step_lr', type=int, default=40, help='change lr per strp_lr epoch')
    parser.add_argument('--checkpoint_step', type=int, default=5, help='checkpointing after batches')

    parser.add_argument('--load_model', default=False, help='wether load model checkpoints or not')
    parser.add_argument('--load_model_path',
         default='/home/disk1/azzh/PycharmProject/video_retrieval_KD/checkpoints/UCF_48bits_14margin_16frames_KD_1Lrep/net_mAP0.768158.pth',help='location to load model')


    parser.add_argument('--teacher_num_frames', type=int, default=32, help='')
    parser.add_argument('--teacher_model_path',
                        default='/home/disk1/azzh/PycharmProject/video_retrieval_KD/checkpoints/UCF_48bits_14margin_32frames_pure_/net_mAP0.801399.pth',
                        help='')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()

    checkpoint_path = './checkpoints/' + opt.dataset_name + '_' + str(opt.hash_length) + 'bits_' + str(
        opt.margin) + 'margin_' + str(opt.num_frames) + 'frames'+ '_' + opt.mode + '_'+ opt.postfix
    os.makedirs(checkpoint_path, exist_ok=True)
    f = open(os.path.join(checkpoint_path, "MAP.log"), "a+")
    f.write(str(opt) + '\n')
    f.close()



    if opt.mode=='pure':
        train_loader, db_loader, test_loader = prepare_data(opt,opt.num_frames)
        Pure(opt,train_loader,test_loader,db_loader,checkpoint_path)
    elif opt.mode=='KD':
        train_loader, db_loader, test_loader = prepare_data(opt,opt.teacher_num_frames)
        KD(opt, train_loader, test_loader, db_loader,checkpoint_path)
    else:
        raise Exception("mode wrong")





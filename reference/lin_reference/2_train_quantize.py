import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import datetime
import numpy as np
import pywt

from torchvision import transforms
from collections import Counter
from model.model_bnfuse import *
from model.transform import *
from model.Quantize import prepare
from torch.utils.data import WeightedRandomSampler

# from model_lzw_gai import MainClassifierCNN_developed
from sklearn.model_selection import train_test_split

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def seed_torch(seed=509):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class ToDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        image = image.float()
        return image, label

def make_weighted_sampler(labels):
    counts = Counter(labels)
    class_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sample_weights = np.array([ class_weights[label] for label in labels ])
    sample_weights = torch.from_numpy(sample_weights).double()
    # print(len(sample_weights))
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler

def calibration(model, train_loader, device):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        output = model(data)

        if batch_idx % 50 == 0:
            print('Calibrating : [{}/{}]'.format(batch_idx * len(data), len(train_loader.dataset)))
    print('Calibrating Done\n')
    with open(log_file, 'a') as f:
        f.write("Calibrating done\n")
        
        
def train(epoch, net, trainloader, trainset, testloader, device):
    print("========================== Training ===========================")
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    with open(log_file, 'a') as f:
        f.write("====================== Hyperparameters ========================\n")
        f.write(f"Init Learning rate: {learning_rate}\n")
        f.write(f"Total of epochs, Batch size: {epoch}, {batch_size}\n")
        f.write("========================== Net ================================\n")
        f.write(str(net))
        f.write('\n')
    best_test_acc = 0
    for e in range(epoch):
        print("Epoch: %d, Learning rate: %f" % (e + 1, optimizer.param_groups[0]['lr']))
        # ============================ Train ============================
        net.train()
        loss_ = 0
        train_sum = 0
        train_n = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # _, predicted = torch.max(outputs, 1)
            predict = torch.max(outputs, 1)[1].data.squeeze()
            train_n += 1
            train_sum += (predict == labels).sum().item() / labels.size(0)
            loss_ += loss.item()
            # if i % batch_size == 0:
            #     print('Epoch: %d, [%6d / %6d], Train loss: %.4f, Train accuracy: %.4f ' % (
            #         e + 1, (i + 1) * batch_size, len(trainset), loss.item(), accuracy))
        train_acc = train_sum / train_n
        print('Epoch: %d, train loss is %f, train accuracy is %f' % (e + 1, loss_, train_acc))
        scheduler.step()
        # ============================ Test ============================
        net.eval()
        total_samples = 0
        correct = 0
        sar_accept = 0
        total_spoof = 0
        frr_reject = 0
        total_real = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                predictions = torch.argmax(outputs, dim=1)
                total_samples += labels.size(0)
                correct += (predictions == labels).sum().item()
                for true_label, pred_label in zip(labels, predictions):
                    true_label = true_label.item()
                    pred_label = pred_label.item()
                    # Fake样本（标签0）
                    if true_label == 0:
                        total_spoof += 1
                        if pred_label == 1:  # Fake被误判为Live → SAR事件
                            sar_accept += 1
                    # Live样本（标签1）
                    elif true_label == 1:
                        total_real += 1
                        if pred_label == 0:  # Live被误判为Fake → FRR事件
                            frr_reject += 1
            test_acc = 100 * correct / total_samples
            sar = sar_accept / total_spoof if total_spoof > 0 else 0.0
            frr = frr_reject / total_real if total_real > 0 else 0.0

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_path = f"{model_path}/best_model_epoch{e + 1}.pth"
                torch.save(net.state_dict(), best_model_path)
                with open(log_file, 'a') as f:
                    f.write(f"Epoch: {e + 1}, Best Test Accuracy: {best_test_acc:.4f}%, SAR: {sar * 100:.2f}%, FRR: {frr * 100:.2f}%\n")
            print('Epoch: %d, Test accuracy: %.4f%% ' % (e + 1, test_acc))
            print('The best test accuracy is %.4f%% ' % best_test_acc)
            print("=======================================================================")
            with open(log_file, 'a') as f:
                f.write('Epoch: %d, Learning rate: %.8f, Loss: %.4f, Train accuracy: %.4f, Test accuracy: %.4f, Best Test accuracy: %.4f\n' % (
                    e + 1, optimizer.param_groups[0]['lr'], loss_, train_acc, test_acc, best_test_acc))
    print("============================ Training Finished ============================")


def test(net, testloader, device):
    print("============================ Test ============================")
    net.eval()
    total_samples = 0
    correct = 0
    sar_accept = 0
    total_spoof = 0
    frr_reject = 0
    total_real = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            predictions = torch.argmax(outputs, dim=1)
            total_samples += labels.size(0)
            correct += (predictions == labels).sum().item()
            for true_label, pred_label in zip(labels, predictions):
                true_label = true_label.item()
                pred_label = pred_label.item()
                # Fake样本（标签0）
                if true_label == 0:
                    total_spoof += 1
                    if pred_label == 1:  # Fake被误判为Live → SAR事件
                        sar_accept += 1
                # Live样本（标签1）
                elif true_label == 1:
                    total_real += 1
                    if pred_label == 0:  # Live被误判为Fake → FRR事件
                        frr_reject += 1
        accuracy = 100 * correct / total_samples
        sar = sar_accept / total_spoof if total_spoof > 0 else 0.0
        frr = frr_reject / total_real if total_real > 0 else 0.0
        results = [
            f"Total samples: {total_samples}",
            f"Accuracy: {accuracy:.4f}%",
            f"SAR (Spoof Accept Rate): {sar * 100:.2f}% ({sar_accept}/{total_spoof})",
            f"FRR (False Reject Rate): {frr * 100:.2f}% ({frr_reject}/{total_real})",
            "==============================================================="
        ]
        for line in results:
            print(line)
        # with open(log_file, 'a') as f:
        #     for line in results:
        #         print(line)
        #         f.write(line + "\n")

    return accuracy, sar, frr


train_transform = transforms.Compose([
    numpy2tensor,
    transforms.ColorJitter(contrast=(0.5, 1.5)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation([0, 360]),
    # wavelet_vec,
])

test_transform = transforms.Compose([
    numpy2tensor,
    # transforms.ToTensor(),
    # wavelet_vec,
])


if __name__ == "__main__":
    seed_torch(seed=509)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-3
    epoch_n = 3000
    T_max = 100
    batch_size = 128
    base_path = './save_pth_qat'
    dataset_path = './dataset/patch224_824'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # load the all dataset
    train_image = np.load(dataset_path + '/train_data.npy')
    train_label = np.load(dataset_path + '/train_labels.npy')
    train_dataset = ToDataset(train_image, train_label, train_transform)
    print('train: ', Counter(train_dataset.labels))
 
    test_image = np.load(dataset_path + '/test_data.npy')
    test_label = np.load(dataset_path + '/test_labels.npy')
    test_dataset = ToDataset(test_image, test_label, test_transform)
    
    print('test: ', Counter(test_dataset.labels))
    print(f"Training set size: {len(train_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")

    # sampler = make_weighted_sampler(train_dataset.labels)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    start_time = datetime.datetime.now()
    current_date = start_time.strftime('%Y%m%d')  
    timestamp = start_time.strftime('%H%M')

    net = DS37K_wav2_bn(num_classes=2).to(device)

    model_name = type(net).__name__
    log_file = f"{base_path}/{current_date}/{model_name}_{timestamp}.txt"
    model_path = f"{base_path}/{current_date}/{model_name}_{timestamp}"
    os.makedirs(model_path, exist_ok=True)
    
    print(net)
    # =================================== Later =============================================
    net = prepare(
        model=net,
        inplace=True,
        a_bits=8,
        w_bits=8,
        qaft=False,).to(device)
    print(net)
    # =================================== Original model Load===========================================
    model_pth = './save_pth_qat/pretrained_bnfuse/DC37K_wav2_0726.pth'
    if model_pth is not None:
        net.load_state_dict(torch.load(model_pth))
        # net.load_state_dict(torch.load(model_pth),strict=False)
    test(net, test_loader, device)
    # =================================== First ======================================
    # net = prepare(
    #     model=net,
    #     inplace=True,
    #     a_bits=8,
    #     w_bits=8,
    #     qaft=False,).to(device)
    # print(net)
    # =================================================================================================
    calibration(net, train_loader, device)
    test(net, test_loader, device)  # PTQ
    train(epoch_n, net, train_loader, train_dataset, test_loader, device) # QAT
    test(net, test_loader, device)

    end_time = datetime.datetime.now()
    ti = (end_time - start_time).seconds
    hou = ti / 3600
    ti = ti % 3600
    sec = ti / 60
    ti = ti % 60
    print('Training Time: %dh-%dm-%ds' % (hou, sec, ti))

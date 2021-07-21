import os
import subprocess
import numpy as np
import random as rd
from PIL import Image
from captcha.image import ImageCaptcha
import torch as t
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
import visdom
import tqdm
from torchnet import meter


class Parameters:
    # 路径相关
    train_path = "captcha/train"
    test_path = "captcha/test"
    model_path = "captcha/model"

    # 不可修改的参数
    tensorLength = 248
    charLength = 62
    charNumber = 4
    ImageWidth = 32
    ImageHeight = 32

    # 可修改的参数
    learningRate = 1e-3
    totalEpoch = 200
    batchSize = 100
    printCircle = 10
    testCircle = 100
    testNum = 6
    saveCircle = 200


class Visualizer:
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_many_stack(self, d):
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        # print(x)
        self.vis.line(Y=y, X=np.ones(y.shape) * x,
                      win=str(name_total),  # unicode
                      opts=dict(legend=name,
                                title=name_total),
                      update=None if x == 0 else 'append'
                      )
        self.index[name_total] = x + 1


class Captcha(data.Dataset):
    def __init__(self, root):
        self.imgsPath = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = transforms.Compose([
            transforms.Resize((Parameters.ImageHeight, Parameters.ImageWidth)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        imgPath = self.imgsPath[index]
        # print(imgPath)
        label = imgPath.split("/")[-1].split(".")[0]
        # print(label)
        labelTensor = t.Tensor(self.StrtoLabel(label))
        img = Image.open(imgPath)
        # print(img.size)
        img = self.transform(img)
        # print(img.shape)
        return img, labelTensor

    def __len__(self):
        return len(self.imgsPath)

    def StrtoLabel(self, Str):
        # print(Str)
        label = []
        for i in range(0, Parameters.charNumber):
            if '0' <= Str[i] <= '9':
                label.append(ord(Str[i]) - ord('0'))
            elif 'a' <= Str[i] <= 'z':
                label.append(ord(Str[i]) - ord('a') + 10)
            else:
                label.append(ord(Str[i]) - ord('A') + 36)
        return label

    @staticmethod
    def LabeltoStr(Label):
        Str = ""
        for i in Label:
            if i <= 9:
                Str += chr(ord('0') + i)
            elif i <= 35:
                Str += chr(ord('a') + i - 10)
            else:
                Str += chr(ord('A') + i - 36)
        return Str


class ResNet(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, inchannel, outchannel, stride=1):
            super().__init__()
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=True)
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(outchannel, track_running_stats=True)
                )

        def forward(self, x):
            out = self.left(x)
            out += self.shortcut(x)
            out = F.relu(out)
            return out
            
    def __init__(self, num_classes=62):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(self.ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(self.ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(self.ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(self.ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 512)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def save(self, circle):
        name = Parameters.model_path + "/resNet" + str(circle) + ".pth"
        t.save(self.state_dict(), name)
        name2 = Parameters.model_path + "/resNet_new.pth"
        t.save(self.state_dict(), name2)

    def loadIfExist(self):
        fileList = os.listdir(Parameters.model_path)
        # print(fileList)
        if "resNet_new.pth" in fileList:
            name = Parameters.model_path + "/resNet_new.pth"
            self.load_state_dict(t.load(name))
            print("the latest model has been load")


class Train:
    def __init__(self):
        self.model = ResNet()
        self.model.loadIfExist()
        trainDataset = Captcha(Parameters.train_path + "/")
        testDataset = Captcha(Parameters.test_path + "/")
        self.trainDataLoader = DataLoader(trainDataset, batch_size=Parameters.batchSize, shuffle=True,
                                          num_workers=4)
        self.testDataLoader = DataLoader(testDataset, batch_size=Parameters.batchSize, shuffle=True, num_workers=4)

    def train(self):
        avgLoss = 0.0
        if t.cuda.is_available():
            self.model = self.model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=Parameters.learningRate)
        vis = Visualizer(env="ResCaptcha")
        loss_meter = meter.AverageValueMeter()
        for epoch in range(Parameters.totalEpoch):
            for circle, inputed in tqdm.tqdm(enumerate(self.trainDataLoader, 0)):
                x, label = inputed
                if t.cuda.is_available():
                    x = x.cuda()
                    label = label.cuda()
                label = label.long()
                label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
                # print(label1,label2,label3,label4)
                optimizer.zero_grad()
                y1, y2, y3, y4 = self.model(x)
                # print(y1.shape, y2.shape, y3.shape, y4.shape)
                loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(y2, label2), criterion(y3, label3), criterion(y4, label4)
                loss = loss1 + loss2 + loss3 + loss4
                loss_meter.add(loss.item())
                # print(loss)
                avgLoss += loss.item()
                loss.backward()
                optimizer.step()
                if circle % Parameters.printCircle == 1:
                    print("after %d circle,the train loss is %.5f" %
                          (circle, avgLoss / Parameters.printCircle))
                    self.writeFile(
                        "after %d circle,the train loss is %.5f" % (circle, avgLoss / Parameters.printCircle))
                    vis.plot_many_stack({"train_loss": avgLoss})
                    avgLoss = 0
                if circle % Parameters.testCircle == 1:
                    accuracy = self.test()
                    vis.plot_many_stack({"test_acc": accuracy})
                if circle % Parameters.saveCircle == 1:
                    self.model.save(str(epoch) + "_" + str(Parameters.saveCircle))

    def test(self):
        totalNum = Parameters.testNum * Parameters.batchSize
        rightNum = 0
        for circle, inputed in enumerate(self.testDataLoader, 0):
            if circle >= Parameters.testNum:
                break
            x, label = inputed
            label = label.long()
            if t.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            y1, y2, y3, y4 = self.model(x)
            y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(Parameters.batchSize, 1), y2.topk(1, dim=1)[1].view(Parameters.batchSize, 1), y3.topk(1, dim=1)[1].view(Parameters.batchSize, 1), y4.topk(1, dim=1)[1].view(Parameters.batchSize, 1)
            y = t.cat((y1, y2, y3, y4), dim=1)
            diff = (y != label)
            diff = diff.sum(1)
            diff = (diff != 0)
            res = diff.sum(0).item()
            rightNum += (Parameters.batchSize - res)
        print("the accuracy of test set is %s" % (str(float(rightNum) / float(totalNum))))
        self.writeFile("the accuracy of test set is %s" % (str(float(rightNum) / float(totalNum))))
        return float(rightNum) / float(totalNum)

    @staticmethod
    def writeFile(text):
        file = open("result.txt", "a+")
        file.write(text)
        file.write("\n\n")
        file.flush()
        file.close()


class GenerateCaptcha:
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u',
                  'v', 'w', 'x', 'y', 'z']
    upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U',
                  'V', 'W', 'X', 'Y', 'Z']

    @staticmethod
    def get_width():
        return int(100 + 40 * rd.random())

    @staticmethod
    def get_height():
        return int(45 + 20 * rd.random())

    def get_string(self):
        string = ""
        for i in range(4):
            select = rd.randint(1, 3)
            if select == 1:
                index = rd.randint(0, 9)
                string += self.nums[index]
            elif select == 2:
                index = rd.randint(0, 25)
                string += self.lower_char[index]
            else:
                index = rd.randint(0, 25)
                string += self.upper_char[index]
        return string

    def get_captcha(self, num, path):
        font_sizes = [x for x in range(40, 45)]
        for i in range(num):
            imc = ImageCaptcha(self.get_width(), self.get_height(), font_sizes=font_sizes)
            name = self.get_string()
            image = imc.generate_image(name)
            image.save(path + name + ".jpg")


class UseModel:
    class LoadFile(data.Dataset):
        def __init__(self, path):
            self.path = path
            self.transform = transforms.Compose([
                transforms.Resize((Parameters.ImageHeight, Parameters.ImageWidth)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        def __getitem__(self, item):
            img = Image.open(self.path)
            img = self.transform(img)
            return img

        def __len__(self):
            return 1

    def __init__(self):
        self.model = ResNet()
        self.model.eval()
        self.model.loadIfExist()
        if t.cuda.is_available():
            self.model = self.model.cuda()

    def run_folder(self, path):
        userTestDataset = Captcha(path)
        dataLoader = DataLoader(userTestDataset, batch_size=1, shuffle=True, num_workers=1)
        for circle, inputed in enumerate(dataLoader, 0):
            if circle >= 200:
                break
            x, label = inputed
            if t.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            # print(label,realLabel)
            y1, y2, y3, y4 = self.model(x)
            y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
            y = t.cat((y1, y2, y3, y4), dim=1)
            # print(x,label,y)
            return Captcha.LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])

    def run_file(self, path):
        userTestDataset = self.LoadFile(path)
        dataLoader = DataLoader(userTestDataset, batch_size=1)
        img = next(iter(dataLoader))
        y1, y2, y3, y4 = self.model(img)
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
        y = t.cat((y1, y2, y3, y4), dim=1)
        return Captcha.LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])


class ModelBuild:
    def retrain(self):
        for x in os.listdir(Parameters.train_path):
            os.remove(Parameters.train_path + "/" + x)
        for x in os.listdir(Parameters.test_path):
            os.remove(Parameters.test_path + "/" + x)
        for x in os.listdir(Parameters.model_path):
            os.remove(Parameters.model_path + "/" + x)
        self.train()

    def train(self):
        if len(os.listdir(Parameters.train_path)) == 0:
            GenerateCaptcha().get_captcha(5000, Parameters.train_path + "/")
        if len(os.listdir(Parameters.test_path)) == 0:
            GenerateCaptcha().get_captcha(500, Parameters.test_path + "/")
        self.optimization()

    @staticmethod
    def optimization():
        subprocess.Popen("python -m visdom.server", shell=True)
        Train().train()


if __name__ == '__main__':
    ModelBuild().train()

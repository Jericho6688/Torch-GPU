
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Normalize, RandomHorizontalFlip
from torch.utils.data import DataLoader
import warnings
from torchvision import transforms
from tqdm import tqdm

# 修改警告过滤
warnings.filterwarnings("ignore", category=UserWarning)

# 数据路径
train_data_dir = 'DATASET/DataSet_1(landscape)/seg_train/seg_train'
test_data_dir = 'DATASET/DataSet_1(landscape)/seg_test/seg_test'

# 定义图像大小、批次大小等参数
image_size = (150, 150)
epochs = 10
batch_size = 32

# 数据预处理
# 创建训练集和测试集的transforms
train_transforms = transforms.Compose([
    Resize(image_size),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    RandomHorizontalFlip()
])

test_transforms = transforms.Compose([
    Resize(image_size),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建训练集和测试集
train_dataset = ImageFolder(train_data_dir, transform=train_transforms)
test_dataset = ImageFolder(test_data_dir, transform=test_transforms)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 构建卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 创建模型实例并将其移动到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=6).to(device)
print(f"Using device: {device}")  # 打印设备信息

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 在GPU上训练模型
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # 训练
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}", unit="batch")
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_loader_tqdm.set_postfix(loss=running_loss/(total/batch_size), accuracy=100*correct/total)

    # 评估
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}/{epochs}", unit="batch")
        for images, labels in test_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loader_tqdm.set_postfix(loss=test_loss/(total/batch_size), accuracy=100*correct/total)

    print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {100 * correct / total:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), f"model_epoch{epoch + 1}.pth")

print("Training complete.")
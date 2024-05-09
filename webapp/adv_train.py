import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PyTorchYOLOv3.models import Darknet
from webapp.utils.CONSTANTS import *
from CustomDataset import AdversarialDataset, ImageDataset

# 加载模型
model = Darknet(CFG_FILE)
model.load_darknet_weights(WEIGHT_FILE)
model.train()

# 加载原始数据
train_data = ImageDataset(STOP_SIGN_ALL_FOLDER)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 加载对抗样本数据
adv_data = AdversarialDataset(ADV_PT_FOLDER)
adv_loader = DataLoader(adv_data, batch_size=64, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
# 对抗训练
for epoch in range(epochs):
    for (images, labels), (adv_images, adv_labels) in zip(train_loader, adv_loader):
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 使用对抗样本进行前向传播
        adv_outputs = model(adv_images)
        adv_loss = criterion(adv_outputs, adv_labels)
        # 反向传播和优化
        optimizer.zero_grad()
        adv_loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Adv Loss: {adv_loss.item()}")
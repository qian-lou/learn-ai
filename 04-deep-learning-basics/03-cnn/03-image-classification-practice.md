# 图像分类实战 / Image Classification Practice

## 1. 背景（Background）
> 图像分类是深度学习的 "Hello World"，通过实战掌握完整的模型训练流程。

## 2-3. 知识点与内容
```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```

## 4-6. 推理/例题/习题
**练习：** 在 CIFAR-10 上训练 CNN，达到 85%+ 准确率。

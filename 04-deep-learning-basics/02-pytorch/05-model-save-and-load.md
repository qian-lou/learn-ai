# 模型保存与加载 / Model Save and Load

## 1. 背景（Background）
> 训练好的模型需要保存和加载。PyTorch 推荐保存 state_dict 而非整个模型。

## 2-3. 知识点与内容
```python
# 保存参数（推荐）/ Save parameters
torch.save(model.state_dict(), 'model.pth')

# 加载 / Load
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 保存完整 checkpoint（恢复训练用）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

## 4-6. 推理/例题/习题
**练习：** 实现训练中断后从 checkpoint 恢复训练的功能。

# CI/CD 与自动化部署 / CI/CD and Automated Deployment

## 1. 背景（Background）
> ML 项目的 CI/CD 比传统软件更复杂——需要管理模型版本、数据版本和代码版本的一致性。Java 工程师对 CI/CD 应该很熟悉。

## 2-3. 知识点与内容
```yaml
# GitHub Actions for ML Pipeline
name: ML Pipeline
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
      - name: Train model
        run: python train.py
      - name: Evaluate
        run: python evaluate.py
```

```
MLOps 完整流程：
代码管理 → 数据版本管理(DVC) → 训练 → 评估 → 模型注册 → 部署 → 监控 → 反馈循环

DVC (Data Version Control)：管理大文件和数据集版本
MLflow/W&B: 跟踪实验和模型版本
Kubernetes: 弹性伸缩推理服务
```

## 4-6. 推理/例题/习题
**练习：** 搭建一个完整的 MLOps Pipeline：代码更新 → 自动训练 → 评估 → 部署。

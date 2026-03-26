# CI/CD 与自动化部署 / CI/CD and Automated Deployment

## 1. 背景（Background）

> **为什么要学这个？**
>
> ML 项目的 CI/CD 比传统软件更复杂——需要管理**模型版本、数据版本和代码版本**的三重一致性。Java 工程师对 CI/CD 很熟悉，ML CI/CD 在此基础上增加了训练和评估步骤。

## 2. 知识点（Key Concepts）

| 工具 | 功能 |
|------|------|
| GitHub Actions | CI/CD 流水线 |
| DVC | 数据版本管理 |
| MLflow | 模型注册表 |
| Docker + K8s | 容器化部署 |

## 3. 内容（Content）

### 3.1 GitHub Actions ML Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline
on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'data/**'
      - 'configs/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  train:
    needs: test
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Train model
        run: python train.py --config configs/production.yaml
      - name: Evaluate
        run: python evaluate.py --model output/model
      - name: Upload model
        uses: actions/upload-artifact@v4
        with:
          name: model
          path: output/model/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          docker build -t llm-api:${{ github.sha }} .
          docker push registry/llm-api:${{ github.sha }}
          kubectl set image deployment/llm-api llm-api=registry/llm-api:${{ github.sha }}
```

### 3.2 DVC 数据版本管理

```bash
# DVC: Git for Data（像 Git 一样管理大文件）
pip install dvc

dvc init
dvc add data/training_data.jsonl
# → 生成 data/training_data.jsonl.dvc（小文件，提交到 Git）
# → 实际数据上传到 S3/GCS 等远端存储

git add data/training_data.jsonl.dvc .gitignore
git commit -m "Add training data v1"

# 版本切换
git checkout v1.0
dvc checkout  # 自动下载对应版本的数据
```

### 3.3 MLOps 完整流程

```
MLOps 流水线:

代码更新 → Git Push
    ↓
CI: 单元测试 + 代码检查
    ↓
数据验证: DVC 拉取 + 数据质量检查
    ↓
训练: GPU Runner 上训练模型
    ↓
评估: 自动评估 + 与基线对比
    ↓ (评估通过)
模型注册: MLflow Model Registry
    ↓
部署: Docker 构建 → K8s 滚动更新
    ↓
监控: Prometheus + Grafana
    ↓ (质量下降告警)
反馈: 收集用户反馈 → 更新训练数据
    ↓
循环 →
```

## 4. 详细推理（Deep Dive）

```
ML CI/CD vs 传统 CI/CD:

传统: 代码 → 测试 → 构建 → 部署
ML:   代码+数据+模型 → 测试 → 训练 → 评估 → 注册 → 部署 → 监控

关键区别:
  1. 需要 GPU Runner（训练步骤）
  2. 需要数据版本管理（DVC）
  3. 需要模型评估门控（评估不过不部署）
  4. 部署后需要持续监控（模型漂移）
```

## 5-6. 例题/习题

**练习 1：** 用 GitHub Actions 搭建自动化测试 + 训练流水线。

**练习 2：** 用 DVC 管理训练数据版本。

**练习 3：** 实现完整的 MLOps 闭环：训练 → 评估 → 部署 → 监控。

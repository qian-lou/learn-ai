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

## 5. 例题（Worked Examples）

### 例题 1：编写完整的 GitHub Actions 工作流进行代码 Lint、单元测试和镜像自动打包 / Automated CI/CD Workflow

在持续集成中，每当有代码提交到 `main` 分支时，我们需要自动拉起测试流程，并使用多阶段镜像打包程序将其推送至 Docker Hub。

```yaml
name: LLM Service CI/CD

on:
  push:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - name: 拉取代码 / Checkout repository
      uses: actions/checkout@v4

    - name: 配置 Python 环境 / Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'

    - name: 安装测试依赖并运行 Lint / Install tools and Lint
      run: |
        pip install ruff pytest
        ruff check .  # 执行代码格式静态分析 / Ruff Lint check.

    - name: 执行单元测试 / Run tests
      run: |
        pytest tests/  # 运行单元测试 / Run unit tests.

  docker-publish:
    needs: build-and-test
    runs-on: ubuntu-latest
    steps:
    - name: 登录 Docker Hub / Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_HUB_USERNAME }}
        password: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}

    - name: 构建并推送 Docker 镜像 / Build and push image
      uses: docker/build-push-action@v6
      with:
        context: .
        push: true
        tags: user/llm-service:latest
```

## 6. 习题（Exercises）

### 基础题
**练习 1**：在 CI/CD 流水线中，将密码、密钥（例如 Docker Hub Token, OpenAI API Key）硬编码在 workflow yaml 文件中有什么危害？应该如何正确配置？
*参考答案*：
硬编码会导致极大的安全隐患，任何能查看代码库的人都能窃取密钥。正确的做法是在 GitHub repository 偏好设置中的 "Secrets and variables -> Actions" 下添加加密的 Repository Secrets，并在 yaml 文件中通过 `${{ secrets.MY_SECRET_NAME }}` 安全引用。

### 进阶题
**练习 2**：在微调大模型的 CI/CD 流程中，由于微调训练往往需要长达数小时且需要昂贵的 GPU 设备，如何将这部分“重型训练任务”与日常提交代码的“轻量测试任务”在 GitHub Actions 流水线中合理分离？
*参考答案*：
可以通过以下两种方式分离：
1. **触发条件分离**：日常提交只触发 Lint 和轻量 CPU 单元测试。只有当向 GitHub 仓库打上特定 Tag（如 `v*.*-train`）或手动选择 `workflow_dispatch` 时才触发微调流水线。
2. **执行节点托管**：通过配置 Self-hosted Runner，将重型微调任务分发至内网自建的 GPU 服务器节点上，而不是运行在 GitHub 提供的免费 CPU 虚拟机上。

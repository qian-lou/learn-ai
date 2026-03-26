# Scikit-learn Pipeline

## 1. 背景（Background）
> Pipeline 将预处理和模型训练串联为一条流水线，保证训练和推理数据处理一致。类似 Java 的责任链模式。

## 2-3. 知识点与内容
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),        # 标准化
    ('classifier', LogisticRegression()), # 分类器
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)  # 自动按顺序执行
```

## 4. 详细推理
- Pipeline 防止数据泄露（test 数据不参与 scaler 的 fit）
- 可嵌套 `ColumnTransformer` 处理混合类型特征

## 5-6. 例题/习题
**练习：** 构建一个包含特征选择、标准化、PCA 降维、分类的完整 Pipeline。

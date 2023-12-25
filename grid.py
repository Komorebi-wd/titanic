import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 载入数据集
data = pd.read_csv('train.csv')  # 确保文件名和路径正确

# 处理目标变量
data['Transported'] = data['Transported'].astype(int)

# 分离特征和目标变量
X = data.drop('Transported', axis=1)
y = data['Transported']

# 定义数值型和分类型特征列
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

# 预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 处理数值型特征的缺失值
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 处理分类型特征的缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 创建 SVM 模型
svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义 SVM 参数网格
param_grid = {
    'classifier__C': [0.01,0.1,1,10,30,50,70,100],  # 正则化参数
    'classifier__gamma': ['scale','auto',0.001,0.01,0.1,1],  # 核函数参数
    'classifier__kernel': ['rbf','linear','poly','sigmoid','precomputed']  # 核函数类型
}

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', verbose=2)

# 运行网格搜索
grid_search.fit(X_train, y_train)

# 最佳参数和准确率
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# 使用最佳参数的模型在测试集上的性能
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy: %.2f%%' % (accuracy * 100.0))

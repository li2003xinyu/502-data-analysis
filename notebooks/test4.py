import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


file_path = '../data/1/Z24-125/800s/Z24-125-BLS-01-5K-20260105135547.txt'
df = pd.read_csv(file_path, sep='\t', encoding='gbk')



df = df[(df['时间s'] >= 0) & (df['时间s'] <=810)]
df = df.iloc[::100, :]
#'燃压' '混合比'
features = ['氧喷前压', '燃喷前压', '氧流量', '燃流量', '氧阀门电流', '燃阀门电流']


target = '推力'


split_time = 567
train_df = df[df['时间s'] < split_time]
test_df = df[df['时间s'] >= split_time]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]


#  RBF（径向基）核函数
svr_model = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=10, gamma='scale')  # C :惩罚系数，gamma :核函数宽度
)

svr_model.fit(X_train, y_train)


y_pred_svr = svr_model.predict(X_test)
y_pred_train = svr_model.predict(X_train)
print("支持向量回归 (SVR) 结果:")
print(f"R² 训练: {r2_score(y_train, y_pred_train):.4f}")
print(f"R² 测试: {r2_score(y_test, y_pred_svr):.4f}")
print(f"MSE训练 : {mean_squared_error(y_train, y_pred_train ):.4f}")
print(f"MSE测试 : {mean_squared_error(y_test, y_pred_svr):.4f}")

#####################################################################
import plotly.graph_objects as go


plot_x = df['时间s']
plot_y_true = df[target]
plot_y_pred = y_pred_svr


fig = go.Figure()


fig.add_trace(go.Scatter(
    x=plot_x,
    y=plot_y_true,
    mode='lines',
    name='真实推力',
    line=dict(color='royalblue', width=2)
))


fig.add_trace(go.Scatter(
    x=test_df['时间s'],
    y=plot_y_pred,
    mode='lines',
    name='重构推力',
    line=dict(color='firebrick', width=2, dash='dash')
))


fig.update_layout(
    title='真实推力 vs 重构推力',
    xaxis_title='时间 (s)',
    yaxis_title='推力',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()
########################################################################################
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置微软雅黑
plt.rcParams['axes.unicode_minus'] = False


result = permutation_importance(
    svr_model, X_test, y_test,
    n_repeats=10,
    random_state=42
)


sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(8, 5))
plt.barh(np.array(features)[sorted_idx],
         result.importances_mean[sorted_idx],
         color='teal')
plt.xlabel("打乱该特征后 R² 的下降幅度")
plt.title("SVR 特征重要性")
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(10, 4))
PartialDependenceDisplay.from_estimator(
    svr_model, X_test,
    features=['氧阀门电流', '燃阀门电流'],
    ax=ax
)
plt.suptitle("SVR 偏依赖图")
plt.show()
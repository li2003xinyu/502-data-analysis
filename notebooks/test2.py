import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
file_path = '../data/1/Z24-125/800s/Z24-125-BLS-01-5K-20260105135547.txt'#800s
df = pd.read_csv(file_path, sep='\t', encoding='gbk')

df = df[(df['时间s'] >=0) & (df['时间s'] <= 810)]
#df = df.iloc[::10, :]


#features = ['燃压','氧喷前压', '燃喷前压', '氧流量', '燃流量', '氧阀门电流', '燃阀门电流','混合比']
#target = '推力'
features = ['氧喷前压','推力', '燃压', '氧流量', '燃流量', '氧阀门电流', '燃阀门电流','混合比']
target = '燃喷前压'
#X = df[features]
#y = df[target]
#一个时间点是一个样本
# 划分训练集和测试集
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #样本打乱
split_time = 567
train_df = df[df['时间s'] < split_time]
test_df = df[df['时间s'] >= split_time]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]



xgb_model = XGBRegressor(n_estimators=180, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print("\nXGBoost结果:")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

#importances_weight = xgb_model.feature_importances_
importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
df_imp = pd.DataFrame({
    '特征': list(importance_gain.keys()),
    '重要性': list(importance_gain.values())
})

df_imp = df_imp.sort_values(by='重要性', ascending=True)


fig = px.bar(
    df_imp,
    x='重要性',
    y='特征',
    orientation='h',
    title='XGBoost 特征重要性 (基于gain)',
    color='重要性',
    color_continuous_scale='Teal'
)

fig.update_layout(template='plotly_white', height=500)
fig.show()
############################################################################
import plotly.graph_objects as go


plot_x = df['时间s']
plot_y_true = df[target]
plot_y_pred = y_pred


fig = go.Figure()


fig.add_trace(go.Scatter(
    x=plot_x,
    y=plot_y_true,
    mode='lines',
    name='真实氧阀门电流',
    line=dict(color='royalblue', width=2)
))


fig.add_trace(go.Scatter(
    x=test_df['时间s'],
    y=plot_y_pred,
    mode='lines',
    name='重构燃喷前压',
    line=dict(color='firebrick', width=2, dash='dash')
))


fig.update_layout(
    title='真实燃喷前压 vs 重构燃喷前压',
    xaxis_title='时间 (s)',
    yaxis_title='燃喷前压',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()
####################################
"""
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置微软雅黑
plt.rcParams['axes.unicode_minus'] = False


result = permutation_importance(
    xgb_model, X_test, y_test,
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
    xgb_model, X_test,
    features=['氧阀门电流', '燃阀门电流'],
    ax=ax
)
plt.suptitle("SVR 偏依赖图")
plt.show()
"""
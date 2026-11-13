import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


file_path = '../data/1/Z24-125/800s/Z24-125-BLS-01-5K-20260105135547.txt'#800s
df = pd.read_csv(file_path, sep='\t', encoding='gbk')

df = df[(df['时间s'] >=0) & (df['时间s'] <= 810)]
#df = df.iloc[::10, :]

#'燃压''混合比'
features = ['氧喷前压', '燃喷前压', '氧流量', '燃流量', '氧阀门电流', '燃阀门电流']
target = '推力'
#features = ['推力', '燃喷前压', '氧流量', '燃流量', '氧阀门电流', '燃阀门电流']
#target = '氧喷前压'
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
##################################################################################
# 多变量线性回归(一阶)
lr_model = make_pipeline(StandardScaler(), LinearRegression())#StandardScaler()标准化（均值为0，方差为1）
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("多变量线性回归:")
print(f"R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")

coefs = lr_model.named_steps['linearregression'].coef_ #公式系数
print("\n各变量对推力的线性影响权重：")
for name, coef in zip(features, coefs):
    print(f"  {name}: {coef:.4f}")
print("-" * 40)
#################################################################################################
# 多项式回归(2阶)
poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    Ridge(alpha=10.0) # 岭回归
)
poly_model.fit(X_train, y_train)
y_pred_poly_2 = poly_model.predict(X_test)

print("多项式回归 (2阶) 结果:")
print(f"R²: {r2_score(y_test, y_pred_poly_2):.4f}")
print(f"MSE : {mean_squared_error(y_test, y_pred_poly_2):.4f}")

coefs = poly_model.named_steps['ridge'].coef_

poly_features = poly_model.named_steps['polynomialfeatures']
expanded_features = poly_features.get_feature_names_out(features)


print("\n多项式回归 (2阶)各项权重：")
for name, coef in zip(expanded_features, coefs):
    print(f"  {name}: {coef:.4f}")
print("-" * 40)
"""
################################################################################################
# 多项式回归(3阶)
poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=3, include_bias=False),
    Ridge(alpha=10.0) # 岭回归
)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

print("多项式回归 (3阶) 结果:")
print(f"R² : {r2_score(y_test, y_pred_poly):.4f}")
print(f"MSE : {mean_squared_error(y_test, y_pred_poly):.4f}")

coefs = poly_model.named_steps['ridge'].coef_

poly_features = poly_model.named_steps['polynomialfeatures']
expanded_features = poly_features.get_feature_names_out(features)


print("\n多项式回归 (3阶)各项权重：")
for name, coef in zip(expanded_features, coefs):
    print(f"  {name}: {coef:.4f}")
print("-" * 40)
################################################################################################
# 多项式回归(4阶)
poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=4, include_bias=False),
    Ridge(alpha=1.0) # 岭回归
)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

print("多项式回归 (4阶) 结果:")
print(f"R² (拟合度): {r2_score(y_test, y_pred_poly):.4f}")
print(f"MSE (均方误差): {mean_squared_error(y_test, y_pred_poly):.4f}")

coefs = poly_model.named_steps['ridge'].coef_

poly_features = poly_model.named_steps['polynomialfeatures']
expanded_features = poly_features.get_feature_names_out(features)


print("\n多项式回归 (4阶)各项权重：")
for name, coef in zip(expanded_features, coefs):
    print(f"  {name}: {coef:.4f}")
print("-" * 40)
"""
#############################################################################
import plotly.graph_objects as go


plot_x = df['时间s']
plot_y_true = df[target]
plot_y_pred = y_pred_poly_2 #2阶


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
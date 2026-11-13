import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import r2_score, mean_squared_error



file_path = '../data/1/Z24-125/800s/Z24-125-BLS-01-5K-20260105135547.txt'#800s
df = pd.read_csv(file_path, sep='\t', encoding='gbk')


df = df[(df['时间s'] >= 0) & (df['时间s'] <=810)]
df = df.iloc[::100, :]
#燃压，混合比
features = ['氧喷前压', '燃喷前压', '氧流量', '燃流量', '氧阀门电流', '燃阀门电流']
target = '推力'

X = df[features]
y = df[target]

split_time = 567
train_df = df[df['时间s'] < split_time]
test_df = df[df['时间s'] >= split_time]
X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

##############################################################
sr_model = SymbolicRegressor(
    population_size=2000,        # 种群大小
    generations=20,              # 进化代数
    p_crossover=0.7,             # 交叉概率
    p_subtree_mutation=0.1,      # 子树变异概率
    p_hoist_mutation=0.05,       # 提升变异概率
    p_point_mutation=0.1,        # 点变异概率
    max_samples=0.9,             # 每次用 90% 的数据训练
    verbose=1,
    parsimony_coefficient=0.01,  # 惩罚系数：逼迫模型寻找更简短、不冗长的公式
    random_state=42
)


sr_model.fit(X_train, y_train)


print("\n推导出的数学公式：")
print(sr_model._program)

y_pred_sr = sr_model.predict(X_test)
print("\n符号回归结果")
print(f"R²: {r2_score(y_test, y_pred_sr):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_sr):.4f}")
####################################################################
import plotly.graph_objects as go


plot_x = df['时间s']
plot_y_true = df[target]
plot_y_pred = y_pred_sr


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
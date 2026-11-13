import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
#file_path = '../data/1/Z24-125/800s/Z24-125-BLS-01-5K-20260105135547.txt'#800s
file_path = '../data/1/Z24-125/1280ms-128ms/Z24-125-BP-01-5K-20260105131755.txt'#1280ms-128ms
file_path_0 ='../data/1/Z24-136/1280ms-128ms/Z24-136-BP-01-5K-20260105125941.txt'
file_path_1 ='../data/1/Z24-150/1280ms-128ms/Z24-150-BP-01-5K-20260104145659.txt'
file_path_2 ='../data/1/Z24-157/1280ms-128ms/Z24-157-BP-01-5K-20260106123700.txt'
file_path_3 ='../data/1/Z24-184/1280ms-128ms/Z24-184-BP-01-5K-20260128112300.txt'
file_path_4 ='../data/1/Z24-188/1280ms-128ms/Z24-188-BP-01-5K-20260323151416.txt'

file_path_5 ='../data/1/Z24-189/1280ms-128ms/Z24-189-BP-01-5K-20260114095655.txt'
file_path_6 ='../data/1/Z24-190/1280ms-128ms/Z24-190-BP-01-5K-20260122123336.txt'
file_path_7 ='../data/1/Z24-191/1280ms-128ms/Z24-191-BP-01-5K-20260112154745.txt'
file_path_8 ='../data/1/Z24-192/1280ms-128ms/Z24-192-BP-01-5K-20260323153304.txt'
file_path_9 ='../data/1/Z24-193/1280ms-128ms/Z24-193-BP-01-5K-20260324135509.txt'

file_path_10 ='../data/1/Z24-194/1280ms-128ms/Z24-194-BP-01-5K-20260113120006.txt'
file_path_11 ='../data/1/Z24-195/1280ms-128ms/Z24-195-BP-01-5K-20260123114842.txt'
file_path_12 ='../data/1/Z24-196/1280ms-128ms/Z24-196-BP-01-5K-20260116113905.txt'
file_path_13 ='../data/1/Z24-197/1280ms-128ms/Z24-197-BP-01-5K-20260326133308.txt'
file_path_14 ='../data/1/Z24-198/1280ms-128ms/Z24-198-BP-01-5K-20260324133558.txt'

file_path1 ='../data/1/Z24-132/1280ms-128ms/Z24-132-BP-01-5K-20260107115443.txt'


df = pd.read_csv(file_path, sep='\t', encoding='gbk')
df_0 = pd.read_csv(file_path_0, sep='\t', encoding='gbk')
df_1 = pd.read_csv(file_path_1, sep='\t', encoding='gbk')
df_2 = pd.read_csv(file_path_2, sep='\t', encoding='gbk')
df_3 = pd.read_csv(file_path_3, sep='\t', encoding='gbk')
df_4 = pd.read_csv(file_path_4, sep='\t', encoding='gbk')

df_5 = pd.read_csv(file_path_5, sep='\t', encoding='gbk')
df_6 = pd.read_csv(file_path_6, sep='\t', encoding='gbk')
df_7 = pd.read_csv(file_path_7, sep='\t', encoding='gbk')
df_8 = pd.read_csv(file_path_8, sep='\t', encoding='gbk')
df_9 = pd.read_csv(file_path_9, sep='\t', encoding='gbk')

df_10 = pd.read_csv(file_path_10, sep='\t', encoding='gbk')
df_11 = pd.read_csv(file_path_11, sep='\t', encoding='gbk')
df_12 = pd.read_csv(file_path_12, sep='\t', encoding='gbk')
df_13 = pd.read_csv(file_path_13, sep='\t', encoding='gbk')
df_14 = pd.read_csv(file_path_14, sep='\t', encoding='gbk')

df1 = pd.read_csv(file_path1, sep='\t', encoding='gbk')

df = df[(df['时间s'] >=0) & (df['时间s'] <= 810)]
df_0 = df_0[(df_0['时间s'] >=0) & (df_0['时间s'] <= 810)]
df_1 = df_1[(df_1['时间s'] >=0) & (df_1['时间s'] <= 810)]
df_2 = df_2[(df_2['时间s'] >=0) & (df_2['时间s'] <= 810)]
df_3 = df_3[(df_3['时间s'] >=0) & (df_3['时间s'] <= 810)]
df_4 = df_4[(df_4['时间s'] >=0) & (df_4['时间s'] <= 810)]

df_5 = df_5[(df_5['时间s'] >=0) & (df_5['时间s'] <= 810)]
df_6 = df_6[(df_6['时间s'] >=0) & (df_6['时间s'] <= 810)]
df_7 = df_7[(df_7['时间s'] >=0) & (df_7['时间s'] <= 810)]
df_8 = df_8[(df_8['时间s'] >=0) & (df_8['时间s'] <= 810)]
df_9 = df_9[(df_9['时间s'] >=0) & (df_9['时间s'] <= 810)]

df_10 = df_10[(df_10['时间s'] >=0) & (df_10['时间s'] <= 810)]
df_11 = df_11[(df_11['时间s'] >=0) & (df_11['时间s'] <= 810)]
df_12 = df_12[(df_12['时间s'] >=0) & (df_12['时间s'] <= 810)]
df_13 = df_13[(df_13['时间s'] >=0) & (df_13['时间s'] <= 810)]
df_14 = df_14[(df_14['时间s'] >=0) & (df_14['时间s'] <= 810)]

df1 = df1[(df1['时间s'] >=0) & (df1['时间s'] <= 810)]
#df = df.iloc[::10, :]


features = ['燃压','氧喷前压', '燃喷前压', '氧流量', '燃流量', '氧阀门电流', '燃阀门电流','混合比']
target = '推力'

#X = df[features]
#y = df[target]
#一个时间点是一个样本
# 划分训练集和测试集
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #样本打乱
#split_time = 567
#train_df = df[df['时间s'] < split_time]
#test_df = df[df['时间s'] >= split_time]

train_df = pd.concat([df, df_0,df_1,df_2,df_3,df_4,df_5,df_6,
                      df_7,df_8,df_9,df_10,df_11,df_12,df_13,df_14],
                     ignore_index=True)
#print(type(df))
test_df = df1

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

fig = go.Figure()


fig.add_trace(go.Scatter(
    x=df1['时间s'],
    y=df1[target],
    mode='lines',
    name='真实推力',
    line=dict(color='royalblue', width=2)
))


fig.add_trace(go.Scatter(
    x=df1['时间s'],
    y=y_pred,
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
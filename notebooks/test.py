import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False


file_path = '../data/1/Z24-125/800s/Z24-125-BLS-01-5K-20260105135547.txt'# 800s
df = pd.read_csv(file_path, sep='\t', encoding='gbk')


start_t = float(input("开始："))
end_t = float(input("结束："))
test = df[(df['时间s'] > start_t) & (df['时间s'] < end_t)]

columns_drop = ['时间s']
test = test.drop(columns=columns_drop, errors='ignore')
#test = test.iloc[::10, :]

corr_matrix = test.corr()#pandas的.corr()函数，算出表格里所有变量两两之间的相关系数（默认:皮尔逊相关系数）
#corr_matrix = test.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title(f'推力器多变量相关性热力图({start_t}s - {end_t}s)', fontsize=14)
plt.tight_layout()
plt.show()
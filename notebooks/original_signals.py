import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#file_path = '../data/1/Z24-125/1280ms-128ms/Z24-125-BP-01-5K-20260105131755.txt'#1280ms-128ms
file_path = '../data/1/Z24-125/800s/Z24-125-BLS-01-5K-20260105135547.txt'#800s
#file_path = '../data/1/Z24-125/10s/Z24-125-BS-01-5K-20260105095016.txt'#10s

df = pd.read_csv(file_path, sep='\t', encoding='gbk')
df_n = df.iloc[::10, :] #降采样 5000HZ->500HZ

names = ['燃压', '推力', '氧喷前压', '燃喷前压', '氧阀门电流', '燃阀门电流', '氧流量', '燃流量', '混合比']
colors = ['blue', 'red', 'orange', 'green', 'pink', 'purple', 'black', 'brown', 'grey']

fig = make_subplots(
    rows=9, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,  # 子图之间垂直间距
    subplot_titles=names
)

for i in range(9):
    fig.add_trace(
        go.Scatter(             #折线图、散点图
            x=df_n['时间s'],
            y=df_n[names[i]],
            line=dict(color=colors[i], width=1.5)
        ),
        row=i+1, col=1
    )

fig.update_layout(
    height=1200,
    template='plotly_white',
    showlegend=False # 隐藏图例
)

fig.update_xaxes(title_text="时间 (s)", row=9, col=1)

fig.show()


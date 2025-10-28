import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("cluster_final.csv")

#데이터 전처리
s = df['객단가구간'].astype(str).str.normalize('NFKC').str.strip()
df['객단가구간'] = pd.to_numeric(s.str.extract(r'^(\d+)')[0], errors='coerce')

#데이터 정제 및 분석용 데이터프레임 생성
analysis_cols = ['재방문고객비중', '객단가구간']
df_analysis = df[analysis_cols].copy()
df_analysis = df_analysis.replace(-999999.9, np.nan)
df_analysis['재방문고객비중'] = pd.to_numeric(df_analysis['재방문고객비중'], errors='coerce')
df_analysis = df_analysis.dropna()

#데이터 스케일링 (표준화)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_analysis)

#데이터 샘플링
sample_indices = np.random.choice(range(len(df_scaled)), size=150, replace=False)
df_sampled = df_scaled[sample_indices]

#계층적 클러스터링
linked = linkage(df_sampled, method='ward')

#덴드로그램 시각화
plt.figure(figsize=(15, 7))
dendrogram(linked,
            orientation='top',
            # labels= # 여기에 라벨을 넣을 수 있지만, 150개는 너무 많아 생략합니다.
            distance_sort='descending',
            show_leaf_counts=True)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("cluster_final.csv")

#분석용 데이터 준비 및 정제
analysis_cols = ['신규고객비중', '재방문고객비중']
df_analysis = df[analysis_cols].copy()

df_analysis = df_analysis.replace(-999999.9, np.nan)
df_analysis['신규고객비중'] = pd.to_numeric(df_analysis['신규고객비중'], errors='coerce')
df_analysis['재방문고객비중'] = pd.to_numeric(df_analysis['재방문고객비중'], errors='coerce')

original_count = len(df_analysis)
df_analysis = df_analysis.dropna()
removed_count = original_count - len(df_analysis)

#데이터 스케일링
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_analysis)

#최적의 K 찾기 - 엘보우 방법 (Elbow Method)
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method (고객 충성도 분류)')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.show()

#최종 클러스터링 실행
OPTIMAL_K = 3 

kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=10, random_state=42)
kmeans.fit(df_scaled)

df_analysis['cluster'] = kmeans.labels_

#클러스터링 결과 분석
print("\n" + "="*30)
print("  클러스터별 평균 특성")
print("="*30)

# '재방문고객비중'이 높은 순으로 정렬
cluster_summary = df_analysis.groupby('cluster')[analysis_cols].mean().sort_values(by='재방문고객비중', ascending=False)
print(cluster_summary)

print("\n" + "="*30)
print("  클러스터별 매장 수")
print("="*30)
print(df_analysis['cluster'].value_counts().sort_index())


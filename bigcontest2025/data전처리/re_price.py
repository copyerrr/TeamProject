import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("cluster_final.csv")

#데이터 전처리
s = df['객단가구간'].astype(str).str.normalize('NFKC').str.strip()
df['객단가구간'] = pd.to_numeric(s.str.extract(r'^(\d+)')[0], errors='coerce')

#분석용 데이터 준비 및 정제
analysis_cols = ['재방문고객비중', '객단가구간']
df_analysis = df[analysis_cols].copy()

df_analysis = df_analysis.replace(-999999.9, np.nan)
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
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# 엘보우(팔꿈치) 그래프 시각화
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method (고객 가치 분류)')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.show()

#최적의 K 찾기 - 실루엣 스코어 (Silhouette Score)
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# 실루엣 스코어 그래프 시각화
plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--', color='b')
plt.title('Silhouette Score for Different k values')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

#최종 클러스터링 실행 및 결과 분석 (K=3)
OPTIMAL_K = 3 

kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=10, random_state=42)
kmeans.fit(df_scaled)
df_analysis['cluster'] = kmeans.labels_
print(f"\nK={OPTIMAL_K}로 클러스터링 실행 완료.")

print("\n" + "="*30)
print("  클러스터별 평균 특성")
print("="*30)
cluster_summary = df_analysis.groupby('cluster')[analysis_cols].mean().sort_values(by='객단가구간', ascending=True)
print(cluster_summary)
print("\n" + "="*30)
print("  클러스터별 매장 수")
print("="*30)
print(df_analysis['cluster'].value_counts().sort_index())




#스캐터 차트 시각화
plt.figure(figsize=(12, 8))
        
scatter_plot = sns.scatterplot(
    data=df_analysis,
    x='재방문고객비중',
    y='객단가구간',
    hue='cluster',     
    palette='Set1',   
    s=50,              
    alpha=0.7
)

scatter_plot.invert_yaxis() 

plt.title('매장 유형 클러스터링 (고객 가치 분류)', fontsize=20)
plt.xlabel('재방문고객비중 (%) (→ 높을수록 좋음)', fontsize=14)
plt.ylabel('객단가구간 (← 높을수록 좋음, 1점에 가까움)', fontsize=14)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_score

plt.rcParams['font.family'] = 'Malgun Gothic'
df = pd.read_csv('merged_data_korean_columns.csv')

# feature 컬럼 설정
feature_columns = ['거주이용고객비율', '직장이용고객비율', '유동인구이용고객비율']
sanggwon_col = '상권'

# 데이터 필터링
df_valid = df.copy()
for col in feature_columns:
    df_valid = df_valid[(df_valid[col].notna()) & (df_valid[col] != -999999.9)]

X = df_valid[['거주이용고객비율', '직장이용고객비율', '유동인구이용고객비율']]

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 엘보우 기법
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled) 
    inertia.append(kmeans.inertia_) 

plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method') 
plt.show()

# 실루엣 계수
sil_scores = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

plt.plot(K, sil_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Optimal k')
plt.show()
import pandas as pd
import requests
from geopy.distance import geodesic
from tqdm import tqdm
import numpy as np
import time

# 카카오맵에서 주소를 입력했을 때, 첫번째 주소 없으면 2번째 주소의 위도,경도 가져오는 함수
def get_kakao_coords_safer(api_key, address, store_name, cache):
    cleaned_store_name = str(store_name).split('*')[0].strip()
    query1 = f"{address} {cleaned_store_name}" if cleaned_store_name else str(address)

    if query1 in cache:
        return cache[query1]

    def fetch_coords(url, query_params):
        try:
            time.sleep(0.05)
            headers = {"Authorization": f"KakaoAK {api_key}"}
            response = requests.get(url, headers=headers, params=query_params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get('documents'):
                first = result['documents'][0]
                return (float(first['y']), float(first['x']))
        except requests.exceptions.RequestException:
            pass
        return None

    coords = fetch_coords("https://dapi.kakao.com/v2/local/search/address.json", {"query": query1})
    if coords is None:
        coords = fetch_coords("https://dapi.kakao.com/v2/local/search/keyword.json", {"query": query1})
    
    if coords:
        cache[query1] = coords
        return coords

    query2 = str(address)
    if query2 in cache:
        return cache[query2]

    if query1 != query2:
        coords = fetch_coords("https://dapi.kakao.com/v2/local/search/address.json", {"query": query2})
        if coords is None:
            coords = fetch_coords("https://dapi.kakao.com/v2/local/search/keyword.json", {"query": query2})
        
        if coords:
            cache[query1] = coords
            cache[query2] = coords
            return coords
        
    cache[query1] = None
    if query1 != query2: cache[query2] = None
    return None

# 위도, 경도 거리 계산 후 거리가 3km 이상이면 상권 제외
def calculate_distance_km(coords1, coords2):
    if coords1 and coords2:
        dist = geodesic(coords1, coords2).kilometers
            
        if dist >= 3:
            return None
        else:
            return dist
    return None

def main():
    KAKAO_REST_API_KEY = "" # 카카오 맵 API
    df = pd.read_csv('./data/cluster_final.csv', encoding='utf-8')
    
    coord_cache = {}
    tqdm.pandas(desc="전체 가맹점 좌표 변환 중")

    # 가맹점들의 좌표를 함수 호출하여 가져와 가맹점_좌표라는 컬럼에 추가
    df['가맹점_좌표'] = df.progress_apply(
        lambda row: get_kakao_coords_safer(KAKAO_REST_API_KEY, row['가맹점주소'], row['가맹점명'], coord_cache),
        axis=1
    )
    
    df_districts = df(subset=['상권']).copy()
    
    # 상권 좌표 생성 정확한 상권 주소가 없다면 {역} 을 붙여서 좌표 생성
    center_coords_map = {}
    for district_name, group in tqdm(df_districts.groupby('상권'), desc="상권 중심 결정 중"):
        region = group['가맹점지역'].iloc[0]
        center_coords = None

        station_address = f"{region} {district_name}역"
        center_coords = get_kakao_coords_safer(KAKAO_REST_API_KEY, station_address, "", coord_cache)

        if center_coords is None:
            district_address = f"{region} {district_name}"
            center_coords = get_kakao_coords_safer(KAKAO_REST_API_KEY, district_address, "", coord_cache)

        if center_coords is None:
            for _, row in group.iterrows():
                if row['가맹점_좌표'] is not None:
                    center_coords = row['가맹점_좌표']
                    break
        
        center_coords_map[district_name] = center_coords

    # 함수 호출하여 가맹점_좌표, 상권_좌표 를 거리계산하여 상권중심_거리_km 컬럼 생성 및 3km 이상이면 추가 x
    df_districts['상권_좌표'] = df_districts['상권'].map(center_coords_map)
    df_districts['상권중심_거리_km'] = df_districts.apply(
        lambda row: calculate_distance_km(row['가맹점_좌표'], row['상권_좌표']),
        axis=1
    ).round(2)
    
    df = df.merge(
        df_districts[['상권_좌표', '상권중심_거리_km']],
        left_index=True, right_index=True, how='left'
    )
    
    df.to_csv('상권_가맹점_거리.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
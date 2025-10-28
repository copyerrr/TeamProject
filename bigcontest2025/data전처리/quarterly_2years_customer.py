import pandas as pd
import numpy as np

def analyze_core_customers(csv_path="cluster_final.csv"):

    df = pd.read_csv(csv_path)

    #분기 매핑
    quarter_map = {
        202301:'2023_Q1', 202302:'2023_Q1', 202303:'2023_Q1',
        202304:'2023_Q2', 202305:'2023_Q2', 202306:'2023_Q2',
        202307:'2023_Q3', 202308:'2023_Q3', 202309:'2023_Q3',
        202310:'2023_Q4', 202311:'2023_Q4', 202312:'2023_Q4',
        202401:'2024_Q1', 202402:'2024_Q1', 202403:'2024_Q1',
        202404:'2024_Q2', 202405:'2024_Q2', 202406:'2024_Q2',
        202407:'2024_Q3', 202408:'2024_Q3', 202409:'2024_Q3',
        202410:'2024_Q4', 202411:'2024_Q4', 202412:'2024_Q4'
    }

    #성/연령 컬럼
    age_gender_cols = [
        '남성20대이하고객비중', '남성30대고객비중', '남성40대고객비중', '남성50대고객비중', '남성60대이상고객비중',
        '여성20대이하고객비중', '여성30대고객비중', '여성40대고객비중', '여성50대고객비중', '여성60대이상고객비중'
    ]

    #기준년월 정규화(문자/하이픈 등 -> 정수 YYYYMM)
    df['기준년월'] = (
        df['기준년월']
        .astype(str)
        .str.replace(r'\D', '', regex=True)  # 숫자만 추출
        .astype(int)
    )

    #분기 컬럼 만들기 + 결측치 치환
    df['분기'] = df['기준년월'].map(quarter_map)
    df[age_gender_cols] = df[age_gender_cols].replace(-999999.9, np.nan)

    #분기별(가맹점/분기) 평균 계산 → 핵심고객(최대 비중 컬럼명)
    #분기가 매핑되지 않은 행(NaN)은 자동으로 제외
    quarterly_avg = df.groupby(['가맹점구분번호', '분기'], dropna=True)[age_gender_cols].mean()

    #idxmax 안전 처리(전부 NaN인 행 방어)
    quarterly_core_customer = quarterly_avg.apply(
        lambda row: row.idxmax() if row.notna().any() else np.nan, axis=1)
    quarterly_core_customer.name = '핵심고객'

    #분기별을 열로 피벗
    final_df = quarterly_core_customer.unstack('분기')
    if final_df is None or final_df.empty:
        #피벗 결과가 비었으면(분기 매핑 실패/데이터 없음) 대비
        final_df = pd.DataFrame(index=df['가맹점구분번호'].unique())

    #열 이름 리네임
    final_df = final_df.rename(columns=lambda q: f"{q} 핵심고객" if pd.notna(q) else q)

    #2개년 전체(가맹점별) 평균 → 핵심고객
    total_avg = df.groupby('가맹점구분번호')[age_gender_cols].mean()
    total_core_customer = total_avg.apply(
        lambda row: row.idxmax() if row.notna().any() else np.nan, axis=1)
    total_core_customer.name = '2개년_핵심고객'

    #가맹점명 결합
    store_names = (
        df[['가맹점구분번호', '가맹점명']]
        .drop_duplicates()
        .set_index('가맹점구분번호')
    )

    final_df = store_names.join(final_df, how='left').join(total_core_customer, how='left')
    final_df = final_df.reset_index().rename(columns={'index': '가맹점구분번호'})

    #보기 좋게 정렬(가맹점명 → 구분번호)
    sort_cols = [c for c in ['가맹점명', '가맹점구분번호'] if c in final_df.columns]
    final_df = final_df.sort_values(by=sort_cols).reset_index(drop=True)

    return final_df


final_df = analyze_core_customers("cluster_final.csv")


final_df.to_csv('quarterly_2years_customer.csv', index=False, encoding='utf-8-sig')
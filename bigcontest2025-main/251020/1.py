import pandas as pd
import numpy as np
from scipy import stats

# ============================================================================
# 데이터 로드 및 전처리
# ============================================================================
df = pd.read_csv('merged_data_korean_columns.csv')

print(f"총 데이터: {len(df):,}행")
print(f"총 가맹점: {df['가맹점구분번호'].nunique():,}개")

# 구간을 점수로 변환
range_mapping = {
    '1_상위1구간': 6,
    '2_10-25%': 5,
    '3_25-50%': 4,
    '4_50-75%': 3,
    '5_75-90%': 2,
    '6_90%초과(하위 10% 이하)': 1
}

# 점수 컬럼 추가
df['매출금액_점수'] = df['매출금액구간'].map(range_mapping)
df['매출건수_점수'] = df['매출건수구간'].map(range_mapping)
df['고객수_점수'] = df['고객수구간'].map(range_mapping)
df['객단가_점수'] = df['객단가구간'].map(range_mapping)

# 결측치 처리
numeric_cols = ['배달매출금액비율', '재방문고객비중', '신규고객비중', 
                '거주이용고객비율', '직장이용고객비율', '유동인구이용고객비율']
for col in numeric_cols:
    df.loc[df[col] == -999999.9, col] = np.nan

# 연도 및 분기 컬럼 추가
df['년도'] = df['기준년월'] // 100
df['월'] = df['기준년월'] % 100

def get_quarter(month):
    if month in [1, 2, 3]:
        return 'Q1'
    elif month in [4, 5, 6]:
        return 'Q2'
    elif month in [7, 8, 9]:
        return 'Q3'
    else:
        return 'Q4'

df['분기'] = df['월'].apply(get_quarter)
df['연도_분기'] = df['년도'].astype(str) + '_' + df['분기']

quarter_order = ['2023_Q1', '2023_Q2', '2023_Q3', '2023_Q4', 
                 '2024_Q1', '2024_Q2', '2024_Q3', '2024_Q4']

# 각 가맹점의 분기별 데이터 개수 확인
all_stores = df['가맹점구분번호'].unique()

# 최소 2개 분기 이상 데이터가 있는 가맹점 선택
stores_with_data = []
for store_id in all_stores:
    store_quarters = df[df['가맹점구분번호'] == store_id]['연도_분기'].unique()
    if len(store_quarters) >= 2:
        stores_with_data.append(store_id)

print(f"\n분석 가능 가맹점: {len(stores_with_data):,}개 (최소 2개 분기 이상)")

# ============================================================================
# 통계 기반 임계값 계산
# ============================================================================
print("\n통계 기반 임계값 계산 중...")

# 매출금액 변화량 수집 (샘플링)
sample_size = min(1000, len(stores_with_data))
sample_stores = np.random.choice(stores_with_data, sample_size, replace=False)

sales_changes = []
revisit_changes = []
new_customer_changes = []

for store_id in sample_stores:
    store_data = df[df['가맹점구분번호'] == store_id]
    
    # 분기별 매출 평균
    quarterly_sales = store_data.groupby('연도_분기')['매출금액_점수'].mean()
    if len(quarterly_sales) >= 2:
        sales_changes.extend(quarterly_sales.diff().dropna().values)
    
    # 재방문율 변화
    quarterly_revisit = store_data.groupby('연도_분기')['재방문고객비중'].mean()
    if len(quarterly_revisit) >= 2:
        revisit_changes.extend(quarterly_revisit.diff().dropna().values)
    
    # 신규고객 변화
    quarterly_new = store_data.groupby('연도_분기')['신규고객비중'].mean()
    if len(quarterly_new) >= 2:
        new_customer_changes.extend(quarterly_new.diff().dropna().values)

# 표준편차 기반 임계값
sales_std = np.std(sales_changes)
revisit_std = np.std(revisit_changes)
new_customer_std = np.std(new_customer_changes)

# 매출 임계값 (표준편차의 배수)
SALES_LARGE_THRESHOLD = sales_std * 0.75  # 상위 25%
SALES_SMALL_THRESHOLD = sales_std * 0.25  # 상위 40%

# 재방문 임계값
REVISIT_LARGE_THRESHOLD = revisit_std * 0.75
REVISIT_SMALL_THRESHOLD = revisit_std * 0.25

# 신규고객 임계값
NEW_LARGE_THRESHOLD = new_customer_std * 0.75
NEW_SMALL_THRESHOLD = new_customer_std * 0.25

# Sequential Quarter 임계값 (조금 더 작은 변화 감지)
SQ_LARGE_THRESHOLD = sales_std * 0.6
SQ_SMALL_THRESHOLD = sales_std * 0.2

# 변동성 기반 계절성 임계값
SEASONALITY_VERY_STRONG = sales_std * 1.2
SEASONALITY_STRONG = sales_std * 0.8
SEASONALITY_MODERATE = sales_std * 0.4

print(f"  매출 대폭변화 기준: ±{SALES_LARGE_THRESHOLD:.3f}")
print(f"  매출 소폭변화 기준: ±{SALES_SMALL_THRESHOLD:.3f}")
print(f"  신규고객 대폭변화 기준: ±{NEW_LARGE_THRESHOLD:.3f}")
print(f"  계절성 강함 기준: {SEASONALITY_STRONG:.3f}")

# ============================================================================
# 추세 계산 함수 (p-value 기반)
# ============================================================================
def calculate_trend(series, significance_level=0.05):
    """시계열 데이터의 추세 계산 (통계적 유의성 기반)"""
    valid_data = series.dropna()
    
    if len(valid_data) < 2:
        return {
            'trend': None, 'slope': None, 'average': None,
            'first_value': None, 'last_value': None, 'change': None,
            'valid_count': len(valid_data), 'p_value': None
        }
    
    x = np.arange(len(valid_data))
    y = valid_data.values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # p-value 기반 추세 분류
    if p_value < significance_level:
        trend = '증가' if slope > 0 else '감소'
    else:
        trend = '평탄'
    
    first_value = valid_data.iloc[0]
    last_value = valid_data.iloc[-1]
    change = last_value - first_value
    
    return {
        'trend': trend,
        'slope': slope,
        'average': y.mean(),
        'first_value': first_value,
        'last_value': last_value,
        'change': change,
        'valid_count': len(valid_data),
        'p_value': p_value
    }


def analyze_quarter(store_data, year_quarter):
    """특정 분기 평균 계산"""
    quarter_data = store_data[store_data['연도_분기'] == year_quarter]
    
    if len(quarter_data) == 0:
        return None
    
    return {
        f'매출금액_평균_{year_quarter}': quarter_data['매출금액_점수'].mean(),
        f'고객수_평균_{year_quarter}': quarter_data['고객수_점수'].mean(),
        f'재방문_평균_{year_quarter}': quarter_data['재방문고객비중'].mean(),
        f'신규_평균_{year_quarter}': quarter_data['신규고객비중'].mean()
    }


def analyze_year_by_quarters(store_data, year):
    """연도별 분기 추세 분석 (p-value 기반)"""
    quarters = [f'{year}_Q1', f'{year}_Q2', f'{year}_Q3', f'{year}_Q4']
    
    sales_values = []
    customer_values = []
    revisit_values = []
    new_customer_values = []
    
    for q in quarters:
        q_data = store_data[store_data['연도_분기'] == q]
        if len(q_data) > 0:
            sales_values.append(q_data['매출금액_점수'].mean())
            customer_values.append(q_data['고객수_점수'].mean())
            revisit_values.append(q_data['재방문고객비중'].mean())
            new_customer_values.append(q_data['신규고객비중'].mean())
    
    sales_trend = calculate_trend(pd.Series(sales_values))
    customer_trend = calculate_trend(pd.Series(customer_values))
    revisit_trend = calculate_trend(pd.Series(revisit_values))
    new_customer_trend = calculate_trend(pd.Series(new_customer_values))
    
    return {
        f'매출금액_추세_{year}': sales_trend['trend'],
        f'매출금액_기울기_{year}': sales_trend['slope'],
        f'매출금액_평균_{year}': sales_trend['average'],
        f'매출금액_유효분기수_{year}': sales_trend['valid_count'],
        
        f'고객수_추세_{year}': customer_trend['trend'],
        f'고객수_기울기_{year}': customer_trend['slope'],
        f'고객수_유효분기수_{year}': customer_trend['valid_count'],
        
        f'재방문_추세_{year}': revisit_trend['trend'],
        f'재방문_평균_{year}': revisit_trend['average'],
        f'재방문_유효분기수_{year}': revisit_trend['valid_count'],
        
        f'신규_추세_{year}': new_customer_trend['trend'],
        f'신규_평균_{year}': new_customer_trend['average'],
        f'신규_유효분기수_{year}': new_customer_trend['valid_count']
    }


# ============================================================================
# 분기별 비교 분석
# ============================================================================
print("\n분기별 비교 분석 시작...")

quarterly_comparison_results = []

for idx, store_id in enumerate(stores_with_data):
    if idx % 500 == 0:
        print(f"  진행 중... {idx}/{len(stores_with_data)}")
    
    store_data = df[df['가맹점구분번호'] == store_id]
    first_record = store_data.iloc[0]
    
    # 전체 8개 분기 중 몇 개 있는지 확인
    available_quarters = store_data['연도_분기'].unique()
    total_quarters_available = len(available_quarters)
    
    result = {
        # 기본 정보
        '가맹점구분번호': store_id,
        '가맹점명': first_record['가맹점명'],
        '업종': first_record['업종'],
        '상권': first_record['상권'],
        '가맹점지역': first_record['가맹점지역'],
        '전체_보유분기수': total_quarters_available
    }
    
    # 각 분기별 평균값 계산
    for yq in quarter_order:
        quarter_analysis = analyze_quarter(store_data, yq)
        if quarter_analysis:
            result.update(quarter_analysis)
    
    # 연도별 추세 (p-value 기반)
    analysis_2023 = analyze_year_by_quarters(store_data, 2023)
    analysis_2024 = analyze_year_by_quarters(store_data, 2024)
    
    result.update(analysis_2023)
    result.update(analysis_2024)
    
    # 전년 동기 대비 (QoQ) + 통계 기반 추세 판정
    sequential_changes = []
    
    for quarter_num in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_2023 = f'2023_{quarter_num}'
        q_2024 = f'2024_{quarter_num}'
        
        # 매출금액 QoQ
        val_2023 = result.get(f'매출금액_평균_{q_2023}')
        val_2024 = result.get(f'매출금액_평균_{q_2024}')
        
        if (val_2023 is not None and val_2024 is not None and 
            not pd.isna(val_2023) and not pd.isna(val_2024)):
            
            qoq_change = val_2024 - val_2023
            result[f'매출금액_QoQ_{quarter_num}'] = qoq_change
            
            # 통계 기반 QoQ 추세 판정
            if qoq_change > SALES_LARGE_THRESHOLD:
                result[f'매출금액_QoQ_{quarter_num}_판정'] = '대폭개선'
            elif qoq_change > SALES_SMALL_THRESHOLD:
                result[f'매출금액_QoQ_{quarter_num}_판정'] = '개선'
            elif qoq_change >= -SALES_SMALL_THRESHOLD:
                result[f'매출금액_QoQ_{quarter_num}_판정'] = '유지'
            elif qoq_change >= -SALES_LARGE_THRESHOLD:
                result[f'매출금액_QoQ_{quarter_num}_판정'] = '악화'
            else:
                result[f'매출금액_QoQ_{quarter_num}_판정'] = '대폭악화'
        else:
            result[f'매출금액_QoQ_{quarter_num}_판정'] = '데이터부족'
        
        # 재방문 QoQ
        revisit_2023 = result.get(f'재방문_평균_{q_2023}')
        revisit_2024 = result.get(f'재방문_평균_{q_2024}')
        
        if (revisit_2023 is not None and revisit_2024 is not None and
            not pd.isna(revisit_2023) and not pd.isna(revisit_2024)):
            result[f'재방문_QoQ_{quarter_num}'] = revisit_2024 - revisit_2023
        
        # 신규고객 QoQ
        new_2023 = result.get(f'신규_평균_{q_2023}')
        new_2024 = result.get(f'신규_평균_{q_2024}')
        
        if (new_2023 is not None and new_2024 is not None and
            not pd.isna(new_2023) and not pd.isna(new_2024)):
            
            new_qoq_change = new_2024 - new_2023
            result[f'신규_QoQ_{quarter_num}'] = new_qoq_change
            
            # 통계 기반 신규 QoQ 판정
            if new_qoq_change > NEW_LARGE_THRESHOLD:
                result[f'신규_QoQ_{quarter_num}_판정'] = '대폭증가'
            elif new_qoq_change > NEW_SMALL_THRESHOLD:
                result[f'신규_QoQ_{quarter_num}_판정'] = '증가'
            elif new_qoq_change >= -NEW_SMALL_THRESHOLD:
                result[f'신규_QoQ_{quarter_num}_판정'] = '유지'
            elif new_qoq_change >= -NEW_LARGE_THRESHOLD:
                result[f'신규_QoQ_{quarter_num}_판정'] = '감소'
            else:
                result[f'신규_QoQ_{quarter_num}_판정'] = '대폭감소'
        else:
            result[f'신규_QoQ_{quarter_num}_판정'] = '데이터부족'
    
    # 직전 분기 대비 변화 (Sequential Quarter) - 통계 기반
    for i in range(len(quarter_order) - 1):
        current_q = quarter_order[i]
        next_q = quarter_order[i + 1]
        
        current_val = result.get(f'매출금액_평균_{current_q}')
        next_val = result.get(f'매출금액_평균_{next_q}')
        
        if (current_val is not None and next_val is not None and 
            not pd.isna(current_val) and not pd.isna(next_val)):
            
            seq_change = next_val - current_val
            result[f'매출금액_SQ_{current_q}→{next_q}'] = seq_change
            sequential_changes.append(seq_change)
            
            # 통계 기반 직전 분기 대비 판정
            if seq_change > SQ_LARGE_THRESHOLD:
                result[f'매출금액_SQ_{current_q}→{next_q}_판정'] = '급등'
            elif seq_change > SQ_SMALL_THRESHOLD:
                result[f'매출금액_SQ_{current_q}→{next_q}_판정'] = '상승'
            elif seq_change >= -SQ_SMALL_THRESHOLD:
                result[f'매출금액_SQ_{current_q}→{next_q}_판정'] = '보합'
            elif seq_change >= -SQ_LARGE_THRESHOLD:
                result[f'매출금액_SQ_{current_q}→{next_q}_판정'] = '하락'
            else:
                result[f'매출금액_SQ_{current_q}→{next_q}_판정'] = '급락'
        else:
            result[f'매출금액_SQ_{current_q}→{next_q}_판정'] = '데이터부족'
    
    # 전년 대비 추세 변화
    trend_2023 = analysis_2023['매출금액_추세_2023'] or '데이터부족'
    trend_2024 = analysis_2024['매출금액_추세_2024'] or '데이터부족'
    result['매출금액_추세_변화'] = f"{trend_2023} → {trend_2024}"
    
    # 재방문 추세 변화 추가
    revisit_trend_2023 = analysis_2023['재방문_추세_2023'] or '데이터부족'
    revisit_trend_2024 = analysis_2024['재방문_추세_2024'] or '데이터부족'
    result['재방문_추세_변화'] = f"{revisit_trend_2023} → {revisit_trend_2024}"
    
    # 신규 추세 변화 추가
    new_trend_2023 = analysis_2023['신규_추세_2023'] or '데이터부족'
    new_trend_2024 = analysis_2024['신규_추세_2024'] or '데이터부족'
    result['신규_추세_변화'] = f"{new_trend_2023} → {new_trend_2024}"
    
    # 데이터 품질 판정
    if result['매출금액_유효분기수_2023'] >= 4 and result['매출금액_유효분기수_2024'] >= 4:
        result['데이터품질'] = '우수'
    elif result['매출금액_유효분기수_2023'] >= 3 and result['매출금액_유효분기수_2024'] >= 3:
        result['데이터품질'] = '양호'
    elif result['매출금액_유효분기수_2023'] >= 2 and result['매출금액_유효분기수_2024'] >= 2:
        result['데이터품질'] = '보통'
    else:
        result['데이터품질'] = '부족'
    
    # 추세 판정
    trend_changes = {
        '증가 → 증가': '지속성장',
        '증가 → 평탄': '성장둔화',
        '증가 → 감소': '급격악화',
        '평탄 → 증가': '성장전환',
        '평탄 → 평탄': '지속정체',
        '평탄 → 감소': '하락전환',
        '감소 → 증가': '회복성공',
        '감소 → 평탄': '하락완화',
        '감소 → 감소': '지속하락'
    }
    
    if '데이터부족' in result['매출금액_추세_변화']:
        result['매출금액_추세_판정'] = '분석불가'
    else:
        result['매출금액_추세_판정'] = trend_changes.get(result['매출금액_추세_변화'], '알수없음')
    
    # 분기별 변동성 및 계절성 (통계 기반)
    quarterly_sales = [result.get(f'매출금액_평균_{yq}') for yq in quarter_order]
    quarterly_sales = [x for x in quarterly_sales if x is not None and not pd.isna(x)]
    
    if len(quarterly_sales) >= 3:
        result['매출금액_분기변동성'] = np.std(quarterly_sales)
        
        std = result['매출금액_분기변동성']
        
        # 통계 기반 계절성 판정
        if std > SEASONALITY_VERY_STRONG:
            result['계절성_판정'] = '매우강함'
        elif std > SEASONALITY_STRONG:
            result['계절성_판정'] = '강함'
        elif std > SEASONALITY_MODERATE:
            result['계절성_판정'] = '보통'
        else:
            result['계절성_판정'] = '약함'
        
        # 최고/최저 분기
        max_idx = quarterly_sales.index(max(quarterly_sales))
        min_idx = quarterly_sales.index(min(quarterly_sales))
        available_quarter_list = [yq for yq in quarter_order if result.get(f'매출금액_평균_{yq}') is not None and not pd.isna(result.get(f'매출금액_평균_{yq}'))]
        result['최고분기'] = available_quarter_list[max_idx]
        result['최저분기'] = available_quarter_list[min_idx]
        result['분기간_최대차이'] = max(quarterly_sales) - min(quarterly_sales)
    else:
        result['계절성_판정'] = '분석불가'
    
    # 연속 상승/하락 분기 수
    if len(sequential_changes) > 0:
        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0
        
        for change in sequential_changes:
            if change > 0:
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
            elif change < 0:
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_up = 0
                consecutive_down = 0
        
        result['최대_연속상승분기수'] = max_consecutive_up
        result['최대_연속하락분기수'] = max_consecutive_down
    
    quarterly_comparison_results.append(result)

# DataFrame 변환
qtr_df = pd.DataFrame(quarterly_comparison_results)

print(f"\n✅ 분기별 비교 분석 완료: {len(qtr_df):,}개 가맹점")

# 결과 저장
output_file = 'quarterly_final_analysis.csv'
qtr_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {output_file}")

# 통계 요약
print("\n" + "="*70)
print("📊 데이터 품질 분포")
print("="*70)
print(qtr_df['데이터품질'].value_counts())

print("\n" + "="*70)
print("📊 추세 판정 분포")
print("="*70)
print(qtr_df['매출금액_추세_판정'].value_counts())

print("\n" + "="*70)
print("📊 계절성 판정 분포")
print("="*70)
print(qtr_df['계절성_판정'].value_counts())

print("\n" + "="*70)
print("✅ 분석 완료!")
print("="*70)

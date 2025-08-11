import datetime
from pathlib import Path
from make_care_report import *

def get_last_7_dates():
    today = datetime.date.today()
    return [(today - datetime.timedelta(days=i))
            for i in reversed(range(7))]

def collect_weekly_results():
    weekly = []
    for date in get_last_7_dates():
        memory_score, fluency_score, coherence_score,emotional_score,complexity_score = score_model(date)
        # (memory_score, memory_reason,
        #  fluency_score, fluency_reason,
        #  coherence_score, coherence_reason,
        #  emotional_score, emotional_reason,
        #  complexity_score, complexity_reason) = score_model(date)

        weekly.append({
            "date": date,
            "memory_score":    memory_score,
            # "memory_reason":   memory_reason,
            "fluency_score":   fluency_score,
            # "fluency_reason":  fluency_reason,
            "coherence_score": coherence_score,
            # "coherence_reason":coherence_reason,
            "emotional_score": emotional_score,
            # "emotional_reason":emotional_reason,
            "complexity_score":complexity_score,
            # "complexity_reason":complexity_reason,
        })
    return weekly

# 3) 일주일치 변화를 요약하는 프롬프트 생성 및 GPT 호출
def generate_weekly_report(weekly):
    # 테이블 형태로 줄 단위 문자열 생성
    table_lines = []
    for day in weekly:
        line = (
            f"{day['date']}: "
            f"회상 {day['memory_score']}, "
            f"유창성 {day['fluency_score']}, "
            f"일관성 {day['coherence_score']}, "
            f"정서 {day['emotional_score']}, "
            f"문장 {day['complexity_score']}"
        )
        table_lines.append(line)
    table_text = "\n".join(table_lines)

    prompt = f"""
당신은 인지 케어 서비스 사용자의 일주일 간 대화 평가를 종합하는 치매 케어 전문가입니다.
왜, 무엇을, 어떻게의 구조로 자세히 설명해주세요
아래는 최근 7일간 날짜별 주요 평가 점수입니다:

{table_text}

1) 각 항목별(회상·유창성·일관성·정서·문장)로 일주일 간 변화 추이를 요약해 주세요. 수치적 변화보다는 추이에 따른 분석에 집중해주세요.
2) 전반적인 인지·언어 기능의 개선 또는 악화 여부를 평가해 주세요.
3) 보호자에게 권장할 케어 포인트를 3가지 제안해 주세요.

결과는 다음 형식으로 작성해 주세요:

---
[MINDI 주간 대화 인지 평가 보고서]

• 변화 요약:
…

• 종합 코멘트:
…

• 권장 케어 포인트:
1) …
2) …
3) …
---
"""
    return call_gpt(prompt)

# 4) 메인 실행 예시
if __name__ == "__main__":
    weekly_results = collect_weekly_results()
    report = generate_weekly_report(weekly_results)
    print(weekly_results)
    print(report)

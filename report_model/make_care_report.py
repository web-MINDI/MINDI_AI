from openai import OpenAI
import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = OpenAI(
    api_key=api_key
)

def call_gpt(prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content

# 1. 회상 충실도 평가
def evaluate_memory_fidelity(yesterday_text, today_conversations):
    prompt = f"""
    당신은 인지 케어 서비스 사용자의 인지 회상 능력을 평가하는 치매 케어 전문가입니다.

    오늘 사용자와 나눈 대화:
    {today_conversations}

    어제 사용자와 나눈 대화:
    {yesterday_text}
    
    다음 기준에 따라 **1~5점**으로 회상 충실도를 평가해 주세요.
    - 어제 나눈 대화를 오늘 얼마나 정확히 기억했는지
    - 구체적인 기억 재현의 세부 정도
    
    결과 형식:
    점수: X  
        """
    return call_gpt(prompt)



# 2. 언어 유창성 평가
def evaluate_fluency(conversation):
    prompt = f"""
    당신은 인지 케어 서비스 사용자의 언어 표현 능력을 평가하는 언어병리 전문가입니다.

    평가 대상 대화:
    {conversation}
    
    다음 기준에 따라 **1~5점**으로 언어 유창성을 평가해 주세요.
    - 단어 다양성의 정도
    - 단어 반복 및 음절 중단 발생 빈도
    - 복합 문장 사용 비율
    
    결과 형식:
    점수: X  
    """
    return call_gpt(prompt)


# 3. 맥락 일관성 평가
def evaluate_coherence(question, answer):
    prompt = f"""
    당신은 인지 케어 서비스 사용자의 인지적 대화 능력을 평가하는 치매 케어 전문가입니다.

    질문-답변 쌍들:
    {question}
    
    다음 기준에 따라 **1~5점**으로 맥락 일관성을 평가해 주세요.
    - 질문 의도 이해 정도
    - 주제 적합성 유지 정도
    - 대화의 논리적 연결성
    
    결과 형식:
    점수: X  
    """
    return call_gpt(prompt)


# 4. 정서적 반응성 평가
def evaluate_emotionality(conversation):
    prompt = f"""
    당신은 인지 케어 서비스 사용자의 치매 예방을 위한 정서·언어 통합 평가 전문가입니다.

    평가 대상 대화:
    {conversation}
    
    다음 기준에 따라 **1~5점**으로 정서적 반응성을 평가해 주세요.
    - 감정 표현 빈도
    - 감정 어휘 사용 다양성
    - 정서 표현의 질
    
    결과 형식:
    점수: X  
    """
    return call_gpt(prompt)


# 5. 문장 구성력 평가
def evaluate_sentence_complexity(conversation):
    prompt = f"""
    당신은 인지 케어 서비스 사용자의 언어 구성 능력을 분석하는 언어병리학자입니다.

    평가 대상 대화:
    {conversation}
    
    다음 기준에 따라 **1~5점**으로 문장 구성력을 평가해 주세요.
    - 문장 길이와 다양성
    - 주제 확장 시도 빈도
    - 문법적 복잡성
    
    결과 형식:
    점수: X  
    """
    return call_gpt(prompt)


def extract_result(evaluation_text):
    score = 0
    reason = None

    for line in evaluation_text.splitlines():
        line = line.strip()
        if line.startswith("점수"):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                score = int(parts[1].strip())
        elif line.startswith("이유"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                reason = parts[1].strip()

    return int(score)

def score_model(today):
    yesterday = today - datetime.timedelta(days=1)

    today_str = today.strftime("%Y%m%d")
    yesterday_str = yesterday.strftime("%Y%m%d")

    base_path = Path("C:/Users/user0102/PycharmProjects/Bert/logs")
    all_files = list(base_path.glob("*.txt"))

    today_file = next((f for f in all_files if today_str in f.name), None)
    yesterday_file = next((f for f in all_files if yesterday_str in f.name), None)

    if not today_file or not yesterday_file:
        raise FileNotFoundError(f"날짜에 해당하는 파일을 찾을 수 없습니다.")

    def read_lines(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    today_conversation = read_lines(today_file)
    yesterday_conversation = read_lines(yesterday_file)
    today_reminiscence = today_conversation[:6]
    today_questions = today_conversation[::2]
    today_answers = today_conversation[1::2]

    memory = evaluate_memory_fidelity(today_reminiscence, yesterday_conversation)
    fluency = evaluate_fluency(today_conversation)
    coherence = evaluate_coherence(today_questions, today_answers)
    emotional = evaluate_emotionality(today_conversation)
    complexity = evaluate_sentence_complexity(today_conversation)

    # memory_score, memory_reason = extract_result(memory)
    # fluency_score, fluency_reason = extract_result(fluency)
    # coherence_score, coherence_reason = extract_result(coherence)
    # emotional_score, emotional_reason = extract_result(emotional)
    # complexity_score, complexity_reason = extract_result(complexity)

    memory_score = extract_result(memory)
    fluency_score = extract_result(fluency)
    coherence_score = extract_result(coherence)
    emotional_score = extract_result(emotional)
    complexity_score = extract_result(complexity)

    # print(f"[{today}]\n")
    # print(f" {memory_score} \n {fluency_score}  \n {coherence_score} \n {emotional_score} \n {complexity_score} \n \n")

    #return memory_score, memory_reason, fluency_score, fluency_reason, coherence_score, coherence_reason, emotional_score, emotional_reason, complexity_score, complexity_reason
    return memory_score, fluency_score,  coherence_score,  emotional_score, complexity_score


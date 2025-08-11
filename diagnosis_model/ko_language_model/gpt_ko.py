from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# GPT API 키 설정 (환경변수에서 로드)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

def check_link(messages: list[str]) -> int:
    client = OpenAI(
        api_key=api_key
    )

    if len(messages) != 2:
        raise ValueError("messages must contain exactly two utterances: [utterance_A, utterance_B]")

    utterance_a, utterance_b = messages

    system_prompt = (
        "당신은 대화의 자연스러움을 평가하는 전문가입니다. "
        "두 개의 발화가 주어졌을 때, 두 발화가 주제적으로 얼마나 자연스럽게 이어지는지를 "
        "1점부터 5점까지 평가해주세요.\n"
        "1점 = 전혀 관련 없음\n"
        "5점 = 주제가 매우 자연스럽게 이어짐\n\n"
        "항상 다음과 같은 형식으로만 응답해야 합니다:\n\n"
        "점수: <숫자>\n"
        "이유: <간단한 설명>"
    )

    user_prompt = (
        f"발화 A: \"{utterance_a}\"\n"
        f"발화 B: \"{utterance_b}\"\n\n"
        "이 두 발화의 주제적 연결 정도를 1~5점 사이로 평가해주세요.\n\n"
        "반드시 아래 형식으로만 응답하세요:\n"
        "점수: <숫자>\n"
        "이유: <간단한 설명>"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )

    reply = response.choices[0].message.content.strip()
    # print("[GPT 응답]:", reply)

    # Score 추출
    import re
    match = re.search(r"점수:\s*([1-5])", reply)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"GPT 응답에서 점수를 추출할 수 없습니다: {reply}")
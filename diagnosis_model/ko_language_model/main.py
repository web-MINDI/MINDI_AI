from ko_language_model.final_KoBERT import *
from ko_language_model.gpt_ko import *
import openai
import json
import subprocess
from datetime import datetime
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# GPT API 키 설정 (환경변수에서 로드)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

openai.api_key = api_key

def count_korean_fruits_vegetables(user_text: str) -> dict:
    system_prompt = (
        "당신은 한국어 문장에서 언급된 과일과 채소 이름만을 추출하는 도우미입니다.\n"
        "동일한 항목은 한 번만 세며(중복 제거), 반드시 아래와 같은 JSON 형식으로만 응답하세요:\n"
        '{"count": <과일과 채소의 총 개수>, "items": ["사과", "당근", ...]}'
    )

    user_prompt = f'사용자 입력: "{user_text}"\n\n이 문장에서 언급된 과일과 채소를 추출해주세요.'


    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=150
    )

    reply = response.choices[0].message.content.strip()
    # print("[GPT 응답]:", reply)

    reply = reply.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(reply)
        return result
    except Exception as e:
        print(f"JSON 파싱 실패: {e}")
        return {"count": -1, "items": []}


def convert_mp4_to_wav(input_path, output_path, sample_rate=16000):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path
    ]

    subprocess.run(command, check=True)

def evaluate_answer(question: str, answer: str, user_answer: str) -> dict:
    system_prompt = (
        "당신은 사용자의 답변이 정답과 의미적으로 일치하는지를 평가하는 역할입니다.\n"
        "단어 선택이나 발음이 조금 다르더라도 의미가 같다면 맞는 답변으로 판단합니다.\n"
        "그러나 의미가 틀리거나 무관한 경우는 오답으로 판단합니다.\n"
        "항상 아래의 JSON 형식 중 하나로만 정확히 응답하세요:\n\n"
        '{"score": 1}  # 의미가 일치할 경우\n'
        '{"score": 0}  # 의미가 일치하지 않을 경우\n'
        "그 외의 말은 절대 하지 마세요."
    )

    user_prompt = (
        f"질문: {question}\n"
        f"정답: {answer}\n"
        f"사용자 답변: {user_answer}\n\n"
        "사용자 답변이 의미적으로 정답과 일치하면 score를 1로, 그렇지 않으면 0으로 JSON 형식으로만 응답하세요."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )

    reply = response.choices[0].message.content.strip()
    # print("[GPT answer]:", reply)

    try:
        return json.loads(reply)
    except Exception as e:
        print(f"JSON 파싱 실패: {e}")
        return {"score": -1}


def ko_language_model(answer1, answer2, answer3, answer4, answer5, answer7, answer8, answer9,answer10,answer11, answer12,
                   answer13, answer14, answer15, answer16, answer17, answer18, answer19, answer20, answer21):
    now = datetime.now()

    year = f"{now.year}년"
    month = f"{now.month}월"
    day = f"{now.day}일"

    weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    weekday = weekdays[now.weekday()]

    if 3 <= now.month <= 5:
        season = "봄"
    elif 6 <= now.month <= 8:
        season = "여름"
    elif 9 <= now.month <= 11:
        season = "가을"
    else:
        season = "겨울"

    score = 0
    Orientation = [
        ("지금은 몇 년도인가요?", year, answer1),
        ("지금은 몇 월인가요?", month, answer2),
        ("오늘은 며칠인가요?", day, answer3),
        ("오늘은 무슨 요일인가요?", weekday, answer4),
        ("지금은 어떤 계절인가요?", season, answer5),
    ]

    Attention = [
        ("제가 말하는 숫자를 순서대로 따라 말해주세요: 6 9 7 3", "6 9 7 3", answer7),
        ("제가 말하는 숫자를 순서대로 따라 말해주세요: 5 7 2 8 4", "5 7 2 8 4", answer8),
        ("'사과-꿀-당근-딸기'를 거꾸로 말해주세요.", "딸기 당근 꿀 사과", answer9),
    ]

    Memory = [
        ("아까 말씀드린 사람의 이름은 무엇이었나요? 1. 영수 2. 민수 3. 진수", "영수", answer11),
        ("그 사람은 무엇을 타고 갔나요? 1. 버스 2. 오토바이 3. 자전거", "자전거", answer12),
        ("그 사람은 어디로 갔나요? 1. 공원 2. 운동장 3. 들판", "공원", answer13),
        ("그 사람은 무엇을 했나요? 1. 농구 2. 축구 3. 야구", "야구", answer14),
        ("그 사람은 몇 시에 시작했나요? 1. 10시 2. 11시 3. 3시", "11시", answer15),
    ]

    Executive_Function = [
        ("[고양이, 토끼, 강아지, 물고기] 다음 단어 중 하나만 다른 성격입니다. 다른 성격인 단어를 말해주세요.", "물고기", answer16),
        ("가, 나, ?, 라, ? — 알파벳 순서를 보고 빈칸에 들어갈 두 글자를 말해주세요.", "다 마", answer17)
    ]

    # answer 18
    Count = [
        ("지금부터 과일이나 채소의 이름을 최대한 많이 말해주세요.")
    ]

    for (q, a, u) in Orientation:
        result = evaluate_answer(q, a, u)
        score += result["score"]

    for (q, a, u) in Attention:
        result = evaluate_answer(q, a, u)
        score += result["score"]

    sentence_memory = evaluate_answer("조금 전에 외워달라고 했던 문장을 다시 말해보세요.","민수는 자전거를 타고 공원에 가서 11시에 야구를 했습니다.",answer10)
    if sentence_memory["score"] == 1:
        score += 10
    else:
        for (q, a, u) in Memory:
            result = evaluate_answer(q, a, u)
            score += result["score"]*2

    for (q, a, u) in Executive_Function:
        result = evaluate_answer(q, a, u)
        score += result["score"]*2

    # Bert_MLM
    Bert_rst = 0
    Bert_only = 0
    link_only = 0
    # bert 1
    bert_score = KoBERT_final(answer18)
    Bert_only += bert_score
    # check link (연관성 점수)
    utterance_a = "공을 생각하면 어떤 기억이 떠오르시나요? 예전에 어떻게 사용하셨는지도 말씀해 주세요."
    utterance_b = answer18

    link_score = check_link([utterance_a, utterance_b])
    link_only += link_score
    if link_score < 2:
        Bert_rst += bert_score * 0.5
    else:
        Bert_rst += bert_score

    # BERT2
    bert_score = KoBERT_final(answer19)
    Bert_only += bert_score
    # check link (연관성 점수)
    utterance_a = "쿠키를 보면 어떤 기억이 떠오르시나요? 맛 표현이나 취향도 말씀해 주세요."
    utterance_b = answer19

    link_score = check_link([utterance_a, utterance_b])
    link_only += link_score
    if link_score < 2:
        Bert_rst += bert_score * 0.5
    else :
        Bert_rst += bert_score

    # Bert3
    bert_score = KoBERT_final(answer20)
    Bert_only += bert_score
    # check link (연관성 점수)
    utterance_a = "핸드폰을 보면 어떤 기억이 떠오르시나요? 예전에는 어떻게 사용하셨는지도 말씀해 주세요."
    utterance_b = answer20


    link_score = check_link([utterance_a, utterance_b])
    link_only += link_score
    if link_score < 2:
        Bert_rst += bert_score * 0.5
    else:
        Bert_rst += bert_score

    result = count_korean_fruits_vegetables(answer21)
    if result["count"] > 10:
        score += 2
    elif result["count"] > 5:
        score += 1
    else:
        score += 0


    return Bert_rst, score, link_only, Bert_only
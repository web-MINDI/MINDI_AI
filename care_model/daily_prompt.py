import os
from .utils import *
import openai

def get_daily_prompt(category_topic):
    return f"""
당신은 노인과 따뜻한 일상 대화를 나누는 AI 도우미입니다.
- 친근하고 간단한 문장으로 질문해 주세요.
- 한 번에 하나의 질문만 하세요.
- 어르신이 기억을 회상하거나 감정을 표현할 수 있도록 도와주세요.
- 너무 길거나 복잡한 문장은 피해주세요.
- 오늘의 대화 주제는 {category_topic}입니다.
"""

def ask_gpt(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

def daily(category):
    filename = f"answer.wav"

    filepath = os.path.join("data/", filename)

    user_text = single_wav_to_text(filepath).strip()
    print(f"사용자(STT): {user_text}")

    messages = [
        {"role": "system", "content": get_daily_prompt(category)},
        {"role": "user", "content": user_text}
    ]
    reply = ask_gpt(messages)
    print(f"GPT: {reply}")
    messages.append({"role": "assistant", "content": reply})
    return messages
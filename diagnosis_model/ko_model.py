import os

from ko_data_make_csv import *
from ko_acoustic_model.main import *
from ko_language_model.main import *

# answer1 = single_wav_to_text("ko_wav/ans1.wav")
# answer2 = single_wav_to_text("ko_wav/ans2.wav")
# answer3 = single_wav_to_text("ko_wav/ans3.wav")
# answer4 = single_wav_to_text("ko_wav/ans4.wav")
# answer5 = single_wav_to_text("ko_wav/ans5.wav")
# answer6 = single_wav_to_text("ko_wav/ans6.wav")
# answer7 = single_wav_to_text("ko_wav/ans7.wav")
# answer8 = single_wav_to_text("ko_wav/ans8.wav")
# answer9 = single_wav_to_text("ko_wav/ans9.wav")
# answer10 = single_wav_to_text("ko_wav/ans10.wav")
# answer11 = single_wav_to_text("ko_wav/ans11.wav")
# answer12 = single_wav_to_text("ko_wav/ans12.wav")
# answer13 = single_wav_to_text("ko_wav/ans13.wav")
# answer14 = single_wav_to_text("ko_wav/ans14.wav")
# answer15 = single_wav_to_text("ko_wav/ans15.wav")
# answer16 = single_wav_to_text("ko_wav/ans16.wav")
# answer17 = single_wav_to_text("ko_wav/ans17.wav")
# answer18 = single_wav_to_text("ko_wav/ans18.wav")
# answer19 = single_wav_to_text("ko_wav/ans19.wav")
# answer20 = single_wav_to_text("ko_wav/ans20.wav")
# answer21 = single_wav_to_text("ko_wav/ans21.wav")

# acoustic_path = [
#     #"C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans1.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans2.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans3.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans4.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans5.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans6.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans7.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans8.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans9.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans10.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans11.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans12.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans13.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans14.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans15.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans16.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans17.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans18.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans19.wav",
#     # "C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans20.wav","C:/Users/user0102/PycharmProjects/Bert/ko_wav/ans21.wav"]

# 실행용
# if __name__ == "__main__":
#     threshold_table = [
#     {"age_min": 0, "age_max": 59, "thresholds": {"ele": 24, "middle": 26, "high": 28, "univ": 30}},
#     {"age_min": 60, "age_max": 69, "thresholds": {"ele": 23, "middle": 25, "high": 27, "univ": 28}},
#     {"age_min": 70, "age_max": 79, "thresholds": {"ele": 21, "middle": 24, "high": 25, "univ": 27}},
#     {"age_min": 80, "age_max": 200, "thresholds": {"ele": 18, "middle": 24, "high": 22, "univ": 24}},
# ]

#     # dementia_rst 0=정상 1=치매
#     dementia_rst = 0

#     language_score,check_score = ko_language_model(answer1, answer2, answer3, answer4, answer5, answer7, answer8, answer9,answer10, answer11, answer12, answer13, answer14, answer15, answer16, answer17, answer18, answer19, answer20, answer21)
#     acoustic_score = ko_acoustic_model(acoustic_path)

#     score = language_score + acoustic_score + check_score
#     print(f"설문 점수 : {check_score}/24 , 언어모델 점수 : {language_score}/6 , 음향모델 점수 : {acoustic_score}/6 , 총점 : {score}")

#     with open('user_info/user1.json', 'r', encoding='utf-8') as f:
#         user_info = json.load(f)

#     age = user_info['age']
#     education = user_info['education']

#     threshold = None
#     for row in threshold_table:
#         if row["age_min"] <= age <= row["age_max"]:
#             threshold = row["thresholds"].get(education)
#             break
#     print(f"{threshold} 이상이면 정상임, 님 점수: {score}")
#     if score <= (threshold-1):
#         dementia_rst = 1
#         print(">>>치매<<<")
#     elif (score >= (threshold-1)) and (score <= (threshold+ 1)):
#         dementia_rst = 0
#         print("---정상이지만 유의 바람---")
#     else:
#         dementia_rst = 0
#         print(">>>정상<<<")


# 함수
# def main():
#     # dementia_rst 0=정상 1=치매
#     dementia_rst = 0

#     language_socre = language_model(answer1, answer2, answer3, answer4, answer5, answer7, answer8, answer9, answer11, answer12, answer13, answer14, answer15, answer16, answer17, answer18, answer19, answer20, answer21)
#     acoustic_score = Hwi("wav/ans1.wav","wav/ans2.wav","wav/ans3.wav","wav/ans4.wav","wav/ans5.wav","wav/ans6.wav","wav/ans7.wav","wav/ans8.wav","wav/ans9.wav","wav/ans10.wav"
#                         ,"wav/ans11.wav","wav/ans12.wav","wav/ans13.wav","wav/ans14.wav","wav/ans15.wav","wav/ans16.wav","wav/ans17.wav","wav/ans18.wav","wav/ans19.wav","wav/ans20.wav")

#     score = language_socre + acoustic_score
#     print(f"language score : {score} / acoustic score : {acoustic_score} / total score : {score}")

#     with open('user_info/user1.json', 'r', encoding='utf-8') as f:
#         user_info = json.load(f)

#     age = user_info['age']
#     education = user_info['education']

#     threshold = None
#     for row in threshold_table:
#         if row["age_min"] <= age <= row["age_max"]:
#             threshold = row["thresholds"].get(education)
#             break

#     if score <= threshold:
#         dementia_rst = 1
#         print("치매")
#     else:
#         print("정상")

#     return dementia_rst
def report_by_score(acoustic_score_vit, acoustic_score_lgbm, language_score_BERT, language_score_gpt):
    evaluate_good_list = []
    evaluate_bad_list = []
    result_good_list = []
    result_bad_list = []
    
    if acoustic_score_vit >= 50:
        evaluate = "말의 속도가 지나치게 느리지 않으며, 억양 변화(감정 기복)가 자연스럽게 나타납니다."
        result = "현재 속도와 억양을 유지하되, '감정 표현 연습'으로 기쁨·흥미·놀라움 등 다양한 억양을 더욱 풍부히 해보세요."
        evaluate_good_list.append(evaluate)
        result_good_list.append(result)
    else:
        evaluate = "말이 지나치게 느리거나 빨라서 의사소통이 어렵고, 억양 변화가 거의 없어 단조롭게 들립니다."
        result = "간단한 문장을 부드럽게, 또 강하게 번갈아가며 발화해 보세요. 또한, 시간을 재며 말의 속도를 점진적으로 맞춰주세요"
        evaluate_bad_list.append(evaluate)
        result_bad_list.append(result)

    if acoustic_score_lgbm >= 50:
        evaluate = "음성 주파수 상에서 목소리의 떨림이 안정적으로 그려집니다."
        result = "자연스럽게 노래를 부르며 목소리의 떨림 정도를 유지해주세요."
        evaluate_good_list.append(evaluate)
        result_good_list.append(result)
    else:
        evaluate = "음성 주파수 상에서 목소리의 떨림이 불안정합니다."
        result = "'으-으-으' 소리를 길게 내면서 목소리가 안정되도록 호흡을 맞춰 보세요."
        evaluate_bad_list.append(evaluate)
        result_bad_list.append(result)

    if language_score_BERT >= 50:
        evaluate = "대화에서 사용된 어휘가 풍부하며, 단문과 복문을 적절히 섞어 자연스러운 문장 구성이 확인됩니다."
        result = "현재 수준을 유지·향상시키려면 '주제별 단어 카드'를 활용해 익숙치 않은 어휘를 일주일에 5개씩 추가 학습해 보세요."
        evaluate_good_list.append(evaluate)
        result_good_list.append(result)
    else:
        evaluate = "문장에 반복 단어가 많고, 대부분 단문 위주로 표현해 어휘 폭과 문장 구조가 단순합니다."
        result = "한 문장에 '왜?', '어떻게?' 질문을 덧붙여 복문으로 확장해 말하는 연습을 해보세요."
        evaluate_bad_list.append(evaluate)
        result_bad_list.append(result)

    if language_score_gpt >= 50:
        evaluate = "질문의 핵심 의도를 정확히 파악하고, 적절한 범위 내에서 답변을 구사합니다."
        result = "현재 말하기 습관을 유지하되, 간단한 퀴즈나 게임으로 의도 파악 능력을 즐겁게 강화하세요."
        evaluate_good_list.append(evaluate)
        result_good_list.append(result)
    else:
        evaluate = "질문 의도와 무관한 답변이 잦거나, 질문 핵심을 놓쳐 엉뚱한 정보를 제공하는 경향이 있습니다."
        result = "질문을 들은 후 5초간 '제가 이해한 질문은 ~입니다'라고 다시 말하는 습관을 들여보세요. 또는 질문의 핵심 키워드를 찾는 훈련을 해야합니다."
        evaluate_bad_list.append(evaluate)
        result_bad_list.append(result)

    return evaluate_good_list, evaluate_bad_list, result_good_list, result_bad_list

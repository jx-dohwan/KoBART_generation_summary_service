import re


def preprocess_sentence(sentence, v2=False):
    if v2==False:
      #sentence = sentence.lower() # 텍스트 소문자화
      sentence = re.sub(r"삭제된 메시지입니다.", "", sentence)
      sentence = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+[/ㄱ-ㅎㅏ-ㅣ]', '', sentence) # 여러개 자음과 모음을 삭제한다.
      sentence = re.sub(r'\[.*?\] \[.*?\]', ',', sentence) # 여러개 자음과 모음을 삭제한다.
      sentence = re.sub(r"[^가-힣a-z0-9#@,-\[\]\(\)]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
      sentence = re.sub(r'[" "]+', " ", sentence) # 여러개 공백을 하나의 공백으로 바꿉니다.
      #sentence = sentence.strip() # 문장 양쪽 공백 제거
      sentence = sentence.split(',')            

    if v2 :
      #sentence = re.sub(r'\[[^)]*\]', '', sentence) 
      sentence = sentence.strip() # 문장 양쪽 공백 제거

    return sentence

def remove_empty_pattern(text_list): # 빈 ''가 발생하여 [sep]가 중복되는 경우가 생겨서 이를 제거하기 위함
    return [x for x in text_list if x.strip() != '' ]#and not re.search(r'^\'\'$', x)

def preprocess_result(sentence, v2=True):
  result = []
  for i in range(len(sentence)):
    result.append(preprocess_sentence(sentence[i], v2).lower())
  return result






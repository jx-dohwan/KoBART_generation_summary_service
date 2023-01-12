
## 💡프로젝트 소개
###  base project : [Base Project link](https://github.com/AIFFEL-NLP-PROJECT/Aiffelthon)
```
1️⃣ 주제 : KoBAR 한국어 대화문 생성요약 서비스
2️⃣ 목표 : KoBART로 학습한 한국어 대화문 요약 모델을 FASTAPI를 통해서 웹 서비스로 배포
3️⃣ 설명 : 기존의 Aiffel에서 진행한 대화문 생성요약 프로젝트를 베이스로 하여 KoBART 모델을 개선 및 서빙을 진행할 것 
```

---

## 🗓️ 프로젝트 구현 진행사항
|기존 서비스|개선 서비스|진행사항(%)|
|:---------:|:----------:|:------:|
|테스트 데이터셋 미구축|테스트 데이터셋 구축||
|일반 rouge사용|rouge를 한국어 전용으로 개선||
|모델 미배포|Huggingface에 배포|100%|
|mecab으로 길이측정|model tokenizer로 길이측정|100%|
|Python Script->Jupyter Notebook|Python Script->Jupyter Notebook||
|모델링까지만 진행|자체 웹에 Serving||
|BOS, SEP 미사용|BOS, SEP사용|100%|

---

## 🗓️ 프로젝트 진행
---

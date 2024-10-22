## 이미지 생성 도구: 프롬프트 분석 및 이미지 생성

이 앱은 사용자가 원하는 이미지를 생성하는 데 도움을 주는 도구입니다. 사용자가 입력한 프롬프트를 분석하고, Imagen 모델에 적합한 프롬프트로 재해석하여 이미지를 생성합니다.

**핵심 기능:**

* **프롬프트 분석:** 사용자 입력 프롬프트를 분석하여 Imagen 모델에 적합한 positive prompt와 negative prompt로 분리합니다.
* **다양한 분석 옵션:** 프롬프트 분석 방법을 선택할 수 있으며, 각 옵션은 Gemini 모델을 사용하여 상세하게 분석합니다.
* **이미지 생성:** 분석된 프롬프트를 사용하여 Imagen 3 모델을 통해 이미지를 생성합니다.
* **이미지 확대:** 생성된 이미지를 클릭하여 확대하여 볼 수 있습니다.

**사용 방법:**

1. **프롬프트 입력:** 원하는 이미지를 설명하는 프롬프트를 입력합니다. 예를 들어, "고양이가 햇빛 아래 잔디밭에서 낮잠을 자고 있다"와 같은 프롬프트를 입력할 수 있습니다.
2. **분석 옵션 선택:** 원하는 분석 옵션을 선택합니다.
* "사용자가 입력한 프롬프트를 Imagen에서 그릴 수 있게 좋은 프롬프트로 변경해줘"
* "프롬프트를 더 상세하게 설명해줘"
* "프롬프트에서 주요 키워드를 추출해줘"
3. **분석 실행:** "분석" 버튼을 클릭하여 프롬프트 분석을 실행합니다.
4. **결과 확인:** 분석 결과가 JSON 형식으로 표시됩니다. Positive prompt와 negative prompt가 표시되며, Imagen 모델에 사용할 수 있는 상세한 프롬프트를 확인할 수 있습니다.
5. **이미지 생성:** 분석 결과를 사용하여 Imagen 3 모델에서 이미지를 생성합니다. 생성된 이미지는 두 개의 컬럼으로 표시됩니다.
6. **이미지 확대:** 이미지를 클릭하여 확대하여 볼 수 있습니다.

**기술 스택:**

* **Streamlit:** 웹 앱 프레임워크
* **Vertex AI:** Google Cloud의 머신러닝 서비스
* **Imagen 3:** Google에서 개발한 이미지 생성 모델
* **Gemini:** Google에서 개발한 대규모 언어 모델
* **dotenv:** 환경 변수를 관리하기 위한 라이브러리

**주의 사항:**

* 이 앱은 Google Cloud Platform의 Vertex AI를 사용합니다. Vertex AI를 사용하기 위해서는 Google Cloud Platform 계정이 필요합니다.
* 이미지 생성에는 시간이 소요될 수 있습니다.

**향후 계획:**

* 사용자 편의성 향상 및 인터페이스 개선
* 추가적인 이미지 생성 모델 지원
* 프롬프트 분석 기능 강화

**설치 방법**
0. GCP Credential이 설치되어 있어야 합니다.
1. 해당 App을 내려받습니다. 
2. virtualenv .venv
3. source .venv/bin/activate
4. pip install -r requirements.txt
5. echo EXPORT PROJECT_ID=<your project id> >> .env
6. echo EXPORT LOCATION=us-central1 >> .env
7. echo EXPORT OUTPUT_URI=gs://<your ourput bucket name> >> .env
8. streamlit run main.py

**라이선스:**

이 앱은 Apache 2.0 라이선스에 따라 배포됩니다.

**저작권:**

Copyright (c) [작성자 이름]

**기여:**

이 앱은 다음 사람들의 기여로 이루어졌습니다:

* [작성자 이름]

**감사의 말:**

이 앱은 Google Cloud Platform과 Vertex AI 팀의 지원으로 개발되었습니다.
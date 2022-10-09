# 2022 PNU CSE Tech Week - Deep Learning Challenge
# Track 1. Landmark Classification

본 repo는 2022 PNU CSE Tech Week 사전 행사인 Deep Learning Challenge - Track 1. Landmark Classification의 참여를 돕고자 생성하였습니다.

본 repo에서는 간단한 Class 명세 및 train / test 예제 코드가 제공 되며, 자세한 행사 안내는 [Tech Week 홈페이지](https://sites.google.com/pusan.ac.kr/pnucse-techweek/home) 또는 [DLC-T1 공식 안내문](https://docs.google.com/document/d/1PVFJimsrsI_e9ye-1NfXhdgDlyimA--yBl92OGhi7zU/edit) 을 참고 해 주세요.

PyTorch를 기반으로 작성되었으며, 아래 내용 중 TensorFlow 사용 시 문제가 되는 점이 있거나 문의사항이 있을 경우, [Tech Week Q&A](https://sites.google.com/pusan.ac.kr/pnucse-techweek/qa)를 통해 문의 부탁드립니다.


## 1. Submission Format

- [ExampleClassifierModule.py](ExampleClassifierModule.py) 모듈은 예제 ExampleClassifier class를 포함하고 있습니다. 해당 클래스의 docstring과 comment를 참고하세요.
- 이용 가능한 Library는 **PyTorch**와 **TensorFlow**로 제한합니다.
- 각 팀은 알고리즘 개발에 사용한 **train / test dataset을 함께 제출**해야 합니다. Dataset structure는 아래 설명된 구조와 동일해야 합니다.
- 각 팀에서 제출하는 데이터셋은 반드시 각 class 별 **train 100장, test 50장 이상**이어야 합니다.
- 개발 과정과 각 팀의 전략을 담은 간단한 보고서[(양식)](https://docs.google.com/document/d/1PVFJimsrsI_e9ye-1NfXhdgDlyimA--yBl92OGhi7zU/edit)도 같이 제출해 주세요.
- 각 팀은 훈련을 진행 한 내용 확인이 가능한 Google Colab ipynb 파일과, 훈련이 완료 된 weight 파일을 함께 제출해야 합니다.
- Colab으로 Training이 불가능 한 경우, 보고서에 Local Machine을 이용해 개발한 내용, 과정, 출력물을 포함해 주세요.
- 추가적인 설치가 필요한 package의 경우 requirements.txt 파일에 추가하여 제출합니다. 평가 이전에 _pip install -r requirements.txt_ 명령을 통해 설치합니다.
- 자세한 제출 Format 예시는 다음과 같습니다.

```commandline
submission/                         // Submission root
├── dataset                         // Dataset root (Dataset format은 아래 참조)
│   ├── test                        // Test set root
│   │   ├── cse                     // 50+ images
│   │   ├── hh                      // 50+ images
│   │   ├── rg                      // 50+ images
│   │   ├── wb                      // 50+ images
│   │   └── wjj                     // 50+ images
│   └── train                       // Train set root
│       ├── cse                     // 100+ images
│       ├── hh                      // 100+ images
│       ├── rg                      // 100+ images
│       ├── wb                      // 100+ images
│       └── wjj                     // 100+ images
├── ExampleClassifierModule.py      // Your classifier module
├── model.pt                        // Pretrained weight
├── report.docx                     // 보고서 (파일 형식은 반드시 docx가 아니어도 됨)
├── requirements.txt                // (Optional) 추가 설치가 필요한 Package list
└── train.ipynb                     // (Optional) Train 과정이 기록 된 Google Colab ipynb
```


## 2. Dataset Structure

- Train 과 Test 데이터셋 구조는 동일합니다.
- Root directory (train 또는 test) 아래에는 5개 Landmark에 대한 subdir이 존재합니다.
- 각 클래스는 **cse(0, 컴퓨터공학관), hh(1, Humanities Hall, 인문관), rg(2, 무지개문), wb(3, 웅비의 탑), wjj(4, 운죽정)** 에 해당합니다.
- 예제 데이터의 경우 7장의 train, 3장의 test image가 각 class 별로 포함되어 있으며 png format 입니다.
- 다만 실제 데이터의 경우 png, jpg, bmp 등의 다양한 이미지 포맷이 포함되어 있을 수 있습니다.
- 예제 데이터의 경우 이름의 형식을 통일해 두었으나, 실제 데이터의 경우 임의의 파일 이름을 가질 수 있습니다.
- 이미지의 최소 사이즈는 가로 x 세로 기준 320 x 240, 최대 사이즈는 1280 x 960 로 제한합니다.

```commandline
dataset
├── test
│   ├── cse
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   ├── hh
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   ├── rg
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   ├── wb
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   └── wjj
│       ├── 00000000.png
│       ├── ...
│       └── XXXXXXXX.png
└── train
    ├── cse
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    ├── hh
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    ├── rg
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    ├── wb
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    └── wjj
        ├── 00000000.png
        ├── ...
        └── XXXXXXXX.png
```


## 3. Evaluation

- 대규모 평가 데이터셋 수집의 어려움으로 인해, **각 팀에서 제출한 데이터셋 중 test set에서 각 class 별 50장을 random sampling 후 augmentation 하여 official test set을 생성**합니다.
- 각 팀에서 제출한 test set이 훈련에 사용되지 않았다는 보장이 없으므로, **test set sampling 중 augmentation을 실시하여 최대한 객관성을 확보**합니다.
- 정확도(%)가 높은 순으로 순위를 평가하며, 정확도(%) 소수 4째 자리까지 같은 경우 Pretrained weight의 크기가 작은 팀을 더 높은 순위로 평가 합니다.
- PyTorch를 이용해 개발 한 경우, official evaluation을 위해 반드시 제공 된 예시 test.py script의 수정 없이 제출 class가 evaluation 가능해야 합니다. 자세한 내용은 test.py를 참고하세요.
- TensorFlow를 이용해 개발 한 경우, class 명세를 지켜 구현 했을 경우 test script를 TensorFlow에 맞게 수정하면 동일한 형태로 평가가 가능할 것입니다.


## 4. Additional Notes

- PyTorch를 이용해 개발하면 운영진의 평가가 매우 수월해 집니다...^^^^^
- 문의사항은 모두와 공유 할 수 있게 [Tech Week Q&A](https://sites.google.com/pusan.ac.kr/pnucse-techweek/qa)를 통해 부탁드립니다.
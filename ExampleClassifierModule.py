"""
    PNU CSE TECH WEEK 2022

    PNU Deep Learning Challenge - Track 01. Landmark Classification

    This script contains classifier specification for challenge submission.
    It is recommended to inherit from this class to implement your classifier.

    Jinsun Park (jspark@pusan.ac.kr / viplab@pusan.ac.kr)
    Visual Intelligence and Perception Lab., CSE, PNU

    ======================================================================

    2022.10.09 - Initial release

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class ExampleClassifier(nn.Module):
    """
    DLC-T1 Submission을 위한 Class 명세
    """
    
    def __init__(self, path_data, pretrain=None):
        """
        1. 생성자는 아래와 같은 인자만 사용해야 합니다.
        :param path_data: Dataset root에 해당하는 경로
        :param pretrain: Evaluation에 필요한 Pretrained weight의 경로

        2. 데이터 처리를 위한 transform과 dataset을 반드시 생성자에서 초기화 해야 합니다.
        3. 모델은 반드시 self.build_model method 내부에서 선언해 사용합니다.
        4. forward method는 입력 이미지 x를 받아서 (class score) y를 반환합니다.
        y는 normalized(sum(y) == 1) / unnormalized(sum(y) != 1) 여부에 상관 없이,
        argmax(y)의 값이 class index를 반환 할 수 있으면 됩니다.
        5. train_model은 반드시 존재해야 하지만, overriding 가능합니다.
        6. eval_model은 반드시 존재 할 필요는 없습니다. 참고용으로 사용하세요.
        7. Official evaluation은 self.model과 self.dataset을 직접 접근하여
        진행합니다. 절대 두 변수 이름을 바꾸지 마세요.
        8. self.dataset을 ImageFolder 이외의 class로 구현 할 경우, 각 샘플은
        반드시 (image, label)을 반환해야 합니다. (train / eval 코드 참조)

        특별한 이유가 없다면, ExampleClassifier를 상속하여 본인의 알고리즘을
        구현 한 뒤 제출하기를 추천합니다.
        각 method에 대한 추가적인 설명은 각 method의 docstring을 참고 해 주세요.
        """
        super().__init__()

        # Please refer to dataset directory structure
        self.path_data = path_data

        if pretrain is not None:
            # For evaluation
            self.model = torch.load(pretrain)
        else:
            self.build_model()

        # Dataset loading에 적용하기 위한 transform은 생성자에서 선언
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        # 데이터셋 구조가 정해져 있으므로, ImageFolder class를 사용하기를 추천
        # 다른 class를 사용 할 경우 반드시 각 샘플은 (image, label)을 반환해야 함.
        self.dataset = ImageFolder(
            self.path_data,
            transform=self.transform
        )
        self.num_data = len(self.dataset)

    def build_model(self):
        """
        Code 점검의 편의를 위해 model 선언은 반드시 build_model 안에서 완료해야 합니다.
        build_model 외부에서 model을 변경하지 마세요. (제발 Plz ^^)
        """
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(16*32*32, 5),
        )

    def forward(self, x):
        """
        forward method는 입력 이미지 x를 받아서 (class score) y를 반환합니다.
        y는 normalized(sum(y) == 1) / unnormalized(sum(y) != 1) 여부에 상관 없이,
        argmax(y)의 값이 class index를 반환 할 수 있으면 됩니다.

        입력과 출력 부분 외에는 자유롭게 변형하여 사용하세요.

        :param x: 입력 영상 (Batch x Channel x Height x Width)
        :return: Class score (unnormalized or normalized)
        """
        # x: [Batch, Channel, Height, Width]
        # y: [Batch x Num_Class(5)]
        y = self.model(x)
        return y

    def train_model(self, config):
        """
        train_model은 반드시 존재해야 하지만, overriding 가능합니다.
        기본으로 제공되는 코드는 구현 참고용 입니다.
        config의 내용은 train.py를 참고 하세요.

        train_model의 호출 이후에는 self.model의 weight가 훈련 완료 된 상태로 간주합니다.

        :param config: dictionary containing all of the training parameters
        :return:
        """
        batch_size = config['batch_size']
        epochs = config['epochs']
        loss = config['loss']
        optim = config['optim']

        train_loader = DataLoader(self.dataset, batch_size=batch_size)

        print('Number of data : {}'.format(self.num_data))

        # 가장 기본적인 훈련 코드 구현의 예시
        self.model.train()
        for epoch in range(1, epochs+1):
            print('Epoch : {} / {}'.format(epoch, epochs))

            # 진행 상황을 보기 위한 tqdm 이용 예시
            pbar = tqdm(total=self.num_data, dynamic_ncols=True)

            for batch, sample in enumerate(train_loader):
                img, label = sample

                optim.zero_grad()

                output = self.forward(img)

                loss_val = loss(output, label)

                loss_val.backward()

                optim.step()

                pbar.set_description('Loss : {}'.format(loss_val.item()))
                pbar.update(batch_size)

            pbar.close()

    def eval_model(self):
        """
        Evaluation 참고용 코드 입니다. 반드시 존재 할 필요는 없으며,
        Official evaluation은 직접 self.model과 self.dataset을 접근하여 진행합니다.

        :return: 
        """
        eval_loader = DataLoader(self.dataset, batch_size=1)

        print('Number of data : {}'.format(self.num_data))

        pbar = tqdm(total=self.num_data, dynamic_ncols=True)

        self.model.eval()

        num_correct = 0

        for batch, sample in enumerate(eval_loader):
            img, label = sample

            output = self.forward(img)

            if label.item() == torch.argmax(output).item():
                num_correct += 1

            pbar.update(1)

        pbar.close()

        print('Accuracy : {:.4f} %'.format(num_correct * 100 / self.num_data))

"""
    PNU CSE TECH WEEK 2022

    PNU Deep Learning Challenge - Track 01. Landmark Classification

    This script contains a simple example training script.

    Jinsun Park (jspark@pusan.ac.kr / viplab@pusan.ac.kr)
    Visual Intelligence and Perception Lab., CSE, PNU

    ======================================================================

    2022.10.09 - Initial release

"""


# Your classifier class will be implemented from the module
from ExampleClassifierModule import ExampleClassifier

import argparse
import torch
from torch import nn

"""
argparse module을 이용해 다양한 argument를 사용하기를 추천합니다.
"""

parser = argparse.ArgumentParser(description='Simple training script')

parser.add_argument('--path_data', type=str, default='dataset/train',
                    help='Path to dataset')
parser.add_argument('--path_save', type=str, default='model.pt',
                    help='Path to save model')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train')
parser.add_argument('--optim', type=str, default='Adam',
                    help='Optimizer')
parser.add_argument('--loss', type=str, default='CrossEntropy',
                    help='Loss function')

args = parser.parse_args()


def main():
    # Classifier는 훈련시 반드시 아래와 같은 형태로 생성됩니다.
    classifier = ExampleClassifier(args.path_data)

    """
    Argument를 전달하는 방식은 수정해도 됩니다.
    """
    if args.optim == 'Adam':
        optim = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    else:
        # Use Adam for this example
        raise NotImplementedError

    if args.loss == 'CrossEntropy':
        loss = nn.CrossEntropyLoss(reduce='mean')
    else:
        # Use CrossEntropyLoss for this example
        raise NotImplementedError

    config = {
        'batch_size': args.batch_size,
        'optim': optim,
        'epochs': args.epochs,
        'loss': loss
    }

    print('\nArguments:')
    for k, v in args.__dict__.items():
        print('{} : {}'.format(k, v))
    print('\nTraining started\n')

    """
    classifier는 반드시 train_model method를 호출하여 훈련을 시작합니다.
    train_model의 실행이 완료 된 후, classifier.model은 훈련 된 weight를
    가지고 있다고 간주합니다.
    """
    classifier.train_model(config)

    print('\nTraining done\n')

    print('\nEvaluation started\n')

    classifier.eval_model()

    print('\nEvaluation done\n')

    """
    훈련이 끝난 뒤 반드시 model을 torch.save를 이용해 저장하여 제출하세요.
    """
    torch.save(classifier.model, args.path_save)

    print('Saved to {}'.format(args.path_save))


if __name__ == '__main__':
    main()

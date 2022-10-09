"""
    PNU CSE TECH WEEK 2002

    PNU Deep Learning Challenge - Track 01. Landmark Classification

    This script contains an example evaluation script.
    Your classifier class must work with this script for official evaluation.

    Jinsun Park (jspark@pusan.ac.kr / viplab@pusan.ac.kr)
    Visual Intelligence and Perception Lab., CSE, PNU

    ======================================================================

    2022.10.09 - Initial release

"""


import argparse
import importlib
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Simple evaluation script')

parser.add_argument('--path_data', type=str, default='dataset/test',
                    help='Path to dataset')
parser.add_argument('--path_module', type=str, default='./',
                    help='Path containing your module file')
parser.add_argument('--module_name', type=str, default='ExampleClassifierModule',
                    help='Your module name')
parser.add_argument('--class_name', type=str, default='ExampleClassifier',
                    help='Your class name')
parser.add_argument('--path_model', type=str, default='model.pt',
                    help='Path to pretrained model')

args = parser.parse_args()


def main():
    print('\nArguments:')
    for k, v in args.__dict__.items():
        print('{} : {}'.format(k, v))

    """
    제출한 class는 반드시 아래 소스코드를 수정하지 않고 평가가 정상적으로 동작해야 합니다.
    path_module 은 class를 구현 한 파일 (e.g., ExampleClassifierModule.py)이 저장 된 경로 이고,
    module_name은 해당 파일의 .py 확장자를 제외한 이름 (e.g., ExampleClassifierModule) 이며,
    class_name은 해당 파일 내부에 선언된 class 이름 (e.g., ExampleClassifier) 입니다.
    
    세 가지 변수를 본인 환경에 맞는 값으로 바꾸어 실행 하여 정상적으로 정확도 평가가 되면 제출 가능합니다.
    """

    # Dynamic import
    classifier_module = importlib.import_module(args.module_name,
                                                package=args.path_module)
    classifier_class = getattr(classifier_module, args.class_name)
    classifier = classifier_class(args.path_data, args.path_model)

    dataset = classifier.dataset

    # For statistics
    result = dict()
    for cls in dataset.classes:
        result[dataset.class_to_idx[cls]] = [0, 0]

    eval_loader = DataLoader(dataset, batch_size=1)

    classifier.model.eval()

    pbar = tqdm(total=classifier.num_data, dynamic_ncols=True)

    for batch, sample in enumerate(eval_loader):
        img, label = sample

        output = classifier.forward(img)

        pred = torch.argmax(output).item()
        gt = label.item()

        result[gt][1] += 1
        if pred == gt:
            result[gt][0] += 1

        pbar.update(1)

    pbar.close()

    # Log file generation
    current_time = time.strftime('%y%m%d_%H%M%S')
    path_log = '{}_log_eval_{}_{}.txt'.format(
        current_time, args.module_name, args.class_name
    )
    f_log = open(path_log, 'w')

    print('\n{:^8s} | {:^8s} | {:^8s} | {:^10s}'.format(
        'Class', 'Correct', 'Total', 'Acc(%)'
    ))
    correct_all = 0
    total_all = 0
    for k, v in result.items():
        cls = dataset.classes[k]
        correct = result[k][0]
        total = result[k][1]
        acc = 100. * correct / (total + 1e-8)
        print('{:^8s} | {:^8d} | {:^8d} | {:>10.4f}'.format(
            cls, correct, total, acc
        ))

        correct_all += correct
        total_all += total

        f_log.write('{},{},{},{:.4f}\n'.format(cls, correct, total, acc))

    acc_all = 100. * correct_all / (total_all + 1e-8)
    print('{:^8s} | {:^8d} | {:^8d} | {:>10.4f}'.format(
        'All', correct_all, total_all, acc_all)
    )
    f_log.write('{},{},{},{:.4f}'.format('All', correct_all, total_all, acc_all))
    f_log.close()


if __name__ == '__main__':
    main()

# GPND(Generative Probabilistic Novelty Detection with Adversarial Autoencoders)
[참고 코드](https://github.com/podgorskiy/GPND)
[논문](https://arxiv.org/abs/1807.02588)

#### 데이터
- Cifar같은 경우 반드시 https://www.cs.toronto.edu/~kriz/cifar.html 에서 CIFAR-10 Binary version을 다운받아야 함

## code 실행

1. python partition_cifar10.py를 실행하여 cifar10 데이터 만들기
2. python train_AAE_mrx.py를 실행하여 aae 학습
    - dataset = 'mnist' or 'fmnist' or 'cifar' 로 설정
    - modal = 'unimodal' or 'multimodal'로 설정
3. python novelty_detector_mrx.py를 실행
    - dataset = 'mnist' or 'fmnist' or 'cifar' 로 설정
    - modal = 'unimodal' or 'multimodal'로 설정
4. 한꺼번에 돌리는 코드 : schedule_mrx.py
    - 실행 코드 : python schedule_mrx.py 'cifar_multimodal' '0' '0' 'cifar' 10 'multimodal'
        - 'cifar' : subject folder
        - '0' : gpu number
        - '0' : test_fold_id (0~5 fold로  6개 중에 1개 test, 1개 valid, 나머지 train)
        - 'cifar' : dataset (mnist / fmnist / cifar)
        - 10 : num_epochs
        - 'multimodal' : modal type (unimodal / multimodal)
    - modal
        - 'unimodal' 일 경우 i=0~9까지 돌면서 각각i 가 normal , 나머지가 novelty
        - 'multimodal'일 경우 i=0~9까지 돌면서 i가 novelty, 나머지가 normal
    - percentages = [10,20,30,40,50] list안에 실험 돌릴 novelty_ratio 선언

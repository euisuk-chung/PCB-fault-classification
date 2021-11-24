# PCB-fault-classification
Multi-label Classification of PCB Faults by Image Processing

## Participants
- Kyoosung So [mons2us](https://github.com/mons2us)
- Euisuk Chung [chung_es](https://github.com/euisuk-chung)
- Yunseung Lee [yun-ss97](https://github.com/yun-ss97)


## 주의사항

### (1) Data Used
- Source : https://github.com/tangsanli5201/DeepPCB
- Used DeepPCB Dataset, a dataset contains 1,500 image pairs, each of which consists of a defect-free template image and an aligned tested image with annotations
    - Dataset includes positions of 6 most common types of PCB defects: `open`, `short`, `mousebite`, `spur`, `pin hole` and `spurious copper`. 

**(Example)**
```bash
├── dataset                    
    ├── group00041  
    │   ├──00041                  
    │   │   ├── 000410000_temp.jpg  - image with fault
    │   │   └── 000410000_test.jpg  - image w/o fault
    │   │
    │   └── 00041_not              
    │        └──000410000.txt       - label
    │
    └──  group12000
```

이외 폴더 구조는 동일하게 하면 됨

### (2) Model
- Uses Following Models
    - resnet18, resnet50, resnet101,
    - vgg11, vgg16, vgg19,
    - densenet121, densenet169, densenet201
    - EfficientNetB4, EfficientNetB5, EfficientNetB7

### (3) Hyper Parameter
- lr scheduler나 optimizer 같은 경우 src/train.py에서 직접 수정하면 됨<br>
- 본 프로젝트는 multisteplr 사용하고 있음<br>


## Usage:
### TRAINING
```bash
cd {dir_to_base_folder} 
python main.py --mode 'train' \
               --model_index {model index}
               --base_model 'plain_efficientnetb4' \ # plain_resnet50, plain_efficientnetb4, plain_efficientnetb5 ...
               --lr_type 'multi' \ # 'exp', 'cos', 'multi'
```

### TEST
```bash
python main.py --mode 'test' \
               --model_index {model index} \ # inference 진행하려는 model index
               --base_model 'plain_efficientnetb4' \ # 마찬가지로 학습한 모델과 동일하게... 아니면 오류남
               --tta # TTA를 사용할 경우는 이 argument를 명시해주면 됨. 안쓰면 TTA 안함. 현재는 [0 90 -90]도로 돌린 세개에 대해 예측해서 평균함
```

# PCB-fault-classification
Multi-label Classification of PCB Faults by Image Processing

## 주의사항

### (1) 폴더구조
- main.py의 base_dir, ckpt_path, pretrained_weights_dir는 각자 working directory 기반으로 맞출 것
- dataset 폴더에 DeepPCB 데이터셋 폴더 그대로 이동

(예시) 
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

### (2) 모델
다른 모델을 사용하는 경우,<br>
src/model.py에 해당 모델을 추가하면 되고(+import)<br>
utils/call_model.py에 앞서와 동일하게 코드 추가하면 됨.<br>

### (3) 하이퍼파라미터
lr scheduler나 optimizer 같은 경우 src/train.py에서 직접 수정하면 됨<br>
현재는 multisteplr 사용하고 있음<br>


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

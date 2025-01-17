
# Lambert-Enhanced-BERT🐣

**L**everaging **A**ttention **M**echanisms to improve the **BERT** fine-tuning model for encrypted traffic classification.

![](./img/LAMBERT.svg)

`Note:` This code is based on [UER-py](https://github.com/dbiir/UER-py) and [ET-BERT](https://github.com/linwhitehat/ET-BERT)

## 1. Introduction
Fine-tuning is an important part of a pre-training based approach. 
Lambert-Enhanced-BERT a novel fine-tuning model, which leverages its unique attention mechanisms to improve sequence loss. 
This improvement is particularly evident on datasets that have not been pre-trained.

## 2. Preparation
### 2.1 Clone the repository
```bash
git clone https://github.com/HungnotDumb/Lambert-Enhanced-BERT.git
cd Lambert-Enhanced-BERT
```
### 2.2 Create a new conda environment
```bash
conda create -n Lambert-Enhanced-BERT python=3.8.16
conda activate Lambert-Enhanced-BERT
```
### 2.3 Install Pytorch
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```
### 2.4 Install other dependencies
```bash
pip install -r requirements.txt
```
### 2.5 Configure environment variables
If you are in Linux, you can do this:
```bash
export PYTHON PATH="${PYTHONPATH}:./"
```
If you are in Windows, you can add the Lambert-Enhanced-BERT root path to the python environment variable by hand.

## 3. Dataset
### 3.1 Original Dataset
| Dataset          | Packet | Label |
|------------------|--------|-------|
| ISCX-VPN-Service | 60000  | 12    |
| ISCX-VPN-App     | 77163  | 17    |
| USTC-TFC         | 97115  | 20    |
| CSTNET-TLS1.3    | 581709 | 120   |
1. [ISCX-VPN](https://drive.google.com/drive/folders/1is609sosAdqf9YJAfwr72hBqM4OeNuZq?usp=sharing) (Contains ISCX-VPN-Service & ISCX-VPN-App)
2. [USTC-TFC](https://drive.google.com/file/d/1F09zxln9iFg2HWoqc6m4LKFhYK7cDQv_/view?usp=sharing)
3. [CSTNET-TLS1.3](https://drive.google.com/drive/folders/1is609sosAdqf9YJAfwr72hBqM4OeNuZq?usp=sharing)
4. Or you can also prepare your own datasets.
### 3.2 Processed Dataset
`Note:` The original traffic data needs to be preprocessed, and the preprocessing part is consistent with [ET-BERT](https://github.com/linwhitehat/ET-BERT)

After processing the data into the datasets folder, the file tree example is as follows:
```
├─iscx-app
│  └─packet
│     └─nolabel_test_dataset.tsv
│     └─test_dataset.tsv
│     └─train_dataset.tsv
│     └─valid_dataset.tsv
├─iscx-service
│  └─packet
│     └─nolabel_test_dataset.tsv
│     └─test_dataset.tsv
│     └─train_dataset.tsv
│     └─valid_dataset.tsv
└─ustc-tfc
│  └─packet
│     └─nolabel_test_dataset.tsv
│     └─test_dataset.tsv
│     └─train_dataset.tsv
│     └─valid_dataset.tsv
└─cstnet-tls1.3
   └─packet
      └─nolabel_test_dataset.tsv
      └─test_dataset.tsv
      └─train_dataset.tsv
      └─valid_dataset.tsv
```

## 3. Usage
### 3.1 Pre-training
```bash
python pre-training/pretrain.py --dataset_path dataset.pt --vocab_path models/encryptd_vocab.txt \
    --output_model_path models/pre-trained_model.bin \
     --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
    --total_steps 500000 --save_checkpoint_steps 10000 --batch_size 32 \
    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```
### 3.2 Fine-tuning
```bash
python fine-tuning/lambert.py \
    --pretrained_model_path models/pre-trained_model.bin \
    --output_model_path output/lambert/lambert.bin \
    --evaluate_output_path output/lambert/lambert.txt \
    --vocab_path models/encryptd_vocab.txt \
    --train_path datasets/ustc-tfc/packet/train_dataset.tsv \
    --dev_path datasets/ustc-tfc/packet/valid_dataset.tsv \
    --test_path datasets/ustc-tfc/packet/test_dataset.tsv \
    --epochs_num 20 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 2e-5
```
### 3.3 Inference
```bash
python inference/infer.py \
    --classifier_file fine-tuning/lambert.py \
    --classifier_name Lambert \
    --load_model_path output/lambert/lambert.bin \
    --test_path datasets/ustc-tfc/packet/test_dataset.tsv \
    --prediction_path output/lambert/infer.tsv \
    --vocab_path models/encryptd_vocab.txt \
    --embedding word_pos_seg --encoder transformer --mask fully_visible
```
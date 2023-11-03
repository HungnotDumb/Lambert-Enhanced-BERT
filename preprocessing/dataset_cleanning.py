#!/usr/bin/python3
#-*- coding:utf-8 -*-

from ml_classifier import unlabel_data

def deal_label():
    lower_label = [23,24,35,44,76,94,95]
    lower_label.extend([22,28,52,62,67,102,104])
    lower_label.sort()
    return lower_label

def deal_finetuning(excluding_label):
    dataset_path = "I:\\datasets\\cstnet-tls1.3\\"
    save_dataset_path = dataset_path
    with open(dataset_path+"train_dataset.tsv",'r') as f:
        train_data = f.read().split('\n')[1:]
    with open(dataset_path+"valid_dataset.tsv",'r') as f:
        valid_data = f.read().split('\n')[1:]
    with open(dataset_path+"test_dataset.tsv",'r') as f:
        test_data = f.read().split('\n')[1:]
    for label_number in excluding_label:
        train_pop_index = []
        valid_pop_index = []
        test_pop_index = []
        for index in range(len(train_data)):
            if str(label_number)+'\t' in train_data[index]:
                train_pop_index.append(index)
        for counter,index in enumerate(train_pop_index):
            index = index - counter
            train_data.pop(index)

        for index in range(len(valid_data)):
            if str(label_number)+'\t' in valid_data[index]:
                valid_pop_index.append(index)
        for counter,index in enumerate(valid_pop_index):
            index = index - counter
            valid_data.pop(index)

        for index in range(len(test_data)):
            if str(label_number)+'\t' in test_data[index]:
                test_pop_index.append(index)
        for counter,index in enumerate(test_pop_index):
            index = index - counter
            test_data.pop(index)
            
    label_number = 120
    count = 0
    while label_number > 105:
        for index in range(len(train_data)):
            data = train_data[index]
            i
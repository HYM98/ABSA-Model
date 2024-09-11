# coding: UTF-8
import numpy as np
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import random
from sklearn.utils import resample

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号



def build_dataset(config):
    # 划分数据集
    def read_and_split_data(path):
        print("数据集划分中......")
        with open(path, 'r', encoding='UTF-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        # 随机打乱数据
        random.shuffle(lines)
        # 划分数据集
        train_size = int(len(lines) * 0.8)
        dev_size = int(len(lines) * 0.1)
        train_data = lines[:train_size]
        dev_data = lines[train_size:train_size + dev_size]
        test_data = lines[train_size + dev_size:]
        return train_data, dev_data, test_data

    # 数据平衡处理
    def balance_data(data):
        negative = [line for line in data if line.split('\t')[1] == '0']
        neutral = [line for line in data if line.split('\t')[1] == '1']
        positive = [line for line in data if line.split('\t')[1] == '2']

        max_len = max(len(negative), len(neutral), len(positive))

        negative_upsampled = resample(negative, replace=True, n_samples=max_len, random_state=42)
        neutral_upsampled = resample(neutral, replace=True, n_samples=max_len, random_state=42)
        positive_upsampled = resample(positive, replace=True, n_samples=max_len, random_state=42)

        balanced_data = negative_upsampled + neutral_upsampled + positive_upsampled
        random.shuffle(balanced_data)

        return balanced_data

    # 处理数据集token
    def process_data(data):
        contents = []
        lengths = []
        # 首先统计所有数据的长度
        for line in tqdm(data):
            content, label = line.split('\t')
            token = config.tokenizer.tokenize(config.aspect) + ['[SEP]'] + config.tokenizer.tokenize(content) + [
                '[SEP]']
            token = ['[CLS]'] + token
            lengths.append(len(token))
        # 计算90%的分位数作为最优截断长度
        optimal_max_length = int(np.percentile(lengths, 70))
        if optimal_max_length > 300:
            optimal_max_length = 200
        print("阈值设置：", optimal_max_length)
        # 使用计算出的最优截断长度处理数据
        for line in tqdm(data):
            content, label = line.split('\t')
            token = config.tokenizer.tokenize(config.aspect) + ['[SEP]'] + config.tokenizer.tokenize(content) + [
                '[SEP]']
            token = ['[CLS]'] + token
            seq_len = len(token)
            if (seq_len < optimal_max_length):
                mask = [1] * seq_len + [0] * (optimal_max_length - seq_len)
                token = token + (['[PAD]'] * (optimal_max_length - seq_len))
            else:
                mask = [1] * optimal_max_length
                token = token[:optimal_max_length]
                seq_len = optimal_max_length
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            contents.append((token_ids, int(label), seq_len, mask))
        return contents

    def load_dataset(path):
        contents = []
        lengths = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = [line.strip() for line in f if line.strip()]
        # 首先计算token长度，以确定截断阈值
        for line in tqdm(data):
            content, label = line.split('\t')
            token = config.tokenizer.tokenize(config.aspect) + ['[SEP]'] + config.tokenizer.tokenize(content) + [
                '[SEP]']
            token = ['[CLS]'] + token
            lengths.append(len(token))
        # 计算90%的分位数作为最优截断长度
        optimal_max_length = int(np.percentile(lengths, 70))
        if optimal_max_length > 300:
            optimal_max_length = 200
        print("测试集阈值：", optimal_max_length)
        # 对数据进行处理
        for line in tqdm(data):
            content, label = line.split('\t')
            token = config.tokenizer.tokenize(config.aspect) + ['[SEP]'] + config.tokenizer.tokenize(content) + [
                '[SEP]']
            token = ['[CLS]'] + token
            seq_len = len(token)
            if (seq_len < optimal_max_length):
                mask = [1] * seq_len + [0] * (optimal_max_length - seq_len)
                token = token + (['[PAD]'] * (optimal_max_length - seq_len))
            else:
                mask = [1] * optimal_max_length
                token = token[:optimal_max_length]
                seq_len = optimal_max_length
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train, dev, test = read_and_split_data(config.train_path)

    # 平衡训练集
    train_balanced = balance_data(train)

    train_data = process_data(train_balanced)
    dev_data = process_data(dev)
    test_data = process_data(test)

    # 测试用
    test_data1 = load_dataset(config.test_path)


    return train_data, dev_data, test_data,test_data1

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

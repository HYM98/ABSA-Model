# coding: UTF-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'robert'
        self.aspect = '食物'
        self.train_path = dataset + '/new_data/ASAP/train_food.txt'  # 训练集
        self.dev_path = dataset + '/new_data/ASAP/dev_food.txt'  # 验证集
        self.test_path = dataset + '/new_data/ASAP/test_food.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + 'class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/'+ self.model_name + '_bilstm_食物'+'.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                          # epoch数
        self.batch_size = 16                                          # mini-batch大小
        self.pad_size = 200                                              # 每句话处理成的长度(短填长切)
        # self.learning_rate = 5e-5                                       # 学习率
        self.learning_rate = 1e-5
        self.bert_path = './roberta_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        print(self.tokenizer)
        self.hidden_size = 768
        self.dropout = 0.5
        self.rnn_hidden = 768
        self.num_layers = 2



class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bidirectional = True

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        output, (hidden_last,cn_last) = self.lstm(encoder_out)

        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]
        out = self.dropout(hidden_last_out)
        out = self.fc(out)
        return out




# coding: UTF-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.aspect = '食物'
        self.train_path = dataset + '/new_data/ASAP/train_food.txt'  # 训练集
        self.dev_path = dataset + '/new_data/ASAP/dev_food.txt'  # 验证集
        self.test_path = dataset + '/new_data/ASAP/test_food.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + 'class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/'+ self.model_name + '_wwm_bilstm_att_食物'+'.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                           # epoch数
        self.batch_size = 16                                          # mini-batch大小
        self.pad_size = 200                                              # 每句话处理成的长度(短填长切)
        # self.learning_rate = 5e-5                                       # 学习率
        self.learning_rate = 1e-5
        self.bert_path = './bert_wwm_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        print(self.tokenizer)
        self.hidden_size = 768
        self.dropout = 0.5
        self.final_dropout = 0.2
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.rnn = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                           bidirectional=True, batch_first=True, dropout=config.dropout)
        # 线性变换层
        self.attention = nn.Linear(config.rnn_hidden * 2, config.rnn_hidden * 2)
        # 注意力得分层
        self.attention_score = nn.Linear(config.rnn_hidden * 2, 1)
        self.query = nn.Linear(config.rnn_hidden * 2, config.rnn_hidden * 2)
        self.key = nn.Linear(config.rnn_hidden * 2, config.rnn_hidden * 2)
        self.value = nn.Linear(config.rnn_hidden * 2, config.rnn_hidden * 2)
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.sqrt_dk = config.rnn_hidden ** 0.5
        self.dropout = nn.Dropout(config.final_dropout)

    def attention_net(self, lstm_output):
        # Calculate Q, K, V
        Q = self.query(lstm_output)  # (batch_size, seq_len, hidden_size * 2)
        K = self.key(lstm_output)  # (batch_size, seq_len, hidden_size * 2)
        V = self.value(lstm_output)  # (batch_size, seq_len, hidden_size * 2)
        # print("进入attention_net")
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.sqrt_dk  # (batch_size, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=2)  # (batch_size, seq_len, seq_len)
        # Compute attention output
        attn_output = torch.bmm(attn_weights, V)  # (batch_size, seq_len, hidden_size * 2)
        # print("attn_output大小为:",attn_output.shape)
        # print("attn_weights大小为:",attn_weights.shape)
        return attn_output, attn_weights

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        lstm_out, _ = self.rnn(encoder_out)
        # print("进入forward")
        # Calculate attention weights and outputs
        attn_output, attn_weights_2d = self.attention_net(lstm_out)
        attn_output = self.dropout(attn_output)
        output = self.fc(attn_output)
        output = output.mean(dim=1)
        # print("output大小为:", output.shape)
        # print("attn_weights_2d大小为:", attn_weights_2d.shape)
        return output, attn_weights_2d



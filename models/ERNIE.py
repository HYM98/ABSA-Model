# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'ERNIE'
        # 方面类别
        self.aspect = '食物'
        self.train_path = dataset + '/new_data/ASAP/train_food.txt'  # 训练集
        self.dev_path = dataset + '/new_data/ASAP/dev_food.txt'  # 验证集
        self.test_path = dataset + '/new_data/ASAP/test_food.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + 'class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/'+ self.model_name + '_食物'+'.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 5                                          # epoch数 10
        self.batch_size = 16                                          # mini-batch大小 16或32
        self.pad_size = 200                                              # 每句话处理成的长度(短填长切)
        # self.learning_rate = 5e-5                                       # 学习率
        self.learning_rate = 1e-5
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        print(self.tokenizer)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

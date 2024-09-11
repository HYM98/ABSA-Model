# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, evaluate,test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', '-m', type=str, required=False, help='choose a model: Bert, ERNIE', default='ERNIE_BiLSTMATT')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'data'  # 数据集
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    print("正在使用模型:",model_name)

    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data, test_data1= build_dataset(config)
    print("成功创建训练集，验证集，测试集")
    #
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter,test_iter)

    # # predict
    # model.load_state_dict(torch.load('data/saved_dict/bert_bilstm_att_食物.ckpt'))
    # test_iter1 = build_iterator(test_data1, config)
    # data_predict = test(config,model,test_iter1)
    # data_evaluate = evaluate(config, model, test_iter1)
    # for label,data in enumerate(data_evaluate[2]):
    #     print(label+1,"预测值:",data)  #

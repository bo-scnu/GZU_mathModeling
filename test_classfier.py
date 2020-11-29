import pandas as pd
import tensorflow as tf
import numpy as np
import re
from transformers import AutoTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.rcParams['font.sans-serif'] = 'WenQuanYi Micro Hei'


def plotResult(result):
    label = []
    precision = []
    recall = []
    f1 = []

    for key, value in result.items():
        if key == 'accuracy':
            break
        label.append(key)
        precision.append(value['precision'])
        recall.append(value['recall'])
        f1.append(value['f1-score'])
    x =list(range(len(label)))
    total_width, n = 0.8, 3
    width = total_width / n
    plt.bar(x, precision, width=width, label='precision',color="w",edgecolor="k", hatch='///')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, recall, width=width, label='recall',tick_label = label, color="w",edgecolor="k", hatch='***')
    plt.xticks(rotation=20)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, f1, width=width, label='f1',color="w",edgecolor="k", hatch='xxx')
    plt.legend()
    plt.savefig("./images/result.png")
    plt.close()

def build_bert_inputs(tokenizer, sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=452, return_tensors='tf')
    return encoded_input

def main(y_true, sentences, tokenizer, bert_model_test):
    predicts = []

    # 计算每个类别precision, recall和f1值
    for sentence in tqdm(sentences):
        test_inputs = build_bert_inputs(tokenizer, sentence)
        # 构建预测函数
        outputs = bert_model_test(dict(test_inputs))
        logits = outputs.logits
        label_id = tf.argmax(tf.nn.softmax(logits)[0]).numpy()
        predicts.append(label_id)

    target_names = ['城乡建设', '环境保护', '交通运输', '教育文体', '劳动和社会保障', '商贸旅游', '卫生计生']
    result = classification_report(y_true, predicts, target_names=target_names, digits=4, output_dict=True)
    plotResult(result)
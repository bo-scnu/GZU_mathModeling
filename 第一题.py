import re
import pandas as pd
import numpy as np
import matplotlib
from itertools import accumulate
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# import tensorflow_hub as hub
from transformers import AutoTokenizer, glue_convert_examples_to_features
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments

from test_classfier import main

matplotlib.rcParams['font.sans-serif'] = 'WenQuanYi Micro Hei'
print(tf.__version__)

df = pd.read_excel('./data/附件2.xlsx')

topic_df = df['留言主题']
data_df = df['留言详情']
label_name = df['一级标签'].unique()

label2id = dict()
id2label = dict()

for index, label in enumerate(np.array(label_name)):
    label2id[label] = index
    id2label[index] = label

print("convert label to id: ", label2id)
print("id2label: ", id2label)

# 统计各标签分布情况
df.loc[:, '一级标签'].value_counts().plot.bar()
plt.xticks(rotation=20)
plt.grid(linestyle='-.', c='g')
plt.title("数据集标签分布频率统计")
plt.xlabel('标签')
plt.ylabel('频率')
plt.savefig("./images/数据集标签分布频率统计.png")
plt.close()

# 将标签转换为数字
df['transfromed'] = df['一级标签'].apply(lambda x : label2id[x])

# 将文本数据中转义字符去掉
df['texts'] = df['留言详情'].apply(lambda x : x.strip())
df['topic'] = df['留言主题'].apply(lambda x : x.strip())

# 将描述和主题拼接起来以增强数据
df['period'] = df[['texts', 'topic']].apply(lambda x: ''.join(x), axis=1)
# 去掉标点符号
punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
df['period'] = df['period'].apply(lambda x : re.sub(r"[%s]+" %punctuation, "",x))

# 统计句子的长度分布
num_sentence = len(np.array(df['period'].str.len()))
max_seq_len = np.array(df['period'].str.len()).max()
print("最大句子长度为：", max_seq_len)
print("句子总数为：", num_sentence)

dict_length = dict(Counter(np.array(df['period'].str.len())))

# 计算合适的最大句子长度
def plot_sentence_len(dict_length):
    """统计句子长度及长度出现的频数"""
    sent_length = []
    sent_freq = []
    for key, value in dict_length.items():
        sent_length.append(key)
        sent_freq.append(value)
    # 绘制句子长度及出现频数统计图
    plt.bar(sent_length, sent_freq)
    plt.title("句子长度及出现频数统计图")
    plt.xlabel("句子长度")
    plt.ylabel("句子长度出现的频数")
    plt.savefig("./images/句子长度及出现频数统计图.png")
    plt.close()
    # 绘制句子长度累积分布函数(CDF)
    sent_pentage_list = [(count/sum(sent_freq)) for count in accumulate(sent_freq)]
    # 绘制CDF
    plt.plot(sent_length, sent_pentage_list)
    # 寻找分位点为quantile的句子长度
    quantile = 0.94
    index = 0
    #print(list(sent_pentage_list))
    for length, per in zip(sent_length, sent_pentage_list):
        if round(per, 2) == quantile:
            index = length
            break
    print("最大句子长度:", index)
    #print("\n分位点为%s的句子长度:%d." % (quantile, index))
    # 绘制句子长度累积分布函数图
    plt.plot(sent_length, sent_pentage_list)
    plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
    plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
    plt.text(0, quantile, str(quantile))
    plt.text(index, 0, str(index))
    plt.title("句子长度累积分布函数图")
    plt.xlabel("句子长度")
    plt.ylabel("句子长度累积频率")
    plt.savefig("./images/句子长度累积分布函数图.png")
    plt.close()
    return index

max_sequence_length = plot_sentence_len(dict_length)

# 数据集构建
data_df = df['period']
label_df = df['transfromed']
sentences = list(data_df)
targets = label_df.values
assert len(targets) == len(sentences)
print(len(targets))

train_x, test_x, train_y, test_y = train_test_split(sentences, targets,test_size=0.25, random_state=42)

batch_size = 32
# 构造bert模型输入
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

train_encoded_input = tokenizer(train_x, padding=True, truncation=True, max_length=452)
test_encoded_input = tokenizer(test_x, padding=True, truncation=True, max_length=452)

# 封装数据
train_datasets = tf.data.Dataset.from_tensor_slices((dict(train_encoded_input), train_y)) # 封装 dataset数据集格式
test_datasets = tf.data.Dataset.from_tensor_slices((dict(test_encoded_input), test_y))

# 构建模型
# print("训练模型.............")
# model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=7)

# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss =  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# metrics=['accuracy']

# # checkpointer = ModelCheckpoint(filepath="/train_logs/weights.hdf5", verbose=1, save_best_only=True)
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics) # can also use any keras loss fn
history = model.fit(train_datasets.shuffle(1000).batch(16), epochs=3, 
                    validation_data=test_datasets.batch(16), callbacks=[tensorboard])

# # 可视化模型的损失和准确率变化情况
# # 绘制训练 & 验证的准确率值
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig("./images/训练和验证准确率.png")
# plt.close()

# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig("./images/训练和验证损失.png")
# plt.close()

# # 保存模型
# model.save_pretrained('./bert_pretraind/')

bert_model_test = TFBertForSequenceClassification.from_pretrained('./bert_pretraind/', return_dict=True)

# 绘制结果
main(y_true=test_y, sentences=test_x, tokenizer=tokenizer, bert_model_test=bert_model_test)
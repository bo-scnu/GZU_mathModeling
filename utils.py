from LAC import LAC
import re
def data_clean(data_df_):
    data_df_['留言详情'] = data_df_['留言详情'].apply(lambda x : x.strip())
    data_df_['留言主题'] = data_df_['留言主题'].apply(lambda x : x.strip())
    # 去除标点符号
    punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
    data_df_['留言详情'] = data_df_['留言详情'].apply(lambda x : re.sub(r"[%s]+" %punctuation, "",x))
    data_df_['留言主题'] = data_df_['留言主题'].apply(lambda x : re.sub(r"[%s]+" %punctuation, "",x))

    # 对留言主题和留言详情进行分词
    lac = LAC(mode='seg')
    data_df_split = data_df_
    del data_df_
    data_df_split['留言详情'] = data_df_split['留言详情'].apply(lambda x : lac.run(x))
    data_df_split['留言主题'] = data_df_split['留言主题'].apply(lambda x : lac.run(x))

    # 去掉停用词
    f = open('./data/stop_word_ZH.txt', 'r')
    stop_words = f.readlines()
    f.close()
    def remove_stop_word(words):
        return [w for w in words if w not in stop_words]

    data_df_split['留言详情'] = data_df_split['留言详情'].apply(lambda x : remove_stop_word(x))
    data_df_split['留言主题'] = data_df_split['留言主题'].apply(lambda x : remove_stop_word(x))
    # print(data_df_split.head())
    
    return data_df_split



if __name__ == "__main__":
    data_dict = {12:['这个','单词', '在', 'words', '中', '出现'],
                 10:['这个','单词', '在', 'words', '中', '出现']}
    RI = ReverseIndex(data_dict)
    all = RI.forwardIndex()
    print(all)
    word_dict = RI.reverseIndex(all)
    print(word_dict)

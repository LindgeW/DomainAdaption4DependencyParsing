import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import numpy as np
import jieba


def cal_sims():
    train = []
    fp = codecs.open('../data/tokenized_files/merge-doc.txt', 'r', encoding='utf8')  #文字檔案，輸入需要提取主題的文件
    stop_wds = ['，', '；', '。', '！', '、', '？', '“', '”', '：', '的', '是', '了', '吗', '我', '跟', '了', '有', '下', '能', '要', '呢',
                '吧', '都']
    for line in fp:
        # line = list(jieba.cut(line))
        line = list(line.strip().split())
        train.append([w for w in line if w not in stop_wds])

    dictionary = Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=4)
    # lda.save('test_lda.model')
    # lda = models.ldamodel.LdaModel.load('test_lda.model')

    for dm1 in ["BC", "PB", "PC", "ZX"]:
        for dm2 in ["BC", "PB", "PC", "ZX"]:
            with open('../data/tokenized_files/'+dm1+'-doc.txt', 'r', encoding='utf-8') as f1:
                s1 = f1.readline()
            with open('../data/tokenized_files/'+dm2+'-doc.txt', 'r', encoding='utf-8') as f2:
                s2 = f2.readline()

            # test_doc = list(jieba.cut(s1))  # 新文件進行分詞
            test_doc = s1.strip().split()
            doc_bow = dictionary.doc2bow(test_doc)  # 文件轉換成bow
            doc_lda = lda[doc_bow]  # 得到新文件的主題分佈

            # test_doc2 = list(jieba.cut(s2))  # 新文件進行分詞
            test_doc2 = s2.strip().split()
            doc_bow2 = dictionary.doc2bow(test_doc2)  # 文件轉換成bow
            doc_lda2 = lda[doc_bow2]  # 得到新文件的主題分佈
            list_doc1 = [i[1] for i in doc_lda]
            list_doc2 = [i[1] for i in doc_lda2]
            sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
            # try:
            #     sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
            # except ValueError:
            #     sim = 0
            print(dm1, dm2, sim)


def run():
    cal_sims()

run()
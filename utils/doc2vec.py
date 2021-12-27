import gensim


def train_test():
    fn = 'data/tokenized_files/merge-doc.txt'
    # file = open(fn, 'r', encoding='utf-8')
    # documents = gensim.models.doc2vec.TaggedLineDocument(file)
    documents = []
    stop_wds = ['，', '；', '。', '！', '、', '？', '“', '”', '：', '的', '是', '了', '吗', '我', '跟', '了', '有', '下', '能', '要', '呢', '吧', '都']
    with open(fn, 'r', encoding='utf-8') as fin:
        for i, txt in enumerate(fin):
            wd_lst = txt.strip().split()
            wd_lst = [w for w in wd_lst if w not in stop_wds]
            doc = gensim.models.doc2vec.TaggedDocument(wd_lst, tags=[i])
            documents.append(doc)

    model = gensim.models.Doc2Vec(documents, vector_size=100, window=5, min_count=1, negative=5, sample=1e-3, workers=4, dm_mean=1, dbow_words=0)
    model.train(documents, total_examples=model.corpus_count, epochs=20)
    # model.save(outp1)
    # model = gensim.models.Doc2Vec.load("D:\python_noweightpathway\TIA\docmodel")
    print(model.docvecs.most_similar(2))
    for i in range(4):
        for j in range(i+1, 4):
            print(i, j, '\t', model.docvecs.similarity(i, j))


train_test()
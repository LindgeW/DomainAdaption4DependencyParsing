#!/usr/bin/python
# coding=utf-8

import sys


def read_data(file):
    try:
        infile = open(file, mode="r", encoding="utf-8")
        sens = {}
        heads = []
        rels = []
        sen = []
        head = []
        rel = []
        index = 0
        for line in infile:
            if line != "\n" and line != "\r\n":
                line_list = line.strip().split("\t")
                sen.append(line_list[1])
                head.append(line_list[6])
                rel.append(line_list[7])
            else:
                sens[tuple(sen)] = index
                heads.append(head)
                rels.append(rel)
                sen = []
                head = []
                rel = []
                index += 1
        infile.close()
        return sens, heads, rels
    except IOError:
        print('IOError: please check the filename')


def evaluation(answer_heads, answer_rels, test_heads, test_rels):
    #answer_sen, answer_heads, answer_rels = read_data(answerfile)
    #test_sen, test_heads, test_rels = read_data(testfile)
    assert len(answer_heads) == len(test_heads)
    correct_arc, total_arc, correct_label, total_label, in_answer_file = 0, 0, 0, 0, 0
    for j in range(len(answer_heads)):
        pre_head = test_heads[j]
        gold_head = answer_heads[j]
        pre_rel = test_rels[j]
        gold_rel = answer_rels[j]
        assert len(pre_head) == len(gold_head)
        for i in range(len(gold_head)):
            if gold_head[i] == -1:
                continue
            total_arc += 1
            total_label += 1
            if gold_head[i] == pre_head[i]:
                correct_arc += 1
                if gold_rel[i] == pre_rel[i]:
                    correct_label += 1
    uas = correct_arc * 100.0 / total_arc
    las = correct_label * 100.0 / total_label
    return uas, las


if __name__ == "__main__":
    answer_name = sys.argv[1]  # gold conll
    test_name = sys.argv[2]  # sys conll
    correct_arc, toltal_arc, correct_label, total_label, uas, las = evaluation(answer_name, test_name)
    print("UAS = %d/%d = %.2f, LAS = %d/%d =%.2f" % (correct_arc, toltal_arc, uas, correct_label, total_label, las))

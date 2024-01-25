import multiprocessing
import os
from gensim.models import Word2Vec
import jieba

import base_tool
import torch

if __name__ == '__main__':
    base_tool.getCurrentExecutionInfo()


    # 创建输出路径
    out_dir = './word2vec'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sentence_list = [
        '他强调：要坚持规划先行，立足高起点、高标准、高质量，科学规划园区组团，提升公共服务水平，注重产城融合发展。',
        '要增强集聚功能，集聚产业、人才、技术、资本，加快构建从基础研究、技术研发、临床实验到药品生产的完整产业链条。'
        '完善支撑产业发展的研发孵化、成果转化、生产制造平台等功能配套，推动产学研用协同创新，做大做强生物医药产业集群。',
        '唐良智在调研中指出，我市生物医药产业具有良好基础，但与高质量发展的要求相比，在规模、结构、创新能力等方面还存在不足。',
        '推动生物医药产业高质量发展，努力培育新兴支柱产业，必须紧紧依靠创新创业创造，着力营造良好发展环境。',
        '要向改革开放要动力，纵深推进“放管服”改革，用好国际国内创新资源，大力引进科技领军人才、高水平创新团队。',
        '要坚持问题导向，聚焦企业面临的困难和问题，把握生物医药产业发展特点，精准谋划、不断完善产业支持政策，切实增强企业获得感。',
        '要精准服务企业，构建亲清新型政商关系，以高效优质服务助力企业发展。']

    # 使用jieba分词对sentence_list进行切分
    sentence_tokens = []
    for sentence in sentence_list:
        cuts = jieba.lcut(sentence)
        sentence_tokens.append(cuts)

    print(sentence_tokens)

    # 训练word2vec模型
    # Word2Vec<vocab=141, vector_size=256, alpha=0.025>
    model = Word2Vec(sentence_tokens, vector_size=64, min_count=1, window=5, sg=0, workers=multiprocessing.cpu_count())
    print(model)

    # 保存模型
    model.save(out_dir + os.sep + 'word2vec.model')

    # 加载模型
    model = Word2Vec.load(out_dir + os.sep + 'word2vec.model')
    print(model)

    # 模型的增量训练
    new_sentence_list = ['我喜欢吃苹果',
        '大话西游手游很好玩',
        '人工智能包含机器视觉和自然语言处理']
    new_sentence_tokens = []
    for new_sentence in new_sentence_list:
        new_cuts = jieba.lcut(new_sentence)
        new_sentence_tokens.append(new_cuts)
    print(new_sentence_tokens)

    model.build_vocab(new_sentence_tokens, update= True)

    # Word2Vec<vocab=155, vector_size=256, alpha=0.025>
    model.train(new_sentence_tokens, total_examples=model.corpus_count,epochs=10)
    print(model)

    # 获取词向量
    w_vec = model.wv['苹果']
    print(w_vec)
    quit()
    pass
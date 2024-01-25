import os

import torch

import base_tool
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    base_tool.getCurrentExecutionInfo()

    # 下载好的预训练模型，
    # config.json, pytorch_model.bin(tf_model.h5), vocab.txt
    # tokenizer.json, tokenizer_config.json
    path = 'D:/git_hub/bert-base-uncased'
    # 创建输出路径
    out_dir = './bert2vec'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 加载分词器和预训练模型
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertModel.from_pretrained(path)
    # print(tokenizer)
    # print(model)

    text = "Hello, my dog is cute"

    # 使用分词器将文本转换为token ids
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    print(inputs)

    # 通过BERT模型获取每个token的隐藏状态（可以视为词向量）
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state是形状为(batch_size, sequence_length, hidden_size)的tensor
    # 对于单个句子，我们只需要最后一个维度，即词向量
    word_vectors = outputs.last_hidden_state[0]
    print(word_vectors)

    # 打印词向量
    for i in range(len(inputs['input_ids'][0])):
        token_id = inputs['input_ids'][0].tolist()[i]
        # print(token_id)
        # quit()
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token != '[PAD]':
            vector = word_vectors[i]
            print(f'Token: {token}, Vector: {vector.tolist()}')
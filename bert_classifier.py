import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import Dataset

# 定义DataSet类
class MyDataSet(Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors='pt')
                      for text in df['text']]
        # print(self.labels)
        # print(self.texts)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

# 定义Bert分类器
# 在Bert预训练的基础上，添加一层分类器
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

# 定义训练模型
# model也就是定义的Bert分类器
def train_func(model, train_data, val_data, learning_rate, epochs):
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 通过自定义的MyDataSet加载训练数据和验证数据
    train, val = MyDataSet(train_data), MyDataSet(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # 定义损失函数（交叉熵损失函数）和优化器（Adam）
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.long().to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.long().to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
                  | Train Loss: {total_loss_train / len(train_data): .3f} 
                  | Train Accuracy: {total_acc_train / len(train_data): .3f} 
                  | Val Loss: {total_loss_val / len(val_data): .3f} 
                  | Val Accuracy: {total_acc_val / len(val_data): .3f}''')


# 定义测试函数
def evaluate(model, test_data):

    test = MyDataSet(test_data)
    test_dataloader = torch.utils.data.DataLoader(test)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
              test_label = test_label.long().to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


if __name__ == '__main__':
    model_path = 'D:\\git_hub\\aliendao\\dataroot\\models\\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 'input_ids': [101, 100, 9339, 9114, 100, 8228, 12734, 102], 它是每个 token 的 id 表示
    # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0] 它是一个 binary mask，用于标识 token 属于哪个 sequence
    # attention_mask，它是一个binary mask，用于标识token是真实word还是只是由填充得到

    labels = {'business': 0,
              'entertainment': 1,
              'sport': 2,
              'tech': 3,
              'politics': 4
              }

    # 对测试数据进行随机切分
    bbc_text_df  = pd.read_csv('./data/bbc-text.csv')
    bbc_text_df = bbc_text_df.iloc[:100, :]
    df = pd.DataFrame(bbc_text_df )
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])
    # print(df_train)
    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    model = BertClassifier()
    EPOCHS = 3
    LR = 1e-6
    train_func(model, df_train, df_val, LR, EPOCHS)
    torch.save(model.state_dict(), 'bert_classifier_weights.pth')

    # 加载模型
    model.load_state_dict(torch.load('bert_classifier_weights.pth'))
    evaluate(model, df_test)


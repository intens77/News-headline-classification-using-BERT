"""
Класс для преобразования исходных данных
"""
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import constants

bert_tokenizer = BertTokenizer.from_pretrained(constants.MODEL_NAME)


class TaskDataset(Dataset):
    def __init__(self, titles, labels):
        self.titles = [bert_tokenizer(title, padding='max_length', max_length=512, truncation=True,
                                      return_tensors='pt') for title in titles]
        self.labels = labels
        self.length = len(self.titles)

    def __getitem__(self, index):
        return self.titles[index], self.labels[index]

    def __len__(self):
        return self.length


def get_loader(titles, labels, batch_size=32, shuffle=True):
    dataset = TaskDataset(titles, labels)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

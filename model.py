"""
Класс модели
"""
from torch import nn
from transformers import BertModel

from constants import MODEL_NAME


class SentenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(MODEL_NAME)
        self.classification_model = nn.Sequential(
            nn.Linear(312, 90),
            nn.ReLU(),
            nn.Linear(90, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, mask):
        _, bert_output = self.bert_model(input_ids=input_ids, attention_mask=mask, 
        return_dict=False)
        classification_model_output = self.classification_model(bert_output)
        return classification_model_output

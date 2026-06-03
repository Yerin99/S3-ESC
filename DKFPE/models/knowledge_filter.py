# coding=utf-8
"""
DeBERTa-v3-large Knowledge Filter for DKFPE

Binary classifier: Relevant (RR) vs Irrelevant (IRR)
Input format: [CLS] context [SEP] commonsense_text [SEP]
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class KnowledgeFilter(nn.Module):
    def __init__(self, model_path='microsoft/deberta-v3-large', num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        outputs = self.forward(input_ids, attention_mask)
        probs = torch.softmax(outputs['logits'], dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs

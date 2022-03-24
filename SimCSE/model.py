import torch
from transformers import AutoModel
import torch.nn.functional as F


class TextBackbone(torch.nn.Module):
    def __init__(self,
                 pretrained='./pretrained/chinese-roberta-wwm-ext',
                 output_dim=128) -> None:
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained).cuda()
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.extractor(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids).pooler_output

        x = self.drop(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def predict(self, x):
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)

        x = self.extractor(**x).pooler_output
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)

        return x

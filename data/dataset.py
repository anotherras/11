from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch


def load_data(filepaths, tokenizer=lambda s: s.strip().replace(' ', '')):
    raw_in_iter = iter(open(filepaths[0], encoding="utf8"))
    raw_out_iter = iter(open(filepaths[1], encoding="utf8"))
    return list(zip(map(tokenizer, raw_in_iter), map(tokenizer, raw_out_iter)))


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path='bert-base-chinese',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
            cache_dir=r'C:\Users\mzm\pythonProject\path_test\NLP_test\project\cache', local_files_only=True
)
        return tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def collate_fn(batch, tokenizer):
        in_batch, out_batch = list(zip(*batch))

        in_tensor = tokenizer.batch_encode_plus(in_batch, padding=True, return_attention_mask=False,
                                                return_tensors='pt', return_token_type_ids=False)
        out_tensor = tokenizer.batch_encode_plus(out_batch, padding=True, return_attention_mask=False,
                                                 return_tensors='pt', return_token_type_ids=False)

        src = in_tensor['input_ids']
        tgt = out_tensor['input_ids'][:, :-1]

        src_mask = (src != 0).unsqueeze(-2)
        tgt_mask = (tgt != 0).unsqueeze(-2)

        def subsequent_mask(size):
            "Mask out subsequent positions."
            attn_shape = (1, size, size)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
                torch.uint8
            )
            return subsequent_mask == 0

        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        in_tensor['src_mask'] = src_mask
        out_tensor['tgt_mask'] = tgt_mask
        return in_tensor, out_tensor

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=lambda batch: self.collate_fn(batch, self.tokenizer))


if __name__ == '__main__':
    torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=2000, profile=None, sci_mode=None)
    filepath = [r'C:\Users\mzm\pythonProject\path_test\NLP_test\raw_data\couplet\train\in.txt',
                r'C:\Users\mzm\pythonProject\path_test\NLP_test\raw_data\couplet\train\out.txt']
    data = MyDataset(filepath)
    loader = data.get_loader(5)
    for i in loader:
        print(type(i))
        print(len(i))
        print(i)
        break

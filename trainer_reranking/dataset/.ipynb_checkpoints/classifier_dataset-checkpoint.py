import random
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class ClassifierDataSet(Dataset):
    # top_k = 256
    # is_predict = True
    # max_desc_num = 80
    # max_desc_len = 128
    # max_tot_len = 256
    top_k = 128
    is_predict = True
    max_desc_num = 240
    max_desc_len = 384
    max_tot_len = 512

    def __init__(self, data_path, sep_token=" ", mask_rate=0, bert_base_path="", token_lens = (128, 240, 384, 512)):
        self.top_k = token_lens[0]
        self.max_desc_num = token_lens[1]
        self.max_desc_len = token_lens[2]
        self.max_tot_len = token_lens[3]
        
        tokenizer = BertTokenizer.from_pretrained(bert_base_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data_list = []
        self.result = []
        if self.is_predict:
            self.result = data
        for item in data:
            desc = item.get("desc")
            top_k = item.get("top_k")
            ground_truth = item.get("labels")
            if len(desc.split(" ")) > self.max_desc_num:
                desc = " ".join(desc.split(" ")[:self.max_desc_num])
            vul_data = tokenizer("[CLS]" + desc + "[SEP]",
                                 add_special_tokens=False,
                                 truncation=True,
                                 max_length=self.max_desc_len,
                                 return_tensors="pt")
            vul_token_size = vul_data.get("input_ids").shape[1]
            for package_info in top_k[:self.top_k]:
                lib_name = package_info.get("lib_name")
                lib_info = [lib_name, package_info.get("website_description")]
                if lib_info[1] and mask_rate:
                    rand = random.random()
                    if rand < mask_rate:
                        mask_index = random.randint(0, 1)
                        lib_info[mask_index] = ""
                text = sep_token.join(lib_info) + "[SEP]"
                tmp = tokenizer(text,
                                add_special_tokens=False,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_tot_len - vul_token_size,
                                return_tensors="pt")
                batch_data = {key: torch.cat([vul_data.get(key), value], dim=1).view(-1) for key, value in tmp.items()}
                batch_data["labels"] = torch.tensor(1 if lib_name in ground_truth else 0)
                batch_data["index"] = len(self.data_list)
                self.data_list.append(batch_data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def save_score(self, predict_score: dict):
        for i in range(len(predict_score["index"])):
            index = predict_score["index"][i].item()
            score = predict_score["score"][i][1].item()
            vul_index = index // self.top_k
            lib_index = index % self.top_k
            vul_info = self.result[vul_index]
            lib_info = vul_info.get("top_k")[lib_index]
            lib_info["score"] = score

    def save_result(self, path, data=None):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(path, "w", encoding="utf-8") as f:
            if data:
                json.dump(data, f, indent=4)
            else:
                json.dump(self.result, f, indent=4)
        self._clear_score()

    def save_simplified_result(self, path):
        res = [dict()] * len(self.result)
        for index, item in enumerate(self.result):
            cve_id = item.get("cve_id")
            top_k = [x.get("score") for x in item.get("top_k")[:self.top_k]]
            res[index] = {"cve_id": cve_id, "top_k": top_k}
        self.save_result(path, res)

    def _clear_score(self):
        for item in self.result:
            top_k = item.get("top_k")[:self.top_k]
            for lib_info in top_k:
                lib_info.pop("score", 0)


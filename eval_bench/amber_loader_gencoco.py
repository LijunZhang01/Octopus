import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random

TOTAL_LEN = 15220
GEN_LEN = 10001

class AMBERDataSet(Dataset):
    def __init__(self, json_path, data_path, trans, model, num_gen=0, num_dis=5000):
        self.json_path = json_path
        self.data_path = data_path
        self.trans = trans
        self.model = model
        self.num_gen = num_gen
        self.num_dis = num_dis  
        
        # gen_idx = random.sample(range(0, GEN_LEN), self.num_gen)
        gen_idx = random.sample(range(0, 10001), self.num_gen)
        # dis_idx = random.sample(range(GEN_LEN + 1, TOTAL_LEN), self.num_dis)
        dis_idx = random.sample(range(1, 20000*6+1), self.num_dis)
        

        image_list, query_list, id_list ,idt_list,label_list= [], [], [], [], []
        yw_list,yl_list=[],[]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)


        
        
        for line in data:
            if line['id'] in gen_idx or line['id'] in dis_idx:
                image_list.append(line['image'])
                query_list.append(line['query'])
                id_list.append(line['id'])
                idt_list.append(line['id_t'])
                # label_list.append(line['label'])
                # if 'y_w' in line and 'y_i' in line:
                # yw_list.append(torch.tensor(line['y_w']))
                # yl_list.append(torch.tensor(line['y_l']))
            

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(id_list)

        self.image_list = image_list
        self.yw_list = yw_list
        self.yl_list = yl_list
        self.query_list = query_list
        self.id_list = id_list
        self.idt_list = idt_list
        self.label_list = label_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        
        if self.model == 'llava':
            raw_image = Image.open(image_path)
            image = self.trans.preprocess(raw_image, return_tensor='pt')['pixel_values'][0]
            query = self.query_list[index]
            id = self.id_list[index]
            idt = self.idt_list[index]
            # label=self.label_list[index]
            # yw=self.yw_list[index]
            # yl = self.yl_list[index]

            return {"image": image, "query": query, "id": id, "idt": idt, "image_path": image_path}
            # return {"image": image, "query": query, "id": id, "idt": idt, "image_path": image_path,"label":label}
            # return {"image": image, "query": query, "id": id, "image_path": image_path,'y_w':yw,'y_l':yl}

        elif self.model == 'qwen-vl':
            raw_image = Image.open(image_path).convert("RGB")
            image = self.trans(raw_image)
            query = self.query_list[index]
            id = self.id_list[index]
            return {"image": image, "query": query, "id": id, "image_path": image_path}
        
        elif self.model == 'instructblip':
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.trans['eval'](raw_image)
            query = self.query_list[index]
            id = self.id_list[index]
            idt = self.idt_list[index]
            return {"image": image_tensor, "query": query, "id": id, "idt": idt, "image_path": image_path}
            # return {"image": image_tensor, "query": query, "id": id, "image_path": image_path}


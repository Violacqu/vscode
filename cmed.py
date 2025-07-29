#cmed.py
import csv
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

@LOAD_DATASET.register_module()  #注册为可用于自动加载的模块类
class CMEDDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope': # 根据环境变量DATASET_SOURCE判断是否ModelScope加载
            from modelscope import MsDataset
            dataset = MsDataset.load(
                path,
                subset_name=name,
            )
            modified_dataset = DatasetDict() #转换为标准的HF格式
            for split in dataset.keys():
                raw_data = []
                for data in dataset[split]:
                    raw_data.append({
                        'question': data['Question'],  # 修改字段对应名
                        'A': data['A'],
                        'B': data['B'],
                        'C': data['C'],
                        'D': data['D'],
                        'answer': data['Answer']  
                    })
                modified_dataset[split] = Dataset.from_list(raw_data)
            dataset = modified_dataset
        else:
            # dataset = DatasetDict()
            # for split in ['dev', 'test']:
            #     raw_data = []
            #     filename = osp.join(path, split, f'{name}.json')  # 支持 json 格式
            #     with open(filename, encoding='utf-8') as f:
            #         for line in f:
            #             data = json.loads(line.strip())
            #             raw_data.append({
            #                 'question': data['question'],
            #                 'answer': data['answer'],
            #             })
            # dataset[split] = Dataset.from_list(raw_data)
            dataset = DatasetDict()
            for split in ['dev','test']: #dev few-shot示例数据 test评测集
                raw_data = []
                filename = osp.join(path, split, f'{name}.csv') #从<path>/<split>/<name>.csv 加载文件
                with open(filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    _ = next(reader)  # skip the header
                    for row in reader:
                        assert len(row) == 7 #数据集列数 分别对应
                        raw_data.append({
                            'question': row[1],
                            'A': row[2],
                            'B': row[3],
                            'C': row[4],
                            'D': row[5],
                            'answer': row[6],
                        })
                dataset[split] = Dataset.from_list(raw_data)
        return dataset
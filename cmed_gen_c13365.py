from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import CMEDDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

cmed_subject_mapping = {
    'gongyi': '工艺基础选择题'
}

# 获取所有子任务列表（目前只含 'gongyi'）
# 加载后的数据集通常以字典形式组织样本，分别指定样本中用于组成 prompt 的输入字段，和作为答案的输出字段
cmed_all_sets = list(cmed_subject_mapping.keys())

# 定义用于注册的 CMED 测评任务
cmed_datasets = []
for _name in cmed_all_sets:
    _ch_name = cmed_subject_mapping[_name]

    # 推理：使用 Prompt + 固定样本（Fixed K）检索
    cmed_infer_cfg = dict(
        # 构造 In Context Example (ice) 的模板
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=f'以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n'
                               '题目：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}'
                    ),
                    dict(role='BOT', prompt='答案是: {answer}'),
                ]
            ),
            ice_token='</E>',
        ),
        # 上下文样本配置，例如 `ZeroRetriever`，即不使用上下文样本
        # 此处为FixKRetriever 固定选前5个样本（id 0～4）作为 few-shot 示例
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), 
        # 推理方式配置
        #   - PPLInferencer 使用 PPL（困惑度）获取答案
        #   - GenInferencer 使用模型的生成结果获取答案
        inferencer=dict(type=GenInferencer), #使用生成式模型推理，模型会输出完整的字符串答案
    )

    # 评估配置 评估配置，使用准确率评估 
    cmed_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        ## 预测结果的后处理：获取第一个大写字母
        pred_postprocessor=dict(type=first_capital_postprocess) 
    )


    # 数据集配置，以上各个变量均为此配置的参数
    # 为一个列表，用于指定一个数据集各个评测子集的配置。
    cmed_datasets.append(
        dict(
            type=CMEDDataset,
            path='opencompass/cmed',
            name=_name,
            abbr=f'cmed-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'
            ),
            infer_cfg=cmed_infer_cfg,
            eval_cfg=cmed_eval_cfg,
        )
    )

# 清理临时变量
del _name, _ch_name

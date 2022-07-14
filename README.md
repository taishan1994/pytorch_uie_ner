# pytorch_uie_ner
基于pytorch的百度UIE命名实体识别，源代码来源：[here](https://github.com/heiheiyoyo/uie_pytorch)。

百度UIE通用信息抽取的样例一般都是使用doccano标注数据，这里介绍如何使用通用数据集，利用UIE进行微调。

# 依赖

```
torch>=1.7.0
transformers==4.20.0
colorlog
colorama
```

# 步骤

- 1、数据放在data下面，比如已经放置的cner（简历数据集），raw_data下面是原始的数据，新建一个process.py，将数据处理成类似mid_data下面的数据，```python process.py```即:

	```python
	{"id": 0, "text": "吴重阳，中国国籍，大学本科，教授级高工，享受国务院特殊津贴，历任邮电部侯马电缆厂仪表试制组长、光缆分厂副厂长、研究所副所长，获得过山西省科技先进工作者、邮电部成绩优异高级工程师等多种荣誉称号。", "relations": [], "entities": [{"id": 0, "start_offset": 0, "end_offset": 3, "label": "人名"}, {"id": 1, "start_offset": 4, "end_offset": 8, "label": "国籍"}, {"id": 2, "start_offset": 9, "end_offset": 13, "label": "学历"}, {"id": 3, "start_offset": 14, "end_offset": 19, "label": "职称"}, {"id": 4, "start_offset": 32, "end_offset": 40, "label": "机构"}, {"id": 5, "start_offset": 40, "end_offset": 46, "label": "职称"}, {"id": 6, "start_offset": 47, "end_offset": 54, "label": "职称"}, {"id": 7, "start_offset": 55, "end_offset": 61, "label": "职称"}]}
	```

- 2、将mid_data下面的数据使用doccano.py转换成final_data下的数据，具体指令是：

	- 训练集：
		```python
		python doccano.py \
		    --doccano_file ./data/cner/mid_data/train.json \
		    --task_type "ext" \  # ext表示抽取任务
		    --splits 1.0 0.0 0.0 \  # 训练、验证、测试数据的比例。训练，不对数据进行切分，因此将第一位设置为1.0
		    --save_dir ./data/cner/final_data/ \
		    --negative_ratio 3  # 生成负样本的比率
		```

	- 验证集：
		```python
		python doccano.py \
		    --doccano_file ./data/cner/mid_data/dev.json \
		    --task_type "ext" \  # ext表示抽取任务
		    --splits 0.0 1.0 0.0 \  # 训练、验证、测试数据的比例。验证，因此将第二位设置为1.0
		    --save_dir ./data/cner/final_data/ \
		    --negative_ratio 0  # 生成负样本的比率
		```

	- 测试集：
		```python
		python doccano.py \
		    --doccano_file ./data/cner/mid_data/test.json \
		    --task_type "ext" \  # ext表示抽取任务
		    --splits 0.0 0.0 1.0 \  # 训练、验证、测试数据的比例。测试，因此将第三位设置为1.0
		    --save_dir ./data/cner/final_data/ \
		    --negative_ratio 0  # 生成负样本的比率
		```

		最终会在final_data下生成train.txt、dev.txt和test.txt。

- 3、将paddle版本的模型转换为pytorch版的模型：

	```python
	python convert.py --input_model=uie-base --output_model=uie_base_pytorch --no_validate_output
	```

	其中input_model可选的模型可参考convert.py里面。output_model是我们要保存的模型路径，下面会用到。之后我们可以测试下转换的效果：

	```python
	from uie_predictor import UIEPredictor
	from pprint import pprint
	
	schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
	ie = UIEPredictor('./uie_base_pytorch', schema=schema)
	pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
	```

- 4、开始微调：

	```python
	python finetune.py \
	    --train_path "./data/cner/final_data/train.txt" \
	    --dev_path "./data/cner/final_data/dev.txt" \
	    --save_dir "./checkpoint" \
	    --learning_rate 1e-5 \
	    --batch_size 16 \
	    --max_seq_len 512 \
	    --num_epochs 1 \
	    --model "uie_base_pytorch" \
	    --seed 1000 \
	    --logging_steps 10 \ 
	    --valid_steps 100 \  # 每100步进行验证
	    --device "gpu" \
	    --max_model_num 2  # 最多保存2个模型
	```

	训练完成后，会在同目录下生成checkpoint/model_best/。

- 5、进行验证：

	```python
	python evaluate.py \
	    --model_path "./checkpoint/model_best" \
	    --test_path "./data/cner/final_data/dev.txt" \
	    --batch_size 16 \
	    --max_seq_len 512
	```

- 6、使用训练好的模型进行预测：

	```python
	from pprint import pprint
	from uie_predictor import UIEPredictor
	
	en2ch = {
	  'PRO':'专业', 
	  'ORG':'机构', 
	  'CONT':'国籍', 
	  'RACE':'民族', 
	  'NAME':'人名', 
	  'EDU':'学历', 
	  'LOC':'籍贯', 
	  'TITLE':'职称',
	}
	schema = en2ch.values()
	# 设定抽取目标和定制化模型权重路径
	my_ie = UIEPredictor('./checkpoint/model_best', schema=schema)
	pprint(my_ie("虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"))
	```

# 补充

- 标签名最好是使用中文。
- 可使用不同大小的模型进行训练和推理，以达到精度和速度的平衡。

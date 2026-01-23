# Off-policy Knowledge Distillation with Watermark-based Teacher

这个脚本实现了基于watermark的off-policy知识蒸馏训练。

## 核心思想

- **Student分布**: 正常模型输出的logits转换的概率分布
- **Teacher分布**: Student的logits经过watermark修改后转换的概率分布
- **损失函数**: Student和Teacher分布之间的KL散度

## 使用方法

### 基本用法

```bash
uv run train_kd_watermark.py \
    --model_name gpt2 \
    --train_file ./data/train.jsonl \
    --output_dir ./outputs/kd_watermark \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

### 完整参数示例

```bash
uv run train_kd_watermark.py \
    --model_name facebook/opt-125m \
    --train_file ./Unigram-Watermark/data/LFQA/inputs.jsonl \
    --val_file ./Unigram-Watermark/data/LFQA/val.jsonl \
    --output_dir ./outputs/kd_watermark \
    --max_length 512 \
    --watermark_fraction 0.5 \
    --watermark_strength 2.0 \
    --watermark_key 0 \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --save_steps 500 \
    --logging_steps 10 \
    --eval_steps 500 \
    --temperature 1.0 \
    --alpha 1.0 \
    --fp16
```

### 多GPU训练

使用环境变量指定GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run train_kd_watermark.py \
    --model_name gpt2 \
    --train_file ./data/train.jsonl \
    --output_dir ./outputs/kd_watermark \
    --batch_size 8 \
    --fp16
```

## 参数说明

### 模型参数
- `--model_name`: 模型名称或路径（如 `gpt2`, `facebook/opt-125m`）
- `--output_dir`: 输出目录，用于保存checkpoints

### 数据参数
- `--train_file`: 训练数据文件（支持JSONL或纯文本）
- `--val_file`: 验证数据文件（可选）
- `--max_length`: 最大序列长度（默认512）

### Watermark参数
- `--watermark_fraction`: Green list的比例（默认0.5）
- `--watermark_strength`: Watermark强度（默认2.0）
- `--watermark_key`: Watermark的随机种子（默认0）

### 训练参数
- `--num_epochs`: 训练轮数（默认3）
- `--batch_size`: 每设备batch size（默认4）
- `--gradient_accumulation_steps`: 梯度累积步数（默认1）
- `--learning_rate`: 学习率（默认5e-5）
- `--warmup_steps`: Warmup步数（默认100）
- `--save_steps`: 保存checkpoint的步数间隔（默认500）
- `--logging_steps`: 打印日志的步数间隔（默认10）
- `--eval_steps`: 评估的步数间隔（默认500）

### KD参数
- `--temperature`: 知识蒸馏的温度（默认1.0）
- `--alpha`: KD损失的权重（默认1.0）

### 硬件参数
- `--fp16`: 使用混合精度训练（float16）
- `--bf16`: 使用bfloat16精度
- `--dataloader_num_workers`: DataLoader的worker数量（默认0）

## 数据格式

### JSONL格式

支持以下字段：
- `text`: 完整文本
- `content`: 内容字段
- `prompt` + `completion`: 提示和完成
- `prefix` + `gold_completion`: 前缀和完成（LFQA格式）

示例：
```json
{"text": "This is a training example."}
{"prefix": "Question: ", "gold_completion": "This is the answer."}
```

### 纯文本格式

每行一个文本样本。

## 工作原理

1. **Student Logits**: 模型正常前向传播得到logits
2. **Watermark应用**: 将watermark（strength * green_list_mask）添加到student logits得到teacher logits
3. **概率分布**: 
   - Student: `log_softmax(student_logits / temperature)`
   - Teacher: `softmax(teacher_logits / temperature)`
4. **KL散度**: 计算 `KL(student || teacher)`
5. **反向传播**: 更新student模型参数

## 注意事项

- 脚本会自动处理多GPU分配（使用`device_map='auto'`）
- 确保有足够的GPU内存，大模型可能需要调整batch size和gradient accumulation
- Watermark参数会影响teacher分布，建议从默认值开始调优

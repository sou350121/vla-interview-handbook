# OpenVLA 微调实战 (OpenVLA Fine-tuning)

本章节提供 OpenVLA 模型微调的实战代码示例，重点展示如何使用 LoRA (Low-Rank Adaptation) 进行高效训练。

> **面试考点**:
> 1. 如何加载 7B+ 的 VLA 模型？(Quantization)
> 2. 如何构建 Action Label？(Normalization)
> 3. LoRA 的配置参数有哪些？

## 1. 环境准备与模型加载
使用 `bitsandbytes` 进行 4-bit 量化加载，节省显存。

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 2. 加载 OpenVLA 模型 (基于 Llama 2)
model_id = "openvla/openvla-7b"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 3. 准备 LoRA 训练
# 冻结原模型权重，只训练 Adapter
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,               # LoRA rank
    lora_alpha=32,      # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 作用于 Attention 层
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: ~10M || all params: ~7B || trainable%: 0.14%
```

## 2. 数据处理与 Action 归一化
VLA 模型通常预测归一化后的 Action。

```python
import numpy as np

def normalize_action(action, min_action, max_action):
    """将动作归一化到 [-1, 1]"""
    return 2 * (action - min_action) / (max_action - min_action) - 1

def format_instruction(text, action):
    """构建训练 Prompt"""
    # OpenVLA 期望的格式: "In: <instruction>\nOut: <action_tokens>"
    # 注意: 实际训练中，Action 通常会被 Tokenizer 编码为特定的 Token ID
    return f"In: {text}\nOut: {action}"

# 示例数据加载 (假设使用 PyTorch Dataset)
class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, processor):
        self.data = load_data(data_path) # 自定义加载函数
        self.processor = processor
        
        # 统计动作空间的 min/max 用于归一化
        self.action_min = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        self.action_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        instruction = item['language']
        raw_action = item['action'] # [x, y, z, roll, pitch, yaw, gripper]
        
        # 1. 归一化动作
        norm_action = normalize_action(raw_action, self.action_min, self.action_max)
        
        # 2. 构建 Prompt (OpenVLA 特有的 Action Token 处理通常在 Processor 内部或单独处理)
        # 这里简化展示，实际需参考 OpenVLA 官方仓库的 dataset 实现
        prompt = f"In: {instruction}\nOut:"
        
        # 3. 处理输入
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length"
        )
        
        # 添加 Labels (用于计算 Loss)
        # 实际代码中需要将 norm_action 转换为对应的 Token IDs
        return inputs
```

## 3. 训练循环 (Training Loop)
使用 HuggingFace Trainer 进行训练。

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./openvla-lora-checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_steps=1000,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset, # 实例化上面的 RobotDataset
    data_collator=default_data_collator,
)

trainer.train()
```

## 4. 推理与反归一化 (Inference)

```python
def unnormalize_action(norm_action, min_action, max_action):
    """将 [-1, 1] 还原为实际物理量"""
    return (norm_action + 1) / 2 * (max_action - min_action) + min_action

# 推理
inputs = processor(text="In: Pick up the coke can\nOut:", images=image, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=7) # 假设预测 7 个动作维度

# 解码
# OpenVLA 的 Action Head 输出通常是去离散化的数值，或者特定的 Action Tokens
# 这里假设模型输出已经是归一化的数值 (OpenVLA 架构特点)
predicted_action_norm = decode_action(generated_ids) 
predicted_action = unnormalize_action(predicted_action_norm, action_min, action_max)

# 发送给机器人执行
robot.move(predicted_action)
```

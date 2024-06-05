import json
import random
from tqdm import tqdm

def extract_keywords(text):
    # 替換為實際的關鍵詞提取邏輯
    return text.split()  # 示例：將文本按空格分割為單詞列表

# 假設這些是你已經準備好的訓練文本和標籤列表
train_texts = []  
train_labels = []  

with open("trainword.json", encoding='utf-8') as f:  # 指定编码为 utf-8
    data = json.load(f)

for i in tqdm(data, desc="Processing training data"):
    for j in i:
        x = random.random()
        if x < 0.0035:
            strs = extract_keywords(j[0])
            str = ','.join(strs)
            train_texts.append(str)
            train_labels.append(int(j[1]))

# 將 train_texts 和 train_labels 存儲為 JSON 文件
output_data = {
    "train_texts": train_texts,
    "train_labels": train_labels
}

with open("output_train_data.json", "w", encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

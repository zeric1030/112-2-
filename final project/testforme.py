import os
import json
import googleapiclient.discovery
import jieba
import jieba.analyse
import time
import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import random
import re

# 设置环境变量以防止 OpenMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

# 设置日志级别
logging.basicConfig(level=logging.INFO)


# 初始化 YouTube API 客户端
def get_youtube_client(api_key):
    return googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)


# 抓取顶层留言的回复
def get_comment_replies(parent_id, youtube):
    replies = []
    next_page_token = None

    while True:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            reply = item['snippet']['textDisplay']
            replies.append(reply)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return replies


# 抓取视频的所有留言及其回复
def get_video_comments(video_id, api_key):
    youtube = get_youtube_client(api_key)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(top_comment)

            # 抓取这个顶层留言的所有回复
            total_reply_count = item['snippet']['totalReplyCount']
            if total_replyCount > 0:
                parent_id = item['id']
                replies = get_comment_replies(parent_id, youtube)
                comments.extend(replies)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments


# 使用 jieba 进行留言分词
def segment_comments(comments):
    segmented_comments = [' '.join(jieba.lcut(comment)) for comment in comments]
    return segmented_comments


# 保留中文字符
def keep_chinese_chars(text):
    pattern = re.compile(r'[^\u4e00-\u9fff]')
    chinese_text = re.sub(pattern, '', text)
    return chinese_text


# 提取关键词
def extract_keywords(text):
    text = keep_chinese_chars(text)
    return jieba.analyse.extract_tags(text, topK=10)


# 训练模型
def train_model(train_texts, train_labels, model_save_path):
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 分词和编码
    logging.info("Tokenizing the data")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

    # 创建数据集
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # 使用 Trainer 训练模型
    logging.info("Starting training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # 保存模型和分词器
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return model


# 加载模型
def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


# 预测情绪
def predict_emotions(model, tokenizer, comments):
    # 分词处理留言
    inputs = tokenizer(comments, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # 进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

    return predictions


# 主程序
if __name__ == "__main__":
    logging.info("Starting the script")
    tStart = time.time()

    api_key = "AIzaSyAYu6KqHx8E96iIM96WD5saOdF2RbfoQmE"  # 替换为你的 YouTube Data API v3 密钥
    video_url = input("请输入视频 URL: ")
    video_id = video_url.split("v=")[1].split("&")[0]  # 确保只取到 video_id

    # 抓取留言
    logging.info("Fetching comments")
    comments = get_video_comments(video_id, api_key)

    # 断句处理
    logging.info("Segmenting comments")
    old_segmented_comments = segment_comments(comments)
    segmented_comments = []
    for c in old_segmented_comments:
        keywords = extract_keywords(c)
        keyword_str = ','.join(keywords)
        segmented_comments.append(keyword_str)
    print(len(segmented_comments))

    # 训练模型（假设您有标记好的数据集）
    # 这里使用假数据，您应该替换为实际的训练数据
    train_texts = []  # 请替换为实际的训练文本
    train_labels = []  # 请替换为实际的标签
    with open("trainword.json", encoding='utf-8') as f:
        data = json.load(f)

    # 添加数据结构调试输出
    logging.info(f"Loaded {len(data)} entries from training data")
    for i, d in enumerate(data[:5]):
        logging.info(f"Sample entry {i}: {d}")

    for entry in data:
        if random.random() < 0.0025:  # 根据需要调整概率以收集更多样本
            for d in entry:
                if isinstance(d, list) and len(d) >= 2 and isinstance(d[1], (int, float, str)):
                    try:
                        keywords = extract_keywords(d[0])
                        keyword_str = ','.join(keywords)
                        train_texts.append(keyword_str)
                        label = int(d[1])
                        if label in [0, 1, 5]:
                            train_labels.append(1)  # 正面
                        else:
                            train_labels.append(0)  # 负面
                    except ValueError as e:
                        logging.warning(f"Skipping invalid label: {d[1]}")
                else:
                    logging.warning(f"Skipping invalid data entry: {d}")

    logging.info(f"Filtered {len(train_texts)} training texts and {len(train_labels)} labels")

    if not train_texts or not train_labels:
        raise ValueError("Training texts or labels are empty. Check the training data and try again.")

    logging.info("Checking for existing model")
    model_save_path = './trained_model'

    if os.path.exists(model_save_path):
        logging.info("Loading existing model")
        model, tokenizer = load_model(model_save_path)
    else:
        logging.info("Training model")
        logging.info(f"train size: {len(train_texts)}")
        model = train_model(train_texts, train_labels, model_save_path)
        tokenizer = BertTokenizer.from_pretrained(model_save_path)

    # 进行情绪预测
    logging.info("Predicting emotions")
    predictions = predict_emotions(model, tokenizer, segmented_comments)

    # 输出预测结果
    for idx, (comment, prediction) in enumerate(zip(comments, predictions)):
        print(f"Comment {idx + 1}: {comment}")
        print(f"Emotion Prediction: {prediction}")
        print("----------------------")

    # 计时结束
    tEnd = time.time()
    logging.info(f"Execution took {tEnd - tStart} seconds.")
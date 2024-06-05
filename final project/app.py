from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import googleapiclient.discovery
import jieba
import jieba.analyse
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import random
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from tqdm import tqdm

app = Flask(__name__)
CORS(app)  # 允許跨域請求


# 初始化 YouTube API 客戶端
def get_youtube_client(api_key):
    return googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)


# 你的其他函數，例如 get_comment_replies, get_video_comments, segment_comments, keep_chinese_chars, extract_keywords 等等
# 抓取頂層留言的回覆
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


# 抓取影片的所有留言及其回覆
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

            # 抓取這個頂層留言的所有回覆
            total_reply_count = item['snippet']['totalReplyCount']
            if total_reply_count > 0:
                parent_id = item['id']
                replies = get_comment_replies(parent_id, youtube)
                comments.extend(replies)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments


# 使用 jieba 進行留言分詞
def segment_comments(comments):
    segmented_comments = [' '.join(jieba.lcut(comment)) for comment in comments]
    return segmented_comments


def keep_chinese_chars(text):
    pattern = re.compile(r'[^\u4e00-\u9fff]')
    chinese_text = re.sub(pattern, '', text)
    return chinese_text


def extract_keywords(text):
    text = keep_chinese_chars(text)
    return jieba.analyse.extract_tags(text, topK=10)


# 訓練模型
def train_model(train_texts, train_labels, model_save_path):
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

    # 分詞和編碼
    print("Tokenizing the data")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

    # 創建數據集
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,  # 每隔 10 步打印一次日志
    )

    # 使用 Trainer 訓練模型
    print("Starting training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # 保存模型
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return model


# 加載模型
def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


# 預測情緒
def predict_emotions(model, tokenizer, comments):
    # 分詞處理留言
    inputs = tokenizer(comments, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # 進行預測
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

    return predictions


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    video_url = data.get('video_url')
    api_key = data.get('api_key')

    if not video_url or not api_key:
        return jsonify({'error': 'Missing video_url or api_key'}), 400

    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        comments = get_video_comments(video_id, api_key)
        segmented_comments = segment_comments(comments)
        keywords = [extract_keywords(comment) for comment in segmented_comments]
        return jsonify({'comments': comments, 'keywords': keywords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

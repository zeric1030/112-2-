from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# 設置環境變量以防止 OpenMP 衝突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

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

# 設定靜態文件夾
app.config['UPLOAD_FOLDER'] = 'static'


# 初始化 YouTube API 客戶端
def get_youtube_client(api_key):
    return googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)


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


# 更新提取關鍵詞的函數，確保只提取兩個字以上的詞
def extract_keywords(text):
    text = keep_chinese_chars(text)
    keywords = jieba.analyse.extract_tags(text, topK=10)
    filtered_keywords = [word for word in keywords if len(word) > 1]
    return filtered_keywords


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


# 繪製文字雲
def create_wordcloud(text, mask_image_path, output_image_path, font_path):
    mask = np.array(Image.open(mask_image_path))
    wordcloud = WordCloud(font_path=font_path, width=800, height=500, background_color='white', mask=mask).generate(
        ' '.join([word for word in text.split() if len(word) > 1]))

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_image_path)
    plt.close()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    video_url = data.get('video_url')
    api_key = "AIzaSyCQvqDg-WLGcFzE6cW7C2Joc1jV7PEh-CU"  # 固定的 API 密鑰

    if not video_url:
        return jsonify({'error': 'Missing video_url'}), 400

    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        comments = get_video_comments(video_id, api_key)
        segmented_comments = segment_comments(comments)
        keywords = [extract_keywords(comment) for comment in segmented_comments]

        # 加載模型
        model_save_path = './trained_model'
        model, tokenizer = load_model(model_save_path)

        # 進行情緒預測
        predictions = predict_emotions(model, tokenizer, segmented_comments)

        # 分离正面和负面评论
        positive_comments = []
        negative_comments = []

        for idx, (comment, prediction) in enumerate(zip(segmented_comments, predictions)):
            if prediction in [1, 5]:
                positive_comments.append(comment)
            elif prediction in [2, 3, 4]:
                negative_comments.append(comment)

        # 生成文字雲
        font_path = "AdobeFanHeitiStd-Bold.otf"
        positive_text = ' '.join(positive_comments)
        negative_text = ' '.join(negative_comments)

        create_wordcloud(positive_text, 'taiwan.jpg',
                         os.path.join(app.config['UPLOAD_FOLDER'], 'positive_wordcloud.png'), font_path)
        create_wordcloud(negative_text, 'taiwan.jpg',
                         os.path.join(app.config['UPLOAD_FOLDER'], 'negative_wordcloud.png'), font_path)

        return jsonify({
            'comments': comments,
            'keywords': keywords,
            'positive_wordcloud': 'positive_wordcloud.png',
            'negative_wordcloud': 'negative_wordcloud.png'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)

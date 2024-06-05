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

# 主程式
#print("hwhwhw")
if __name__ == "__main__":
    # 計時開始
    print("Starting the script")
    tStart = time.time()

    api_key = "AIzaSyAYu6KqHx8E96iIM96WD5saOdF2RbfoQmE" # 替換為你的 YouTube Data API v3 密鑰
    video_url = input("請輸入影片 URL: ")
    video_id = video_url.split("v=")[1].split("&")[0]  # 確保只取到 video_id

    # 爬取留言
    print("Fetching comments")
    comments = get_video_comments(video_id, api_key)

    # 斷句處理
    print("Segmenting comments")
    old_segmented_comments = segment_comments(comments)
    segmented_comments = []
    for c in tqdm(old_segmented_comments, desc="Extracting keywords"):
        strs = extract_keywords(c)
        str = ','.join(strs)
        segmented_comments.append(str)
    print(len(segmented_comments))
    '''
    for c in segmented_comments:
        print(c)'''

    # 訓練模型（假設您有標記好的數據集）
    # 這裡使用假數據，您應該替換為實際的訓練數據
    

    print("Checking for existing model")
    model_save_path = './trained_model'

    if os.path.exists(model_save_path):
        print("Loading existing model")
        model, tokenizer = load_model(model_save_path)
    else:
        train_texts = []  # 請替換為實際的訓練文本
        train_labels = []  # 請替換為實際的標籤
        with open("trainword.json", encoding='utf-8') as f:  # 指定编码为 utf-8
            data = json.load(f)
        for i in tqdm(data, desc="Processing training data"):
            for j in i:
                x = random.random()
                if x <= 1:
                    strs = extract_keywords(j[0])
                    str = ','.join(strs)
                    train_texts.append(str)
                    train_labels.append(int(j[1]))
        print("Training model")
        print("train size:", len(train_texts))
        model = train_model(train_texts, train_labels, model_save_path)
        tokenizer = BertTokenizer.from_pretrained(model_save_path)

    # 進行情緒預測
    print("Predicting emotions")
    predictions = predict_emotions(model, tokenizer, segmented_comments)

    # 分离正面和负面评论
    positive_comments = []
    negative_comments = []

    for idx, (comment, prediction) in enumerate(zip(segmented_comments, predictions)):
        if prediction in [1, 5]:
            positive_comments.append(comment)
        elif prediction in [2, 3, 4]:
            negative_comments.append(comment)

    # 繪製文字雲
    def create_wordcloud(text, mask_image_path, output_image_path, font_path):
        mask = np.array(Image.open(mask_image_path))
        wordcloud = WordCloud(font_path=font_path, width=800, height=500, background_color='white', mask=mask).generate(text)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_image_path)
        plt.show()

    # 生成正面和負面评论的文字雲
    font_path = "AdobeFanHeitiStd-Bold.otf"  # 指定中文字體
    positive_text = ' '.join(positive_comments)
    negative_text = ' '.join(negative_comments)

    create_wordcloud(positive_text, 'taiwan.jpg', 'positive_wordcloud.png', font_path)
    create_wordcloud(negative_text, 'taiwan.jpg', 'negative_wordcloud.png', font_path)

    # 計時結束
    tEnd = time.time()

   

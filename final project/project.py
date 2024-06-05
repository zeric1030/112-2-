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
import logging
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import random
import re


# 設置日志級別
logging.basicConfig(level=logging.INFO)

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

        try:
            response = request.execute(num_retries=3)
        except Exception as e:
            logging.error(f"Error fetching replies: {e}")
            break

        logging.info(f"Fetching replies for comment ID {parent_id}, received {len(response['items'])} replies")

        for item in response['items']:
            reply = item['snippet']['textDisplay']
            replies.append(reply)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            logging.info(f"All replies fetched for comment ID {parent_id}. Total replies: {len(replies)}")
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

        try:
            response = request.execute(num_retries=3)
        except Exception as e:
            logging.error(f"Error fetching comments: {e}")
            break

        logging.info(f"Fetching comments for video ID {video_id}, received {len(response['items'])} comments")

        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(top_comment)

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

# 保留中文字符
def keep_chinese_chars(text):
    pattern = re.compile(r'[^\u4e00-\u9fff]')
    chinese_text = re.sub(pattern, '', text)
    return chinese_text

# 提取關鍵詞
def extract_keywords(text):
    text = keep_chinese_chars(text)
    return jieba.analyse.extract_tags(text, topK=10)

# 訓練模型
def train_model(train_texts, train_labels, model_save_path):
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    logging.info("Tokenizing the data")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

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

    logging.info("Starting training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return model

# 加載模型
def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

# 預測情緒
def predict_emotions(model, tokenizer, comments):
    inputs = tokenizer(comments, return_tensors='pt', padding=True, truncation=True, max_length=512)

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

    api_key = "AIzaSyCQvqDg-WLGcFzE6cW7C2Joc1jV7PEh-CU"  # 替換為你的 YouTube Data API v3 密鑰
    video_url = input("請輸入影片 URL: ")
    video_id = video_url.split("v=")[1].split("&")[0]  # 確保只取到 video_id

    # 抓取留言
    logging.info(f"Fetching comments for video ID {video_id}")
    comments = get_video_comments(video_id, api_key)

    if not comments:
        logging.error("Failed to fetch comments.")
    else:
        logging.info("Segmenting comments")
        old_segmented_comments = segment_comments(comments)
        segmented_comments = []
        for c in old_segmented_comments:
            keywords = extract_keywords(c)
            keyword_str = ','.join(keywords)
            segmented_comments.append(keyword_str)
        logging.info(f"Total segmented comments: {len(segmented_comments)}")

        train_texts = []
        train_labels = []
        with open("trainword.json", encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"Loaded {len(data)} entries from training data")
        for i, d in enumerate(data[:5]):
            logging.info(f"Sample entry {i}: {d}")

        for entry in data:
            if random.random() < 0.0025:
                for d in entry:
                    if isinstance(d, list) and len(d) >= 2 and isinstance(d[1], (int, float, str)):
                        try:
                            keywords = extract_keywords(d[0])
                            keyword_str = ','.join(keywords)
                            train_texts.append(keyword_str)
                            label = int(d[1])
                            if label in [0, 1, 5]:
                                train_labels.append(1)
                            else:
                                train_labels.append(0)
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

        logging.info("Predicting emotions")
        predictions = predict_emotions(model, tokenizer, segmented_comments)

        for idx, (comment, prediction) in enumerate(zip(comments, predictions)):
            print(f"Comment {idx + 1}: {comment}")
            print(f"Emotion Prediction: {prediction}")
            print("----------------------")

        tEnd = time.time()
        logging.info(f"Execution took {tEnd - tStart} seconds.")

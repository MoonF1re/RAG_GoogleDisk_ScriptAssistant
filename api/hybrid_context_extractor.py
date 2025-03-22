import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Инициализация модели и токенизатора
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def compute_cosine_similarity(vec1, vec2): #Сравнивает два вектора насколько они похожи друг на друга
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_transformer_embedding(text): #Преобразуем текст в эмбединг
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Берем среднее по токенам последнего скрытого состояния
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings


def get_keyword_score(query, text):
    # Простой поиск по ключевым словам: разбиваем запрос на слова и считаем совпадения
    query_tokens = re.findall(r'\w+', query.lower())
    text_lower = text.lower()
    score = sum(text_lower.count(token) for token in query_tokens)
    return score


def hybrid_extract_context(query, chunks, top_k=2, weight_transformer=0.7, weight_keyword=0.3):
    query_embedding = get_transformer_embedding(query)
    scores = []

    # Вычисляем оценки для каждого чанка
    for chunk in chunks:
        chunk_embedding = get_transformer_embedding(chunk)
        transformer_score = compute_cosine_similarity(query_embedding, chunk_embedding)

        keyword_score = get_keyword_score(query, chunk)
        # Нормализация keyword_score (можно настроить)
        keyword_score = keyword_score / (len(query.split()) or 1)

        final_score = weight_transformer * transformer_score + weight_keyword * (keyword_score/10)
        scores.append(final_score)

    # Сортировка чанков по финальному баллу
    scores = np.array(scores)
    top_indices = scores.argsort()[-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]

    return selected_chunks

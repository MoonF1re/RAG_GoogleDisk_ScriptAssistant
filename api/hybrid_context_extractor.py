import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import spacy
from langchain_core.documents import Document
from typing import List

# Инициализация модели и токенизатора для эмбеддингов
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Изменение: Добавлена поддержка русских стоп-слов для TF-IDF
# Изменение: Добавлен импорт typing.List для аннотации типов
nlp = spacy.load("ru_core_news_sm")
RUSSIAN_STOP_WORDS = list(nlp.Defaults.stop_words)

def compute_cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# Изменение: Добавлена обработка пустого текста в get_transformer_embedding
def get_transformer_embedding(text):
    if not text.strip():
        return np.zeros(768)  # Возвращаем нулевой вектор для пустого текста
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Изменение: Добавлена обработка пустых чанков и min_df=1 в TF-IDF
def compute_topic_distributions(chunks, n_topics=5):
    if not chunks or not any(chunk.strip() for chunk in chunks):
        return None, None, np.zeros((len(chunks), n_topics))
    vectorizer = TfidfVectorizer(stop_words=RUSSIAN_STOP_WORDS, min_df=1)
    try:
        tfidf_matrix = vectorizer.fit_transform(chunks)
        nmf_model = NMF(n_components=n_topics, random_state=42)
        topic_distributions = nmf_model.fit_transform(tfidf_matrix)
        return vectorizer, nmf_model, topic_distributions
    except ValueError:
        return None, None, np.zeros((len(chunks), n_topics))

def get_topic_score(query, chunks, vectorizer, nmf_model):
    if vectorizer is None or nmf_model is None:
        return [0] * len(chunks)
    query_tfidf = vectorizer.transform([query])
    query_topic = nmf_model.transform(query_tfidf)[0]
    _, _, topic_distributions = compute_topic_distributions(chunks, n_topics=len(nmf_model.components_))
    # Вычисляем косинусное сходство между темой запроса и каждого чанка
    scores = [compute_cosine_similarity(query_topic, td) for td in topic_distributions]
    return scores

def compute_ner_score(query, text):
    #Если ключевые сущности присутствуют или совпадают, чанк считается более релевантным.
    doc_query = nlp(query)
    doc_text = nlp(text)
    ents_query = {ent.text.lower() for ent in doc_query.ents}
    ents_text = {ent.text.lower() for ent in doc_text.ents}
    return len(ents_query.intersection(ents_text)) / len(ents_query) if ents_query else 0.0

def get_position_weight(index, total_chunks):
    if index < 0.3 * total_chunks or index > 0.7 * total_chunks:
        return 1.2
    return 1.0

# Изменение: Добавлена проверка на пустые чанки в hybrid_extract_context
# Изменение: Используются RUSSIAN_STOP_WORDS и min_df=1 для TF-IDF
def hybrid_extract_context(query, chunks, top_k=3,
                           weight_transformer=0.7, weight_tfidf=0.2,
                           weight_ner=0.1, weight_topic=0.2, weight_position=0.1):
    if not chunks or not any(chunk.strip() for chunk in chunks):
        return []

    query_embedding = get_transformer_embedding(query)
    tfidf_vectorizer = TfidfVectorizer(stop_words=RUSSIAN_STOP_WORDS, min_df=1)
    # Шаг 1: Получаем эмбеддинги и оценки всех методов
    query_embedding = get_transformer_embedding(query)
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = (query_tfidf.multiply(tfidf_matrix)).toarray().sum(axis=1)
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
        query_tfidf = tfidf_vectorizer.transform([query])
        tfidf_scores = (query_tfidf.multiply(tfidf_matrix)).toarray().sum(axis=1)
    except ValueError:
        tfidf_scores = np.zeros(len(chunks))

    vectorizer, nmf_model, topic_distributions = compute_topic_distributions(chunks, n_topics=5)
    topic_scores = get_topic_score(query, chunks, vectorizer, nmf_model)

    total_chunks = len(chunks)
    scores = []
    # Шаг 2: Объединяем оценки c чанками
    for i, chunk in enumerate(chunks):
        transformer_score = compute_cosine_similarity(query_embedding, get_transformer_embedding(chunk))
        ner_score = compute_ner_score(query, chunk)
        position_weight = get_position_weight(i, total_chunks)
        topic_score = topic_scores[i] if i < len(topic_scores) else 0
        final_score = (weight_transformer * transformer_score +
                       weight_tfidf * tfidf_scores[i] +
                       weight_ner * ner_score +
                       weight_topic * topic_score) * position_weight
        scores.append(final_score)
    scores = np.array(scores)

    embeddings = [get_transformer_embedding(chunk) for chunk in chunks]
    k = min(len(chunks), top_k)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)

    selected_chunks = []
    for cluster_label in range(k):
        indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_label]
        if indices:
            best_idx = indices[np.argmax(scores[indices])]
            selected_chunks.append(chunks[best_idx])

    unique_chunks = []
    threshold = 0.95
    for chunk in selected_chunks:
        duplicate = False
        for u_chunk in unique_chunks:
            sim = compute_cosine_similarity(get_transformer_embedding(chunk), get_transformer_embedding(u_chunk))
            if sim > threshold:
                duplicate = True
                break
        if not duplicate:
            unique_chunks.append(chunk)
    return unique_chunks

class HybridRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.last_context = ""

    def __call__(self, query, chat_history=""):
        if isinstance(query, dict):
            query = query.get("input", "")
        initial_chunks = self.base_retriever.invoke({"input": query, "chat_history": chat_history})
        texts = []
        for chunk in initial_chunks:
            if isinstance(chunk, dict):
                texts.append(chunk.get("page_content", str(chunk)))
            elif hasattr(chunk, "page_content"):
                texts.append(chunk.page_content)
            else:
                texts.append(str(chunk))
        refined_chunks = hybrid_extract_context(query, texts)
        self.last_context = "\n\n".join(refined_chunks)
        refined_docs = [Document(page_content=txt, metadata={}) for txt in refined_chunks]
        return refined_docs

    def similarity_search(self, query: str, **kwargs):
        chat_history = kwargs.get("chat_history", "")
        return self.__call__(query, chat_history)

    def with_config(self, **kwargs):
        return self
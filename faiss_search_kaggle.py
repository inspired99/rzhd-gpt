from sentence_transformers import SentenceTransformer
from load_env import load_all_classes

import faiss
import numpy as np
import pickle
import pandas as pd


def make_tfidf_embedding(query, vectorizer):
    query_tfidf = np.array(vectorizer.transform([query]).toarray(), dtype=np.float32)

    return query_tfidf


def make_bert_embedding(sentence, model):
    embedding = model.encode(sentence)

    return np.array(embedding)


# def get_df_by_ind(indexes):
#     return df_qa.loc[indexes, ['query', 'answer']]


def get_topn_by_tfidf_index(query, tfidf_vectorizer, tfidf_index, topn):
    tfidf_embedding = make_tfidf_embedding(query, tfidf_vectorizer)

    D, I = tfidf_index.search(tfidf_embedding, topn)

    pairs = [tuple([i, d]) for i, d in zip(I[0], D[0])]

    return pairs


def get_topn_by_bert_index(query, bert_model, bert_index, topn):
    bert_embedding = make_bert_embedding(query, bert_model).reshape(1, -1)

    D, I = bert_index.search(bert_embedding, topn)

    pairs = [tuple([i, d]) for i, d in zip(I[0], D[0])]

    return pairs


def merge_candidates(tfidf_pairs, bert_pairs, mode, topn):
    union_pairs = bert_pairs + tfidf_pairs

    if mode == 'cos':
        union_pairs = sorted(union_pairs, key=lambda pair: -pair[1])
    elif mode == 'l2':
        union_pairs = sorted(union_pairs, key=lambda pair: pair[1])

    result = []
    unique_indexes = set()

    for pair in union_pairs:
        if not pair[0] in unique_indexes:
            result.append(pair[0])
            unique_indexes.add(pair[0])

        if len(result) == topn:
            break

    return result


def retrieve_candidate(query, bert_model, tfidf_vectorizer, tfidf_index, bert_index, mode, retrieve_topn, final_topn):
    tfidf_pairs = get_topn_by_tfidf_index(query, tfidf_vectorizer, tfidf_index, topn=retrieve_topn)
    bert_pairs = get_topn_by_bert_index(query, bert_model, bert_index, topn=retrieve_topn)

    candidates = merge_candidates(tfidf_pairs, bert_pairs, mode, topn=final_topn)

    return candidates


def filter_indexes_by_train_type(train_type, train_type_to_boarder, tfidf_index, bert_index):
    if train_type:
        min_boarder, max_boarder = train_type_to_boarder[train_type]

        # print(min_boarder, max_boarder)
        # print(tfidf_index.ntotal, bert_index.ntotal)
        # RECONSTRUCT_N ??
        tfidf_vectors_train = tfidf_index.reconstruct_n(n0=min_boarder, ni=max_boarder-min_boarder)
        bert_vectors_train = bert_index.reconstruct_n(n0=min_boarder, ni=max_boarder-min_boarder)

        new_tfidf_index = faiss.IndexFlatIP(tfidf_vectors_train.shape[1])
        new_bert_index = faiss.IndexFlatIP(bert_vectors_train.shape[1])

        new_tfidf_index.add(tfidf_vectors_train)
        new_bert_index.add(bert_vectors_train)

    else:
        new_tfidf_index = tfidf_index
        new_bert_index = bert_index

    return new_tfidf_index, new_bert_index


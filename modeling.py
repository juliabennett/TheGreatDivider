import re
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def search_posts(phrase, engine):
    lemmatizer = WordNetLemmatizer()
    words = ["(^|[^a-z])" + lemmatizer.lemmatize(word)
                for word in word_tokenize(phrase)
                    if word not in stopwords.words('english')
                    and len(word) >= 3]

    if len(words) == 0:
        return None

    params = {'phrase': "|".join(words)}
    query = ["SELECT link_id, url, title FROM threads", 
             "WHERE title_lower ~ %(phrase)s"]
    found = pd.read_sql(" ".join(query), 
                       engine, 
                       params=params)
    
    if len(found['link_id']) == 0: 
        return None 

    link_ids = ', '.join(found['link_id'].apply(lambda lid: "'" + lid + "'"))
    query = ["SELECT clean_body as body, affil, link_id FROM cleaned", 
             "WHERE link_id IN (" + link_ids + ")"]
    data = pd.read_sql(" ".join(query), engine)
    
    valid = data[data['body'].apply(lambda text: len(text.split()) >= 10 
                                 and not bool(re.search("[^a-z]bot[^a-z]", text)))]
    
    if valid.shape[0] < 60: 
        return None
    
    return valid, found.set_index('link_id')

def compute_sims(bodies):
    vect = TfidfVectorizer(
                stop_words = 'english', 
                token_pattern="[a-z]{3,}", 
                min_df = .025, 
                norm = None)
    try: 
        X = vect.fit_transform(bodies)
    except:
        return None

    normed = normalize(X, axis=0)
    sims = (normed.T * normed).toarray() # This is just cosine similarity!

    return sims, vect.vocabulary_


def rank_words(sims, vocab):
    nx_graph = nx.from_numpy_matrix(sims)
    rank_scores = nx.pagerank(nx_graph)
    
    ranked_list = [] 
    for word, ind in vocab.items():
        ranked_list.append((rank_scores[ind], word, ind))
    ranked_df = pd.DataFrame(ranked_list, columns = ['rank_score', "word", "vocab_index"])
        
    return ranked_df


def sil_score(sims, labels):
    scores = []
    
    if len(set(labels)) == 1:
        return [0]*len(labels) 
    
    for i, label in enumerate(labels):
        for_b = []
        for comp_lab in set(labels) - set([label]):
            for_b.append(np.mean([sims[i, j] 
                                      for j, lab in enumerate(labels) 
                                          if lab == comp_lab]))
        b = np.max(for_b)
        
        for_a = [sims[i, j] 
                     for j, lab in enumerate(labels) 
                         if lab == label and i!=j]
        if len(for_a) == 0:
            scores.append(0)
            continue
        a = np.mean(for_a)
        
        if b > a:
            scores.append((a/b)-1)
        elif a > b: 
            scores.append(1-(b/a))
        else: 
            scores.append(0)
            
    return scores

def choose_num_clusts(sims):
    if sims.shape[0] <= 50:
        return None

    means = []
    possibles = [10, 15, 20]
    for n in possibles:
        clusterer = AgglomerativeClustering(
            n, 
            affinity='precomputed', 
            linkage='complete')
        clusterer.fit(1-sims)

        labels = clusterer.labels_
        sil_scores = sil_score(sims, labels)
        means.append(np.mean(sil_scores))

    return possibles[np.argmax(means)]

def model(phrase, affil, engine, fit=True):
    results = search_posts(phrase, engine)
    if results is None:
        return None
    
    data, found = results

    if min(sum(data['affil'] == 1), sum(data['affil'] == -1)) < 30:
        return None
    
    affiled = data[data['affil']==affil]
    
    sim_results = compute_sims(affiled['body'])
    if sim_results is None:
        return None
    original_sims, vocab = sim_results

    if not fit: 
        return original_sims

    ranked_df = rank_words(original_sims, vocab)
    ranked_df.sort_values('rank_score', ascending=False, inplace=True)
    ranked_df.reset_index(drop=True, inplace=True)
    
    best_inds = ranked_df['vocab_index'][:100]
    sims = original_sims[best_inds, :][:, best_inds]
    
    n = choose_num_clusts(sims)
    if n is None: 
        return None

    clusterer = AgglomerativeClustering(
                    n, 
                    affinity='precomputed', 
                    linkage='complete')
    clusterer.fit(1-sims)

    ranked_df['sil_score'] = np.nan
    ranked_df['label'] = None

    ranked_df.loc[:99, 'label'] = clusterer.labels_
    ranked_df.loc[:99, 'sil_score'] = sil_score(sims, clusterer.labels_)

    return ranked_df, sims, original_sims, affiled, found


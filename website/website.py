import re
import json
import pandas as pd
import numpy as np
import networkx as nx

from sqlalchemy import create_engine

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from nltk.corpus import stopwords

with open('credentials.txt', "rb") as f:
        creds = json.load(f)

password = creds['postgres']
engine = create_engine('postgres://postgres:' + password + '@localhost/affiled')

def search_posts(phrase):
    if not re.search("[a-zA-Z]", phrase):
        return None
    
    phrase = "".join([let.lower() if re.search(" |[a-zA-Z]", let.lower()) else " "
                                    for let in phrase])

    words = ["%" + word + "%" 
                for word in phrase.split()
                    if word not in stopwords.words('english')]

    params = {'phrase': "|".join(words)}
    query = ["SELECT link_id, url, title FROM threads", 
             "WHERE title_lower SIMILAR TO %(phrase)s"]
    found = pd.read_sql(" ".join(query), 
                       engine, 
                       params=params)
    
    if len(found['link_id']) == 0: 
        return None 

    link_ids = ', '.join(found['link_id'].apply(lambda lid: "'" + lid + "'"))
    query = ["SELECT clean_body as body, affil, link_id FROM cleaned", 
             "WHERE link_id IN (" + link_ids + ")"]
    data = pd.read_sql(" ".join(query), engine)
    
    valid = data[data['body'].apply(lambda text: len(text.split()) >= 5 and 'bot' not in text)]
    
    if valid.shape[0] < 60: 
        return None
    
    return valid, found.set_index('link_id')


def compute_sims(bodies):
    vect = TfidfVectorizer(
                stop_words = 'english', 
                token_pattern="[a-z]{3,}", 
                min_df =.025, 
                norm = None)
    try: 
        X = vect.fit_transform(bodies)
    except:
        return None

    normed = normalize(X, axis=0)
    sims = (normed.T * normed).toarray() # This is just cosine similarity!

    return sims, vect.vocabulary_, X


def rank_words(sims, vocab):
    nx_graph = nx.from_numpy_matrix(sims)
    rank_scores = nx.pagerank(nx_graph)
    
    ranked = []
    for word, ind in vocab.items():
        ranked.append((rank_scores[ind], word, ind))
        
    return sorted(ranked, reverse=True)[:100]


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

def package_clusts(best, labels, sil_scores):
    clusts = dict(zip(
                set(labels), 
                [{"clust_score": 0, 
                  "words": []} for label in set(labels)]
            ))
    
    for i, label in enumerate(labels):
        clusts[label]['clust_score'] += sil_scores[i]*(best[i][0]**2)
        clusts[label]['words'].append((best[i][1], 
                                       i, 
                                       sil_scores[i]*best[i][0]))
        
    for key in clusts.keys():
        clusts[key]['clust_score'] = clusts[key]['clust_score']/len(clusts[key]['words'])

    ordered = sorted([(clust['clust_score'], clust['words']) 
                          for clust in clusts.values()], 
                     reverse=True)
    return ordered

    
def get_urls(clust, X, affiled, found):
    word_inds = zip(*clust[1])[1]

    ordered_docs = sorted(list(enumerate(np.sum(X[:, word_inds], axis=1))), 
                    reverse = True, 
                    key = lambda pair: pair[1])

    rel_doc_inds = [ind for ind, rel_score in ordered_docs if rel_score >0]
    rel_scores = [rel_score for ind, rel_score in ordered_docs if rel_score >0]

    rel_docs = affiled.reset_index().loc[rel_doc_inds, :]
    rel_docs['rel_score'] = rel_scores

    by_thread = rel_docs.groupby('link_id')['rel_score']
    ordered_threads = by_thread.sum().sort_values(ascending=False)

    url_list = []
    for link_id in ordered_threads.index:
        row = found.loc[link_id, :]
        url_list.append((row['url'], row['title']))

    return url_list[:3]


def model(phrase, affil):
    results = search_posts(phrase)
    if results is None:
        return None
    
    data, found = results

    if min(sum(data['affil'] == 1), sum(data['affil'] == -1)) < 30:
        return None
    
    affiled = data[data['affil']==affil]
    
    sim_results = compute_sims(affiled['body'])
    if sim_results is None:
        return None
    sims, vocab, X = sim_results
    best = rank_words(sims, vocab)

    best_inds = zip(*best)[2]
    sims = sims[best_inds, :][:, best_inds]
    X = X[:, best_inds].toarray()
    
    n = choose_num_clusts(sims)
    if n is None: 
        return None

    clusterer = AgglomerativeClustering(
                    n, 
                    affinity='precomputed', 
                    linkage='complete')
    clusterer.fit(1-sims)

    labels = clusterer.labels_
    sil_scores = sil_score(sims, labels)
    
    top_clusts = package_clusts(best, labels, sil_scores)

    clust_urls = []
    for clust in top_clusts:
        clust_urls.append(get_urls(clust, X, affiled, found))

    return top_clusts, clust_urls

def scale_sizes(scores):
    return [min(10000*(score + .0001), 45) for score in scores]

def scale_tran(score):
    return max(min(np.sqrt(score) * 1000, 1), .6)

def prep_clusts(model_results):
    clusters, urls = model_results

    preped_clusts = []
    preped_urls = []
    for clust, url in zip(clusters, urls):
        words, _, scores = zip(*clust[1])
        if len(words) < 3 or len(words) > 10 or clust[0] <= 0:
            continue
        pairs = sorted(zip(scale_sizes(scores), words), reverse=True)
        preped_clusts.append({
            'score': scale_tran(clust[0]), 
            'words': [dict(zip(['size', 'text'], pair)) for pair in pairs]
        })
        preped_urls.append(url)
    return preped_clusts[:3],  preped_urls[:3]


app = Flask(__name__)

@app.route('/')
def reset():
    return render_template('website.html')

@app.route('/search/')
def search():
    phrase = request.args.get('keyword', " ")

    lib_model_results = model(phrase, 1)
    if lib_model_results is None:
        return jsonify(left=None, right=None, left_urls=None, right_urls = None)
    lib_clusts, lib_urls = prep_clusts(lib_model_results)

    con_model_results = model(phrase, -1)
    if con_model_results is None:
        return jsonify(left=None, right=None, left_urls=None, right_urls = None)
    con_clusts, con_urls = prep_clusts(con_model_results)

    return jsonify(
        left=lib_clusts, 
        right=con_clusts,
        left_urls = lib_urls, 
        right_urls = con_urls
    )

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)


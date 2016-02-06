import sys
import json
import numpy as np

from sqlalchemy import create_engine

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("../")
from modeling import model

with open('credentials.txt', "rb") as f:
        creds = json.load(f)

password = creds['postgres']
engine = create_engine('postgres://postgres:' + password + '@localhost/affiled')

def get_urls(model_results, label):

    ranked_df = model_results[0]
    affiled = model_results[3]
    found = model_results[4]

    words = list(ranked_df[ranked_df['label'] == label]['word'])

    counter = CountVectorizer(vocabulary = words)
    X = counter.fit_transform(affiled['body'])

    affiled['count_score'] = np.sum(X.toarray(), axis=1)

    by_threads = affiled.groupby('link_id')['count_score']
    ordered_threads = by_threads.sum().sort_values(ascending=False)

    url_pairs = []
    for link_id in ordered_threads.index:
        if ordered_threads[link_id] < 3:
            break

        row = found.loc[link_id, :]
        url_pairs.append((row['url'], row['title']))

    return url_pairs[:3]

def prep_clusts(model_results):
    ranked_df = model_results[0]

    ranked_df['display_score'] = ranked_df['sil_score']*(ranked_df['rank_score']**2)

    by_clusts = ranked_df.groupby('label')['display_score'].aggregate(np.mean)

    clusts = []
    for label in by_clusts.index:
        tup = (
            by_clusts[label], 
            list(ranked_df[ranked_df['label']==label]['word']), 
            get_urls(model_results, label)
        )
        if len(tup[1]) >= 3 and len(tup[1]) <= 15:
            clusts.append(tup)

    if len(clusts) < 2:
        return None

    clusts.sort(reverse=True)
    top_clusts = clusts[:3]

    return zip(*top_clusts)[1:]

app = Flask(__name__)

@app.route('/')
def reset():
    return render_template('website.html')

@app.route('/search/')
def search():
    phrase = request.args.get('keyword', " ")
    print phrase

    lib_model_results = model(phrase, 1, engine)
    if lib_model_results is None:
        return jsonify(success="few")

    lib_prepped_clusts = prep_clusts(lib_model_results)
    if lib_prepped_clusts is None:
        return jsonify(success="bad")
    lib_clusts, lib_urls = lib_prepped_clusts
    
    con_model_results = model(phrase, -1, engine)
    if con_model_results is None:
        return jsonify(success="few")

    con_prepped_clusts = prep_clusts(con_model_results)
    if con_prepped_clusts is None:
        return jsonify(success="bad")
    con_clusts, con_urls = con_prepped_clusts

    return jsonify(
        success = True,
        left = lib_clusts, 
        right = con_clusts,
        left_urls = lib_urls, 
        right_urls = con_urls
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


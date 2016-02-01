import re
import json
import time 
import pandas as pd 
import requests

from tqdm import tqdm
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def get_headers():
    with open('credentials.txt', "rb") as f:
        creds = json.load(f)

    client_id = creds['client_id']
    secret = creds['secret']
    password = creds['password']

    client_auth = requests.auth.HTTPBasicAuth(client_id, secret)
    post_data = {"grant_type": "password", "username": "juliaben", "password": password}
    auth_headers = {"User-Agent": "PoliAffils/0.1 by juliaben"}

    auth_url = "https://www.reddit.com/api/v1/access_token"
    auth_response = requests.post(auth_url, 
                                  auth=client_auth, 
                                  data=post_data, 
                                  headers=auth_headers)
    headers = {
        "Authorization": "bearer " + auth_response.json()['access_token'], 
        "User-Agent": "PoliAffils/0.1 by juliaben"
    }

    return headers

def breathe(response):
    if float(response.headers["X-Ratelimit-Remaining"]) <= 0: 
        time.sleep(float(response.headers["X-Ratelimit-Reset"]) + 1)

def get_ids(subred, engine, max_iters=100):
    print "GETTING IDS FOR:", subred

    headers = get_headers()

    base = "https://oauth.reddit.com/r/" + subred + "/hot?limit=100"
    url = base

    for _ in tqdm(range(max_iters)): 
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            headers = get_headers()
            continue

        breathe(response)

        data = json.loads(response.text)['data']

        ids = pd.DataFrame([(thread['data']['name'], subred)
                                for thread in data['children']
                                       if thread['data']['num_comments']>0], 
                            columns=['name', 'subred'])

        for index, name in zip(ids.index, ids['name']):
            params = {'name': name}
            query = "SELECT * FROM thread_ids WHERE name = %(name)s"
            if pd.read_sql(query, engine, params = params).shape[0] != 0:
                ids.drop(index, axis=0, inplace=True)
        
        ids.to_sql('thread_ids', engine, if_exists='append', index=False)

        if data['after'] is None:
            break

        url = base + "&after=" + data['after']

    print "DONE"

def get_comments(subred, affil, engine):
    print "GETTING COMMENTS FOR:", subred

    ids = pd.read_sql("SELECT * FROM thread_ids WHERE subred=%(subred)s", 
                      engine, 
                      params={'subred': subred})

    headers = get_headers()

    for name in tqdm(ids['name']):
        url = 'https://oauth.reddit.com/r/' + subred + '/comments/' + name[3:]
        response = requests.get(url, headers=headers)

        if response.status_code == 200: 
            try: 
                save_comments(json.loads(response.text), subred, affil, engine)
            except: 
                continue
            breathe(response)
        else: 
            headers = get_headers()

    print "DONE"

def cleaner(comments):
    exp = "[^ \(\)]*[a-zA-Z]\.[a-zA-Z][^ \(\)]*|\[removed\]|deleted|\n|\(|\)|\[|\]"
    cleaned = [re.sub(exp, " ", comment) if comment else " " 
                    for comment in tqdm(comments)]

    tokenized = [word_tokenize(comment) for comment in tqdm(cleaned)]
    filtered = [[word.lower() for word in comm if word.lower() not in stopwords.words('english')] 
                        for comm in tqdm(tokenized)]

    lemmatizer = WordNetLemmatizer()
    lemmatized = [[lemmatizer.lemmatize(word) for word in comm] 
                        for comm in tqdm(filtered)]

    return [" ".join(comment) for comment in lemmatized]

def save_comments(tree, subred, affil, engine):
    cols = [
        'subreddit_id',
        'link_id',
        'likes',
        'id',
        'author',
        'parent_id',
        'score',
        'enginetroversiality',
        'body',
        'downs',
        'subreddit',
        'name',
        'created',
        'author_flair_text',
        'created_utc',
        'ups',
        'distinguished', 
        'replies'
    ]
    df = pd.DataFrame(columns = cols)

    comments = tree[1]['data']['children']

    while comments: 
        comment = comments.pop()

        data = comment['data']

        replies = data.pop('replies', None)
        if replies: 
            comments += replies['data']['children']
            data['replies'] = len(replies['data']['children'])
        else:
            data['replies'] = 0

        df.loc[df.shape[0], :] = [data.get(col, None) for col in cols]

    df['subred'] = subred
    df['affil'] = affil

    for index, name in zip(df.index, df['name']):
        params = {'name': name}
        query = "SELECT * FROM comments WHERE name = %(name)s"
        if pd.read_sql(query, engine, params = params).shape[0] != 0:
            df.drop(index, axis=0, inplace=True)

    df.to_sql('comments', engine, if_exists='append', index=False)

    df = df[['name', 'body', 'affil', 'link_id', 'created_utc']]
    df['clean_body'] = cleaner(df['body'])
    df.drop_duplicates(inplace=True)
    df.to_sql("cleaned", engine, if_exists ='append', index=False)


def get_title(link_id, headers):
    url = "https://oauth.reddit.com/api/info?id=" + link_id
    response = requests.get(url, headers=headers)

    if response.status_code == 200: 
        try: 
            data = json.loads(response.text)['data']['children'][0]['data']
        except: 
            return None
        breathe(response)

    else: 
        headers = get_headers()
    
    return data['title'], data['created_utc'], data['permalink']

def get_all_titles(engine):
    all_thread_ids = pd.read_sql("SELECT name FROM thread_ids", engine)
    known_thread_ids = pd.read_sql("SELECT link_id FROM threads", engine)
    link_ids = list(set(all_thread_ids['name']) - set(known_thread_ids['link_id']))

    headers = get_headers()
    for lid in tqdm(link_ids):
        if not lid:
            continue

        title_result = get_title(lid, headers)
        if title_result is None:
            continue

        title, date, url = title_result
        df = pd.DataFrame({
                'title': [title], 
                'title_lower': [title.lower()], 
                'link_id':[lid], 
                'created_utc': [date], 
                'url': [url]})

        df.to_sql('threads', engine, if_exists='append', index=False)


if __name__ == "__main__":

    affils = {
        'conservative': -1,
        'conservatives': -1,
        'republicans': -1,
        'republican': -1,
        'democrats': 1,
        'liberal': 1,
        'libs': 1,
        'progressive': 1,
        'demsocialist': 1,
        'socialdemocracy': 1
    }

    engine = create_engine('postgres://julia@localhost/affiled')

    for subred in affils.keys():
        get_ids(subred, engine)

    for subred, affil in affils.items():
        get_comments(subred, affil, engine)

    get_all_titles(engine)



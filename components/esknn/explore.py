#%%
import tensorflow as tf
import tensorflow_hub as hub
from elasticsearch import Elasticsearch, helpers

#%%
es = Elasticsearch(hosts="http://192.168.86.122:9200")
#%%
es.ping()
#%%
embedding_version = 1
embedding = "nnlm-en-dim128/" + str(embedding_version)
embed = hub.KerasLayer('https://tfhub.dev/google/'+embedding)

#%%
settings = {
        "number_of_shards": 2,
        "number_of_replicas": 1,
        "elastiknn": True
}
mappings = {
    "properties": {
        "title": {
            "type": "text"
        },
        "embedding_version":{
            "type": "integer"
        },
        "title_vector": {
            "type": "elastiknn_dense_float_vector", # 1
            "elastiknn": {
                "dims": 128,                        # 2
                "model": "lsh",                     # 3
                "similarity": "cosine",             # 4
                "L": 99,                            # 5
                "k": 1                              # 6
            }
        }
    }
}
emb = es.indices.create(index="emb", mappings=mappings, settings=settings, ignore=400)

#%%

#%%
sentences = [
    "This is a simple sentense with no particular meaning",
    "The president of the united states is now Joe Biden",
    "The market sentiment is reflected in the stock prices"
    ]

for embedding_version in range(1, 3):
    embedding = "nnlm-en-dim128/" + str(embedding_version)
    embed = hub.KerasLayer('https://tfhub.dev/google/'+embedding)
    for sentence in sentences:
        doc = {
            "title": sentence,
            "embedding_version": embedding_version,
            "title_vector": {
                "values": embed(tf.constant([sentence])).numpy().tolist()[0]
            }
        }
        es.index(index="emb", document=doc)

#%%
newtext = "businesses observe a hype"

q = {
    "elastiknn_nearest_neighbors": {
        "field": "title_vector",
        "vec": {
            "values": embed(tf.constant([newtext])).numpy().tolist()[0],
        },
        "model": "lsh",
        "similarity": "cosine",
        "candidates": 10
    }
}
res=es.search(index='emb', query=q, _source=['title','title_vector','embedding_version'])
for i in res['hits']['hits']:
    print(i['_source']['title'], i['_source']['embedding_version'],i['_score'])
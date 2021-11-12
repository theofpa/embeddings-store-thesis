#%%
import tensorflow as tf
import tensorflow_hub as hub
from elasticsearch import Elasticsearch, helpers
import json
#%%
es = Elasticsearch(hosts="http://192.168.86.122:9200")
#%%
es.ping()
#%%
embed = hub.KerasLayer('https://tfhub.dev/google/nnlm-en-dim128/2')

#%%
settings = {
    "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1,
        "elastiknn": True
    },
    "mappings": {
        "dynamic": "true",
        "_source": {
            "enabled": "true"
        },
        "properties": {
            "title": {
                "type": "text"
            },
            "title_vector": {
                "type": "elastiknn_dense_float_vector",
                "elastiknn": {
                    "dims": 128,
                    "model": "lsh",
                    "similarity": "cosine",
                    "L": 99,
                    "k": 1
                }
            }
        }
    }
}
emb = es.indices.create(index="emb", mappings=settings, ignore=400)

#%%

#%%
text = "this is another example"
doc = {
    "title": text,
    "title_vector": {
        "values": embed(tf.constant([text])).numpy().tolist()[0]
    }
}
es.index(index="emb", document=doc)

#%%
newtext = "this is an example"

q = {
    "elastiknn_nearest_neighbors": {
        "field": "title_vector",
        "vec": {
            "values": embed(tf.constant([newtext])).numpy().tolist()[0],
        },
        "model": "lsh",
        "similarity": "cosine",
        "candidates": 1
    }
}

print(es.search(index='emb', query=q, _source=['title'])['hits']['hits'][0]['_source']['title'])
#%%
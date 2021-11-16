#%%
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from elasticsearch import Elasticsearch, helpers

#%%
es = Elasticsearch(hosts="http://192.168.86.122:9200")
#%%
es.ping()

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
        "model_version":{
            "type": "integer"
        },
        "title_vector": {
            "type": "elastiknn_dense_float_vector", # 1
            "elastiknn": {
                "dims": 768,                        # 2
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
sentences = [
    "This is a simple sentence with no particular meaning",
    "The president of the united states is now Joe Biden",
    "The market sentiment is reflected in the stock prices"
    ]

for model_version in range(3, 5):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/"+str(model_version), trainable=True)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]      # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    embed = tf.keras.Model(text_input, pooled_output)
    for sentence in sentences:
        doc = {
            "title": sentence,
            "model_version": model_version,
            "title_vector": {
                "values": embed(tf.constant([sentence])).numpy().tolist()[0]
            }
        }
        es.index(index="emb", document=doc)

#%%
newtext = "there is a hype in the all countries indexes"

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
res=es.search(index='emb', query=q, _source=['title','title_vector','model_version'])
for i in res['hits']['hits']:
    print(i['_source']['title'], i['_source']['model_version'],i['_score'])
#%%
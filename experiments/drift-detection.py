#%%
from datasets import load_dataset
import alibi_detect
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.models.pytorch import TransformerEmbedding
from functools import partial
from transformers import AutoTokenizer
import torch
import time
from sacred import Experiment
from sacred.observers import MongoObserver
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, MaxPool1D
from transformers import TFRobertaModel
from collections import Counter
ex = Experiment(name='drift-detection-v10')
ex.observers.append(MongoObserver(url='mongodb+srv://xxx:xxx@xxx.otmss.mongodb.net/uva?retryWrites=true', db_name='xxx'))

@ex.config
def cfg():
    dataset = 'go_emotions' # go_emotions, amazon_us_reviews
    subset = 'raw' # Books_v1_02, Music_v1_00, raw
    ref_size = 10000
    seed = 2
    h_size = 5000
    detector = "LSDD" # MMD, LSDD, KS, Classifier
    test_set = "h1" # h0, h1
    drift_attribute = "curiosity" # amazon star_rating, go_emotions curiosity
    drift_attribute_value = 1 # amazon 4, go_emotions 1
    text_field = 'text' # amazon review_body, go_emotions text
    classifier = 'distilroberta-base'
    embeddings = 'bert-base-cased'
    n_layers = 8
    max_len = 100  # max length for the tokenizer
    batch_size = 64


def remove_small_strings(lst, min_len):
    return [x for x in lst if len(x.split()) >= min_len]

@ex.automain
def detect_drift(dataset, subset, ref_size, seed, h_size, detector, test_set, classifier, embeddings, n_layers, max_len, drift_attribute, drift_attribute_value, text_field, batch_size):

    layers = [-_ for _ in range(1, n_layers + 1)]
    tokenizer = AutoTokenizer.from_pretrained(embeddings)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding = TransformerEmbedding(embeddings, embedding_type='hidden_state', layers=layers).to(device).eval()
    preprocess_fn = partial(preprocess_drift, model=embedding, tokenizer=tokenizer, max_len=max_len, batch_size=batch_size)
    labels = ['No ', 'Yes']

    class ClassifierTF(tf.keras.Model):
        def __init__(self) -> None:
            super(ClassifierTF, self).__init__()
            self.lm = TFRobertaModel.from_pretrained(classifier)
            self.lm.trainable = False  # freeze language model weights
            self.head = tf.keras.Sequential([Dense(512), LeakyReLU(alpha=.1), Dense(2)])

        def call(self, tokens) -> tf.Tensor:
            h = self.lm(**tokens).last_hidden_state
            h = tf.squeeze(MaxPool1D(pool_size=100)(h), axis=1)
            return self.head(h)

        @classmethod
        def from_config(cls, config):  # not needed for sequential/functional API models
            return cls(**config)

    model = ClassifierTF()


    ds=load_dataset(dataset, subset, split='train').shuffle(seed)
    shards = int(len(ds)/200000)
    words=40
    ds1=ds.shard(num_shards=shards, index=0)
    # ref = ds1[text_field][:ref_size]
    ref = remove_small_strings(ds1[text_field], min_len=words)[:ref_size]

    from alibi_detect.utils.prediction import tokenize_transformer
    batch_fn = partial(tokenize_transformer, tokenizer=tokenizer, max_len=max_len, backend='tf')
    # detector = MMDDrift(x_ref=ref, preprocess_fn=preprocess_fn, backend='tensorflow', n_permutations=1000)
    # detector = KSDrift(x_ref=ref, preprocess_fn=preprocess_fn, input_shape=(max_len,))
    # detector = LSDDDrift(x_ref=ref, preprocess_fn=preprocess_fn, backend='tensorflow')
    # detector = ClassifierDrift(x_ref=ref, model=model, preds_type='logits', n_folds=3, epochs=2, preprocess_batch_fn=batch_fn, train_size=None)

    if detector == "MMD":
        detector = alibi_detect.cd.MMDDrift(x_ref=ref, preprocess_fn=preprocess_fn, backend='tensorflow', n_permutations=1000)
    elif detector == "KS":
        detector = alibi_detect.cd.KSDrift(x_ref=ref, preprocess_fn=preprocess_fn, input_shape=(max_len,))
    elif detector == "LSDD":
        detector = alibi_detect.cd.LSDDDrift(x_ref=ref, preprocess_fn=preprocess_fn, backend='tensorflow')
    elif detector == "Classifier":
        detector = alibi_detect.cd.ClassifierDrift(x_ref=ref, model=model, preds_type='logits', n_folds=3, epochs=2, preprocess_batch_fn=batch_fn, train_size=None, batch_size=batch_size)

    start = time.time()
    ds2=ds.shard(num_shards=shards, index=1)
    if test_set == "h0":
        h0 = ds2[text_field][ref_size:ref_size+h_size]
        # h0 = remove_small_strings(ds2[text_field], min_len=words)[h_size]
        ex.log_scalar("filtered_dataset_size", len(h0))
        print(Counter(ds2[drift_attribute]))
        preds = detector.predict(h0)
        test_len=len(h0)

    elif test_set == "h1":
        h1 = ds2.filter(lambda x: x[drift_attribute]==drift_attribute_value)[text_field][:h_size]
        # h1 = remove_small_strings(ds2.filter(lambda x: x[drift_attribute]==drift_attribute_value)[text_field], min_len=words)[:h_size]
        ex.log_scalar("filtered_dataset_size", len(h1))
        print(Counter(ds2.filter(lambda x: x[drift_attribute]==drift_attribute_value)[drift_attribute]))
        preds = detector.predict(h1)
        test_len=len(h1)
        print("sample:")

    print(f'Ref size: {ref_size}, '
          f'Detector: {detector.meta["name"]}, '
          f'Time: {(time.time() - start):.2f}s, '
          f'Distance: {preds["data"]["distance"]}, '
          f'Test size: {test_len}, '
          f'Drift: {labels[preds["data"]["is_drift"]]}, '
          f'P-value: {preds["data"]["p_val"] if detector.meta["name"]!="KSDrift" else ""} ')

    for i in preds["data"].keys():
        if not ((i == "p_val" or i == "distance") and detector.meta["name"] == "KSDrift"):
            if type(preds["data"][i]) == float:
                ex.log_scalar(f'data.{i}', preds["data"][i])
    ex.log_scalar("runtime", time.time() - start)

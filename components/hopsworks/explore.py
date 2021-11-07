# install hopsworks:
# git clone https://github.com/logicalclocks/karamel-chef.git
# cd karamel-chef
# give some more love: modifyvm memory 32g
# ./run.sh ubuntu 1 hopsworks no-random-ports
#%%
import hsfs
conn = hsfs.connection(
    host = '127.0.0.1',
    port = 8181,
    project = 'demo_fs_meb10000',
    hostname_verification=False,
    api_key_value='nr6VqUVoTQGtyQxv.hqdeAsc8SQN8kShlymDIyCM5L8JqdKuG3rtxtrvkrOnXlmxvdNIPPPOqfpVRGz2W'
)
fs = conn.get_feature_store()
#%%
td_meta = fs.get_training_dataset("sales_model", 10)
#%%
td_meta.init_prepared_statement(external=True) # use external=True so or else it's trying to connect to vagrant's ip
td_meta.serving_keys
#%%
incoming_data = [(31,"2010-02-05",47),
                 (2,"2010-02-12",92),
                 (20,"2010-03-05",11),
                 (4,"2010-04-02",52),
                 (12,"2010-05-07",27)
                 ]
for i in incoming_data:
    serving_vector = td_meta.get_serving_vector({'store': i[0],'date': i[1], 'dept': i[2]})
    print (serving_vector)
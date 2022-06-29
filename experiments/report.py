#%%
import pandas as pd
import itertools
from incense import ExperimentLoader
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [15, 5]
image_path = "docs/thesis/images/"
colorsh0 = [
    'darkred',
    'red',
    'lightcoral',
    'mistyrose']
colorsh1 = [
    'darkblue',
    'blue',
    'cornflowerblue',
    'lightskyblue',
]
#%%
image_path = "docs/thesis/images/"
#%%
loader=ExperimentLoader(mongo_uri='mongodb+srv://xxx:xxx@xxx.otmss.mongodb.net/xxx',db_name='xxx')
exp = loader.find({"$and": [
    # {"tags": {"$in": ["VIEW1"]}},
    {"experiment.name": "drift-detection-v13"},
#    {"config.classifier": "distilroberta-base"},
]})

df=pd.DataFrame()
for i in exp.data:
    df = pd.concat(
        [
            df,pd.concat(
            [
                pd.DataFrame([i.config]),
                pd.DataFrame(i.metrics)
            ],axis=1)
         ], axis=0)
#%%
# 4 subplots with distances of each detector
f, a = plt.subplots(2,2)
ax=a.flatten()
for i, detector in enumerate(df['detector'].unique()):
    distances=df[df['detector']==detector] \
        .groupby(['h_size','test_set']).mean()['data.distance'].reset_index() \
        .pivot(columns='test_set',index=['h_size'],values='data.distance')
    distances.plot(title=detector, ax=ax[i], ylabel=detector+' Distance')
    ax[i].set_ylim(bottom=-0.0001)
# plt.ylim(bottom=0)
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.9, wspace=0.2, hspace=0.8)
plt.savefig(image_path+'ALL-detector-distances.eps', format='eps')
plt.show()


#%%
# Runtime across all detectors on in-distribution samples
a=df[(df["detector"].isin(['MMD','KS','Classifier','LSDD'])) & \
     (df["test_set"]=='h0') \
     ][['h_size','runtime','detector']] \
    .groupby(['h_size','detector']).mean()['runtime'].reset_index() \
    .pivot(index='h_size',columns='detector',values='runtime')
# (a-a.loc[1000]).plot(title="Drifted")
a.plot()
plt.legend(loc=0);
plt.ylabel("runtime")
plt.savefig(image_path+'ALL-detectors-h0-runtime.eps', format='eps')
plt.show()
#%%
# Runtime across all detectors on drifted samples
a=df[(df["detector"].isin(['MMD','KS','Classifier','LSDD'])) & \
   (df["test_set"]=='h1') \
   ][['h_size','runtime','detector']] \
    .groupby(['h_size','detector']).mean()['runtime'].reset_index() \
    .pivot(index='h_size',columns='detector',values='runtime')
# (a-a.loc[1000]).plot(title="Drifted")
a.plot()
plt.legend(loc=0);
plt.ylabel("runtime")
plt.savefig(image_path+'ALL-detectors-h1-runtime.eps', format='eps')
plt.show()
#%%
# Runtime across all detectors
a=df[(df["detector"].isin(['MMD','KS','Classifier','LSDD']))  \
     ][['h_size','runtime','detector']] \
    .groupby(['h_size','detector']).mean()['runtime'].reset_index() \
    .pivot(index='h_size',columns='detector',values='runtime')
# (a-a.loc[1000]).plot(title="Drifted")
a.plot()
plt.legend(loc=0);
plt.ylabel("runtime (seconds)")
plt.savefig(image_path+'ALL-detectors-runtime.eps', format='eps')
plt.show()
#%%
a=df[(df["detector"].isin(['MMD','KS','Classifier','LSDD']))  \
     ][['h_size','runtime','detector']] \
    .groupby(['h_size','detector']).mean()['runtime'].reset_index() \
    .pivot(index='h_size',columns='detector',values='runtime')
(a-a.loc[1000]).plot()
plt.legend(loc=1);
plt.ylabel("runtime (seconds")
plt.savefig(image_path+'ALL-detectors-zeroed-runtime.eps', format='eps')
plt.show()
#%%
h0=df[(df["detector"].isin(['MMD','KS','Classifier','LSDD'])) & \
     (df["test_set"]=='h0')
     ][['h_size','data.distance','detector']] \
    .groupby(['h_size','detector']).mean()['data.distance'].reset_index() \
.pivot(index='h_size',columns='detector',values='data.distance')
h0.plot(title="Original data")
plt.legend(loc=1);
plt.ylabel("distance")
plt.show()
#%%
h1=df[(df["detector"].isin(['MMD','KS','Classifier','LSDD'])) & \
     (df["test_set"]=='h1')
     ][['h_size','data.distance','detector']] \
    .groupby(['h_size','detector']).mean()['data.distance'].reset_index() \
.pivot(index='h_size',columns='detector',values='data.distance')
h1.plot(title="Drifted")
plt.legend(loc=1);
plt.ylabel("distance")
plt.show()
#%%
h0.rename(columns = {'Classifier':'H0-Classifier', 'KS':'H0-KS', 'LSDD':'H0-LSDD', 'MMD':'H0-MMD'}, inplace = True)
h1.rename(columns = {'Classifier':'H1-Classifier', 'KS':'H1-KS', 'LSDD':'H1-LSDD', 'MMD':'H1-MMD'}, inplace = True)
ax = h0.plot(color=colorsh0)
h1.plot(ax=ax, color=colorsh1)
plt.show()
 #%%
df[(df["detector"].isin(['MMD','KS','Classifier','LSDD'])) & \
   (df["dataset"]=='amazon_us_reviews') & \
   (df["test_set"]=='h1')
   ][['seed','h_size','runtime','detector']].groupby(['h_size','detector']).mean()['runtime'].reset_index()\
.pivot(index='h_size',columns='detector',values='runtime') \
    .plot()
plt.legend(loc=1);
plt.show()
#%%
toplot=df[(df["detector"]=='MMD') & \
          (df["dataset"]=='amazon_us_reviews') & \
          (df["test_set"]=='h1')
          ][['seed','h_size','runtime']] \
    .pivot(index='h_size',columns='seed',values='runtime').mean(axis=1) \
    .plot()
plt.legend(loc=1);
plt.show()
#%%

loader=ExperimentLoader(mongo_uri='mongodb+srv://xxx:xxx@xxx.otmss.mongodb.net/xxx',db_name='xxx')
exp = loader.find({"$and": [
    {"experiment.name": "drift-detection-v9"},
    #    {"config.classifier": "distilroberta-base"},
]})

df=pd.DataFrame()
for i in exp.data:
    df = pd.concat(
        [
            df,pd.concat(
            [
                pd.DataFrame([i.config]),
                pd.DataFrame(i.metrics)
            ],axis=1)
        ], axis=0)
#%%
for detector in ["LSDD","MMD","Classifier"]:
    for dataset in ["amazon_us_reviews"]:
        for y in ["data.distance"]:
            for hs in ["h1"]:
                if hs=="h0":
                    drift="non-drifted"
                else:
                    drift="drifted"
                toplot=df[(df["detector"]==detector) & \
                          (df["dataset"]==dataset) & \
                          (df["test_set"]==hs)
                          ][['seed','h_size',y]] \
                    .pivot(index='h_size',columns='seed',values=y) \
                    .plot(title=f"detector={detector} dataset={dataset} test_set={drift}")
                plt.legend(loc=1, title="seed")
                plt.ylabel(y)
                plt.show()
#%%
#%%
for detector in ["MMD"]:
    for dataset in ["amazon_us_reviews"]:
        for y in ["runtime"]:
            for hs in ["h0","h1"]:
                if hs=="h0":
                    drift="non-drifted"
                else:
                    drift="drifted"
                toplot=df[(df["detector"]==detector) & \
                          (df["dataset"]==dataset) & \
                          (df["test_set"]==hs)
                          ][['seed','h_size',y]] \
                    .pivot(index='h_size',columns='seed',values=y) \
                    .plot(title=f"detector={detector} dataset={dataset} test_set={drift}")
                plt.legend(loc=1, title="seed")
                plt.ylabel(y)
                plt.show()
#%%
detector="Classifier"
dataset="amazon_us_reviews"
hs="h0"
y="data.distance"
df[(df["detector"]==detector) & \
   (df["dataset"]==dataset) & \
   (df["test_set"]==hs)
   ][['seed','h_size',y]] \
    .pivot(index='h_size',columns='seed',values=y) \
    .plot(title=f"detector={detector} dataset={dataset} test_set={hs}")
plt.legend(loc=1, title="seed")
plt.ylabel(y)
plt.show()
#%%
# table output
df[(df["detector"].isin(['MMD','KS','Classifier','LSDD'])) & \
   (df["test_set"]=='h0') \
   ][['h_size','runtime','detector']] \
    .groupby(['h_size','detector']).mean()['runtime'].reset_index() \
    .pivot(index='h_size',columns='detector',values='runtime')
#%%
def latex_with_lines(df, *args, **kwargs):
    kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels
                                       + ['r'] * df.shape[1] + [''])
    res = df.to_latex(*args, **kwargs)
    return res.replace('\\\\\n', '\\\\ \\midrule\n')
#%%
# table 1 runtime columns hsize, index detector
table1=df[['h_size','runtime','detector','test_set']] \
    .groupby(['h_size','detector','test_set']).mean()['runtime'].reset_index() \
    .pivot(index=['test_set','detector'],columns='h_size',values='runtime')
print(latex_with_lines(table1, float_format='%.1f'))
#%%
# table 1 distance columns hsize, index detector

table1=df[['h_size','data.distance','detector','test_set']] \
    .groupby(['h_size','detector','test_set']).mean()['data.distance'].reset_index() \
    .pivot(index=['test_set','detector'],columns='h_size',values='data.distance').abs()
print(latex_with_lines(table1, float_format='%.5f'))
#%%
# table 1 distance columns detector, index hsize
table1=df[['h_size','data.distance','detector','test_set']] \
    .groupby(['h_size','detector','test_set']).mean()['data.distance'].reset_index() \
    .pivot(index=['test_set','h_size'],columns='detector',values='data.distance').abs()
print(latex_with_lines(table1, float_format='%.5f'))
#%%
# table 1 distance columns detector, index hsize
table1=df[['h_size','data.distance','detector','test_set']] \
    .groupby(['h_size','detector','test_set']).mean()['data.distance'].reset_index() \
    .pivot(index=['h_size'],columns=['detector','test_set'],values='data.distance').abs()
print(latex_with_lines(table1, float_format='%.5f'))
#%%
# table 1 runtime columns hsize, index detector
table1=df[['h_size','runtime','detector','test_set']] \
    .groupby(['h_size','detector']).mean()['runtime'].reset_index() \
    .pivot(index=['h_size'],columns=['detector'],values='runtime')
print(latex_with_lines(table1, float_format='%.1f'))
#%%

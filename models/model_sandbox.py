#%%
import model_helper as mh
import torch

ds1 = mh.Dataset()
ds1.load_dataset(folder='64K_5_v2_s1',splits=[.9,.1,0],split_type="imbalanced_method")
ds2 = mh.Dataset()
ds2.load_dataset(folder='64K_5_v2_s2',splits=[0,0,1])

model = mh.Model(ds1.data)
model.load_model("SAGE",K=5,F=16)
model.w = 0.25
model.w = 1


#%%

model.train_model(epochs=1000)

#%%

model.test_model(ds1.data)
print()
model.test_model(ds2.data)
import pandas as pd
from sklearn import tree

train = pd.read_csv("train_dataset.csv")
test = pd.read_csv("test.csv")

sim1 = train["sim_number"]
vnd1 = train["price_vnd"]
sim2 = test["sim_number"]

sim1 = sim1.to_numpy().reshape(-1, 1)
sim2 = sim2.to_numpy().reshape(-1, 1)

model = tree.DecisionTreeClassifier()
model.fit(sim1, vnd1)
vnd2 = model.predict(sim2)

test["price_vnd"] = vnd2
test.to_csv("result.csv", index=False)
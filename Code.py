import pandas as pd
from sklearn import tree

train = pd.read_csv("train_dataset.csv")
test = pd.read_csv("test.csv")

train["sim_number"] = train["sim_number"].astype(str)
test["sim_number"] = test["sim_number"].astype(str)

train["digit1"] = train["sim_number"].str[0]
train["digit2"] = train["sim_number"].str[1]
train["digit3"] = train["sim_number"].str[2]
train["digit4"] = train["sim_number"].str[3]
train["digit5"] = train["sim_number"].str[4]
train["digit6"] = train["sim_number"].str[5]
train["digit7"] = train["sim_number"].str[6]
train["digit8"] = train["sim_number"].str[7]
train["digit9"] = train["sim_number"].str[8]

test["digit1"] = test["sim_number"].str[0]
test["digit2"] = test["sim_number"].str[1]
test["digit3"] = test["sim_number"].str[2]
test["digit4"] = test["sim_number"].str[3]
test["digit5"] = test["sim_number"].str[4]
test["digit6"] = test["sim_number"].str[5]
test["digit7"] = test["sim_number"].str[6]
test["digit8"] = test["sim_number"].str[7]
test["digit9"] = test["sim_number"].str[8]

sim1 = train.drop(columns= ["price_vnd", "sim_number"])
sim2 = test.drop(columns= ["sim_number"])
vnd1 = train["price_vnd"]

model = tree.DecisionTreeClassifier()
model.fit(sim1, vnd1)
vnd2 = model.predict(sim2)

test["price_vnd"] = vnd2
test.to_csv("result.csv", index=False)

result = pd.read_csv("result.csv")
result = result.drop(columns= ["digit1","digit2","digit3","digit4","digit5","digit6","digit7","digit8","digit9"])
result.to_csv("result.csv", index=False)
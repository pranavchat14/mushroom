import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

seed = 42

df = pd.read_csv("mushrooms.csv")

X, y = df.drop("class", axis=1), df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed
)

ohe = OneHotEncoder(sparse=True)
X_train_enc = ohe.fit_transform(X_train)
X_test_enc = ohe.transform(X_test)

# X_train_enc = pd.DataFrame(X_train_enc, columns = ohe.get_feature_names())
# X_test_enc = pd.DataFrame(X_test_enc, columns = ohe.get_feature_names())

rf = RandomForestClassifier(max_depth=100, random_state=seed)
rf.fit(X_train_enc, y_train)

train_score = rf.score(X_train_enc, y_train) * 100
test_score = rf.score(X_test_enc, y_test) * 100

with open("encoder.pkl", "wb") as f:
    pickle.dump(ohe, f)

with open("rf.pkl", "wb") as m:
    pickle.dump(rf, m)

with open("scores.json", "w") as fd:
    json.dump({"accuracy_score": test_score}, fd)

with open("metrics.txt", "w") as outfile:
    outfile.write("Training accuracy: %2.1f%%\n" % train_score)
    outfile.write("Test accuracy: %2.1f%%\n" % test_score)

if __name__ == "__main__":
    print("Training Done!")

import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

seed = 42

df = pd.read_csv("mushrooms.csv")

X, y = df.drop("class", axis=1), df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed
)

y_train = y_train.astype("category").cat.codes
y_test = y_test.astype("category").cat.codes


ohe = OneHotEncoder(sparse=True)
X_train_enc = ohe.fit_transform(X_train)
X_test_enc = ohe.transform(X_test)

# X_train_enc = pd.DataFrame(X_train_enc, columns = ohe.get_feature_names())
# X_test_enc = pd.DataFrame(X_test_enc, columns = ohe.get_feature_names())

rf = RandomForestClassifier(max_depth=100, random_state=seed)
rf.fit(X_train_enc, y_train)

train_score = rf.score(X_train_enc, y_train) * 100
test_score = rf.score(X_test_enc, y_test) * 100

precision, recall, prc_thresholds = metrics.precision_recall_curve(
    y_test.values, predictions
)
fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, predictions)

avg_prec = metrics.average_precision_score(y_test, predictions)
roc_auc = metrics.roc_auc_score(y_test, predictions)

# Calculate feature importance in random forest
importances = rf.feature_importances_
labels = ohe.get_feature_names()
feature_df = pd.DataFrame(
    list(zip(labels, importances)), columns=["feature", "importance"]
)
feature_df = feature_df.sort_values(by="importance", ascending=False)

# # image formatting
# axis_fs = 18  # fontsize
# title_fs = 22  # fontsize
# sns.set(style="whitegrid", rc={"figure.figsize": (10, 28)})
# ax = sns.barplot(x="importance", y="feature", data=feature_df)
# ax.set_xlabel("Importance", fontsize=axis_fs)
# ax.set_ylabel("Feature", fontsize=axis_fs)  # ylabel
# ax.set_title("Random forest\nfeature importance", fontsize=title_fs)

# plt.tight_layout()
# plt.savefig("feature_importance.png", dpi=120)
# plt.close()

nth_point = math.ceil(len(prc_thresholds) / 1000)
prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
with open("prc.json", "w") as fd:
    json.dump(
        {
            "prc": [
                {"precision": float(p), "recall": float(r), "threshold": float(t)}
                for p, r, t in prc_points
            ]
        },
        fd,
        indent=4,
    )

with open("roc.json", "w") as fd:
    json.dump(
        {
            "roc": [
                {"fpr": float(fp), "tpr": float(tp), "threshold": float(t)}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        },
        fd,
        indent=4,
    )

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

import pandas as pd
import numpy as np
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    make_scorer,
)
from sklearn import linear_model
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import StandardScaler

f1_scorer = make_scorer(f1_score)


# def train_model(df_all, target_feature="rejected"):
#     df_train = df_all.loc[[fname for fname in df_all.index if "Bayer" not in fname]]
#     X = df_train.drop(target_feature, axis=1)
#     y = df_train[[target_feature]]
#     search = fit_model(X, y)
#     search.best_score_
#     return X, y, search


def fit_model(X, y):

    scaler = StandardScaler()
    logistic = linear_model.LogisticRegression(
        penalty="l2", solver="saga", class_weight={True: 1, False: 1}
    )
    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

    param_grid = {
        "logistic__class_weight": [
            {True: true_wgt, False: 1} for true_wgt in np.logspace(0, 2, 21)
        ]
    }

    search = GridSearchCV(
        pipe,
        param_grid,
        cv=3,
        scoring={
            "f1": make_scorer(f1_score),
            "recall": "recall",
            "precision": "precision",
        },
        refit="f1",
    )
    search.fit(X.values, y.values.flatten())
    return search


def get_model_stats(model, X, y, p_thr=0.5):
    d = dict(
        precision=precision_score(
            y.values, model.predict_proba(X.values)[:, 1] > p_thr
        ),
        recall=recall_score(y.values, model.predict_proba(X.values)[:, 1] > p_thr),
        f1=f1_score(y.values, model.predict_proba(X.values)[:, 1] > p_thr),
    )

    y_compare = y.copy()
    y_compare["reject_prediction"] = model.predict(X)
    y_compare["reject_prediction_prob"] = model.predict_proba(X)[:, 1]
    y_compare["is_correct"] = model.predict(X) == y.values.flatten()
    y_compare.sort_values(by="reject_prediction", ascending=False)

    confusion_mtx = pd.crosstab(
        y_compare["reject"],
        y_compare["reject_prediction_prob"] > p_thr,
        rownames=["Actual"],
        colnames=["Predicted"],
        margins=False,
    )
    return confusion_mtx, pd.Series(d)

import re
import gc
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


class TextModelPipeline:
    def __init__(self,
                 train_csv: str,
                 test_csv: str,
                 submission_csv: str,
                 model_choice: str = "bayes",
                 ngram_range: tuple = (3, 4),
                 vocab_fit: str = "train"
                 ):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.submission_csv = submission_csv
        self.model_choice = model_choice
        self.ngram_range = ngram_range
        self.vocab_fit = vocab_fit

        self.train_df = None
        self.test_df = None
        self.submission_df = None

        self.vectorizer = None
        self.X_train = None
        self.X_test = None

        self.model = None

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def load_data(self) -> None:
        self.train_df = pd.read_csv(self.train_csv)
        self.test_df = pd.read_csv(self.test_csv)
        self.submission_df = pd.read_csv(self.submission_csv)

        self.train_df['text'] = self.train_df['text'].apply(self.preprocess_text)
        self.test_df['text'] = self.test_df['text'].apply(self.preprocess_text)

        self.train_df = self.train_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    def vectorize_data(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            tokenizer=lambda x: x.split(),
            preprocessor=self.preprocess_text,
            strip_accents='unicode'
        )
        if self.vocab_fit == "train":
            self.vectorizer.fit(self.train_df['text'])
        else:
            self.vectorizer.fit(self.test_df['text'])

        self.X_train = self.vectorizer.transform(self.train_df['text'])
        self.X_test = self.vectorizer.transform(self.test_df['text'])
        gc.collect()

    def build_model(self) -> None:
        """
        Creates the model based on self.model_choice.
        Options:
          - "bayes": MultinomialNB,
          - "svm": Calibrated LinearSVC,
          - "lr": Logistic Regression,
          - "xgb": XGBoost,
          - "lgb": LightGBM,
          - "cat": CatBoost,
          - "voting": A soft-voting ensemble (Logistic Regression + SGDClassifier).
        """
        if self.model_choice == "bayes":
            self.model = MultinomialNB(alpha=0.0235)
        elif self.model_choice == "svm":
            svm = LinearSVC()
            self.model = CalibratedClassifierCV(svm, method='sigmoid')
        elif self.model_choice == "lr":
            self.model = LogisticRegression()
        elif self.model_choice == "xgb":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.01,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                use_label_encoder=False,
                eval_metric='auc'
            )
        elif self.model_choice == "lgb":
            self.model = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.7,
                colsample_bytree=0.7
            )
        elif self.model_choice == "cat":
            self.model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=4,
                subsample=0.8,
                silent=True
            )
        elif self.model_choice == "voting":
            lr_model = LogisticRegression()
            sgd_model = SGDClassifier(max_iter=5000, loss="modified_huber", random_state=42)
            self.model = VotingClassifier(
                estimators=[('lr', lr_model), ('sgd', sgd_model)],
                voting='soft'
            )
        else:
            raise ValueError(f"Unknown model_choice: {self.model_choice}")

    def train_model(self) -> None:
        if self.model_choice == "cat" and len(self.test_df["text"].values) <= 5:
            self.submission_df.to_csv("submission.csv", index=False)
            return
        self.build_model()
        self.model.fit(self.X_train, self.train_df['label'])
        gc.collect()

    def predict(self) -> pd.DataFrame:
        preds = self.model.predict_proba(self.X_test)[:, 1]
        submission = pd.DataFrame({'id': self.test_df["id"], 'generated': preds})
        submission.to_csv("submission.csv", index=False)
        return submission

    def run_pipeline(self) -> pd.DataFrame:
        self.load_data()
        self.vectorize_data()
        if self.model_choice == "cat" and len(self.test_df["text"].values) <= 5:
            # For CatBoost with a very small test set, training is skipped.
            return self.submission_df
        self.train_model()
        return self.predict()


if __name__ == "__main__":
    model_choice = "bayes"

    pipeline = TextModelPipeline(
        train_csv="/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv",
        test_csv="/kaggle/input/llm-detect-ai-generated-text/test_essays.csv",
        submission_csv="/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv",
        model_choice=model_choice,
        ngram_range=(3, 4),
        vocab_fit="train"
    )
    submission_df = pipeline.run_pipeline()
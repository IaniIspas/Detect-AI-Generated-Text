import gc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast


class AIGeneratedTextDetector:
    """
    A complete pipeline for detecting AI-generated text. This class loads training and test data,
    builds a custom byte-level BPE tokenizer, vectorizes tokenized texts with TF-IDF (using n-grams),
    and trains an ensemble of classifiers. Finally, it outputs prediction probabilities in a submission CSV.
    """

    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: str,
        submission_csv_path: str,
        lowercase: bool = False,
        vocab_size: int = 30522,
    ):
        """
        Parameters:
            train_csv_path (str): Path to the training CSV file.
            test_csv_path (str): Path to the test CSV file.
            submission_csv_path (str): Path to the sample submission CSV.
            lowercase (bool): Whether to lowercase texts during tokenization.
            vocab_size (int): Vocabulary size for the BPE tokenizer.
        """
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.submission_csv_path = submission_csv_path
        self.lowercase = lowercase
        self.vocab_size = vocab_size

        self.train_df = None
        self.test_df = None
        self.submission_df = None

        self.tokenizer = None
        self.tokenized_train_texts = None
        self.tokenized_test_texts = None
        self.tfidf_train = None
        self.tfidf_test = None
        self.ensemble_model = None

    def load_data(self) -> None:
        """Loads the train, test, and submission data from CSV files."""
        self.train_df = pd.read_csv(self.train_csv_path)
        self.test_df = pd.read_csv(self.test_csv_path)
        self.submission_df = pd.read_csv(self.submission_csv_path)

        self.train_df = self.train_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    def build_tokenizer(self) -> None:
        """Builds and trains a custom byte-level BPE tokenizer on the test set texts."""
        raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        normalization_steps = [normalizers.NFC()]
        if self.lowercase:
            normalization_steps.append(normalizers.Lowercase())
        raw_tokenizer.normalizer = normalizers.Sequence(normalization_steps)

        raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        bpe_trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens)

        test_dataset = Dataset.from_pandas(self.test_df[["text"]])

        def corpus_iterator():
            for i in range(0, len(test_dataset), 1000):
                # Yield a list of texts
                yield test_dataset[i : i + 1000]["text"]

        tqdm.write("Training custom BPE tokenizer on test texts...")
        raw_tokenizer.train_from_iterator(corpus_iterator(), trainer=bpe_trainer)

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    def tokenize_texts(self) -> None:
        """Tokenizes the text fields in both training and test datasets using the custom tokenizer."""
        tqdm.write("Tokenizing training texts...")
        self.tokenized_train_texts = self.train_df["text"].apply(self.tokenizer.tokenize).tolist()

        tqdm.write("Tokenizing test texts...")
        self.tokenized_test_texts = self.test_df["text"].apply(self.tokenizer.tokenize).tolist()

    @staticmethod
    def identity_tokenizer(text):
        """A dummy tokenizer that returns the input as-is."""
        return text

    def vectorize_texts(self) -> None:
        """
        Vectorizes tokenized texts using TF-IDF over word n-grams (n=3 to 5).
        First fits a vectorizer on the test tokens to extract a vocabulary,
        then uses this vocabulary to vectorize the training and test token lists.
        """
        tqdm.write("Fitting initial TF-IDF vectorizer on test tokens to extract vocabulary...")
        tfidf_temp = TfidfVectorizer(
            ngram_range=(3, 5),
            lowercase=False,
            sublinear_tf=True,
            analyzer="word",
            tokenizer=self.identity_tokenizer,
            preprocessor=self.identity_tokenizer,
            token_pattern=None,
            strip_accents="unicode",
        )
        tfidf_temp.fit(self.tokenized_test_texts)
        vocabulary = tfidf_temp.vocabulary_

        tqdm.write("Vectorizing training and test tokens using fixed vocabulary...")
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(3, 5),
            lowercase=False,
            sublinear_tf=True,
            vocabulary=vocabulary,
            analyzer="word",
            tokenizer=self.identity_tokenizer,
            preprocessor=self.identity_tokenizer,
            token_pattern=None,
            strip_accents="unicode",
        )
        self.tfidf_train = tfidf_vectorizer.fit_transform(self.tokenized_train_texts)
        self.tfidf_test = tfidf_vectorizer.transform(self.tokenized_test_texts)

        del tfidf_temp, tfidf_vectorizer
        gc.collect()

    def train_ensemble(self) -> None:
        """Trains an ensemble classifier using multiple models and soft voting."""
        tqdm.write("Training ensemble classifier...")

        y_train = self.train_df["label"].values

        nb_model = MultinomialNB(alpha=0.0235)
        sgd_model = SGDClassifier(max_iter=9000, tol=3e-4, loss="modified_huber")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=0.5,
            use_label_encoder=False,
            eval_metric="auc",
        )
        lgb_model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.7,
            colsample_bytree=0.7,
        )
        cat_model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=4,
            subsample=0.8,
            silent=True,
        )

        self.ensemble_model = VotingClassifier(estimators=[
                ("sgd", sgd_model),
                ("nb", nb_model),
                ("xgb", xgb_model),
                ("lgb", lgb_model),
                ("cat", cat_model),
            ],
            weights=[0.23, 0.08, 0.23, 0.23, 0.23],
            voting="soft",
            n_jobs=-1,
        )
        self.ensemble_model.fit(self.tfidf_train, y_train)
        gc.collect()

    def predict(self):
        """
        Generates predictions on the test data.
        If the test set is very small (<=5 texts), simply writes the submission file as-is.
        Otherwise, predicts the probability of the positive class.
        """
        if len(self.test_df["text"].values) <= 5:
            self.submission_df.to_csv("submission.csv", index=False)
            return self.submission_df
        else:
            preds = self.ensemble_model.predict_proba(self.tfidf_test)[:, 1]
            self.submission_df["generated"] = preds
            self.submission_df.to_csv("submission.csv", index=False)
            return self.submission_df

    def run_pipeline(self):
        """Runs the complete pipeline: load data, tokenize, vectorize, train, and predict."""
        self.load_data()
        self.build_tokenizer()
        self.tokenize_texts()
        self.vectorize_texts()
        if len(self.test_df["text"].values) > 5:
            self.train_ensemble()
        return self.predict()

if __name__ == "__main__":
    detector = AIGeneratedTextDetector(
        train_csv_path="/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv",
        test_csv_path="/kaggle/input/llm-detect-ai-generated-text/test_essays.csv",
        submission_csv_path="/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv",
        lowercase=False,
        vocab_size=30522,
    )
    submission_df = detector.run_pipeline()
    print("Submission predictions:")
    print(submission_df.head())
import re
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.functional import softmax
from tqdm import tqdm

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)


class TransformerPipeline:
    def __init__(self,
                 train_csv: str,
                 test_csv: str,
                 submission_csv: str,
                 model_type: str = "bert",
                 batch_size: int = 16,
                 num_epochs: int = 3,
                 learning_rate: float = 5e-5):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.submission_csv = submission_csv
        self.model_type = model_type.lower()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.train_df = None
        self.test_df = None
        self.submission_df = None

        self.tokenizer = None
        self.model = None

        self.train_loader = None
        self.test_loader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "bert":
            self.model_path = "/kaggle/input/bert-base-uncased-zip/bert-base-uncased/bert-base-uncased"
        elif self.model_type == "distilbert":
            self.model_path = "/kaggle/input/distilbert-base-uncased/distilbert-base-uncased/distilbert-base-uncased"
        else:
            raise ValueError("model_type must be either 'bert' or 'distilbert'.")

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

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, labels=None, max_length=512):
            self.texts = texts
            self.labels = labels
            self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.texts)

    def prepare_datasets(self) -> None:
        if self.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        elif self.model_type == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)

        train_dataset = self.TextDataset(self.train_df['text'], self.tokenizer, labels=self.train_df['label'])
        test_dataset = self.TextDataset(self.test_df['text'], self.tokenizer)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

    def build_model(self) -> None:
        if self.model_type == "bert":
            self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=2)
        elif self.model_type == "distilbert":
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path, num_labels=2)
        self.model.to(self.device)

    def train_epoch(self, dataloader, optimizer) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(self.device)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        return total_loss / len(dataloader)

    def train_model(self) -> None:
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            train_loss = self.train_epoch(self.train_loader, optimizer)
            print(f'Train Loss: {train_loss:.4f}')

    def predict(self) -> np.ndarray:
        self.model.eval()
        preds = []
        progress_bar = tqdm(self.test_loader, desc="Predicting")
        with torch.no_grad():
            for batch in progress_bar:
                inputs = {key: val.to(self.device) for key, val in batch.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds.append(logits.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        preds = softmax(torch.tensor(preds), dim=-1)[:, 1].numpy()
        return preds

    def run_pipeline(self) -> None:
        self.load_data()
        self.prepare_datasets()
        self.build_model()
        self.train_model()
        preds = self.predict()
        submission = pd.DataFrame({'id': self.test_df["id"], 'generated': preds})
        submission.to_csv("submission.csv", index=False)
        print("Submission file written.")


if __name__ == "__main__":
    chosen_model = "bert"
    pipeline = TransformerPipeline(
        train_csv="/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv",
        test_csv="/kaggle/input/llm-detect-ai-generated-text/test_essays.csv",
        submission_csv="/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv",
        model_type=chosen_model,
        batch_size=16,
        num_epochs=3,
        learning_rate=5e-5
    )
    pipeline.run_pipeline()
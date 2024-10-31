import re
import unicodedata
from pathlib import Path

import contractions
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Embedding, TextVectorization
from keras.metrics import classification_report
from keras.models import Sequential
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


class SentimentModel:

    def __init__(
        self,
        dataset: str,
        seed=42,
        sentiment_field="sentiment",
        text_field="Tweet"
    ):
        self.data_path = dataset
        self.seed = seed
        self.sentiment_field = sentiment_field
        self.text_field = text_field
        self.label_encoder = LabelEncoder()
        self.vectorize_layer = None
        self.history = None

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, dataset: str):
        dir_name = dataset.split("/")[1]
        path = Path(dir_name)
        if not path.exists():
            self.dataset_download(path, dataset)
        if not path.is_dir():
            raise ValueError(f"La carpeta {path} no es un directorio valido")
        self._data_path = list(path.glob('*.csv'))[0]

    def dataset_download(self, path: Path, value: str, unzip=True):
        print("Descargando datos ...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            value,
            path=path,
            unzip=unzip
        )

    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = pd.read_csv(self.data_path)
        return self._data

    def data_value_count(self):
        return self.data[self.sentiment_field].value_counts()

    @staticmethod
    def _remove_stopwords(words: str):
        tokens = words.split()
        filtered_words = [
            word for word in tokens if word.lower() not in STOPWORDS
        ]
        return " ".join(filtered_words)

    @staticmethod
    def _preprocess_text(text):
        text = text.lower()
        # Remover puntuacion
        text = re.sub(r"[\"(),¡!¿?:;'>]", "", text)  # Corregido para espacios!
        # Remover caracteres no ASCII
        text = unicodedata.normalize("NFKD", text). \
            encode("ascii", "ignore").decode("utf-8", "ignore")
        # Remover stopwords
        text = SentimentModel._remove_stopwords(text)
        return text

    def preprocess_data(self):
        if self.data.duplicated().sum():
            raise ValueError("El dataset tiene valores duplicados")
        self.data[self.text_field] = self.data[self.text_field]. \
            apply(contractions.fix)
        print(f"Sin contracciones ...\n{self.data[self.text_field]}")
        self.data[self.text_field] = self.data[self.text_field]. \
            apply(self._preprocess_text)
        print(f"Preprocesado ...\n{self.data[self.text_field]}")
        self.data[self.sentiment_field] = self.label_encoder.fit_transform(
            self.data[self.sentiment_field]
        )
        print(f"Variable codificada ...\n{self.data[self.sentiment_field]}")
        train, test = train_test_split(
            self.data,
            test_size=0.2,
            stratify=self.data[self.sentiment_field],
            random_state=self.seed
        )
        train, val = train_test_split(
            train,
            test_size=0.2,
            stratify=train[self.sentiment_field],
            random_state=self.seed
        )
        self.X_train, self.y_train = (
            train[self.text_field],
            train[self.sentiment_field]
        )
        self.X_val, self.y_val = (
            val[self.text_field],
            val[self.sentiment_field]
        )
        self.X_test, self.y_test = (
            test[self.text_field],
            test[self.sentiment_field]
        )
        print(f"Tamaño de datos de entrenamiento: {train.shape}")
        print(
            f"Shape X_train: {self.X_train.shape}"
            + f"Shape y_train: {self.y_train.shape}"
        )
        print(f"Tamaño de datos de validacion: {val.shape}")
        print(
            f"Shape X_val: {self.X_val.shape}"
            + f"Shape y_val: {self.y_val.shape}"
        )
        print(f"Tamaño de datos de prueba: {test.shape}")
        print(
            f"Shape X_test: {self.X_test.shape}"
            + f"Shape y_test: {self.y_test.shape}"
        )

    def plot_distribution(self):
        unique, counts = np.unique(self.y_train, return_counts=True)
        plt.bar(unique, counts)
        plt.xlabel("Sentimiento")
        plt.ylabel("Frecuencia")
        plt.title("Distribucion de categorias de sentimiento")
        plt.show()

    def convert_to_tensor(self):
        self.X_train_tf = tf.convert_to_tensor(self.X_train)
        self.X_val_tf = tf.convert_to_tensor(self.X_val)
        self.X_test_tf = tf.convert_to_tensor(self.X_test)
        print(f"The X_train_tf: \n{self.X_train_tf}")

    def create_vectorizer(self):
        self.vectorize_layer = TextVectorization(
            standardize=None,
            split="whitespace",
            max_tokens=None,
            output_mode="int",
            output_sequence_length=None
        )
        self.vectorize_layer.adapt(self.X_train_tf)
        print(
            f"Vocabulario (primeros 100 campos):"
            f"\n{self.vectorize_layer.get_vocabulary()[:100]}"
        )
        print(f"Configuracion:\n{self.vectorize_layer.get_config()}")

    def plot_longitud_secuencias(self):
        sequence_lengths = [len(text.split()) for text in self.X_train]
        plt.hist(sequence_lengths, bins=50)
        plt.xlabel('Longitud de Secuencia')
        plt.ylabel('Frecuencia')
        plt.show()

    def build_model(self):
        self.model = Sequential()
        self.model.add(self.vectorize_layer)
        self.model.add(Embedding(input_dim=10000, output_dim=300))
        self.model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.8))
        print(f"Num clases del label encoder: {self.label_encoder.classes_}")
        self.model.add(Dense(
            len(self.label_encoder.classes_),
            activation='softmax')
        )
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self):
        self.history = self.model.fit(
            self.X_train_tf,
            self.y_train,
            validation_data=(self.X_val_tf, self.y_val),
            epochs=20,
            callbacks=[
                EarlyStopping(patience=4)
            ]
        )

    def evaluate_model(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Predecir
        y_pred_train = self.model.predict(self.X_train_tf)
        y_pred_test = self.model.predict(self.X_test_tf)

        # Indexar
        y_pred_train = np.argmax(y_pred_train, axis=1)
        y_pred_test = np.argmax(y_pred_test, axis=1)

        # Reportar
        self.report_train = pd.DataFrame(classification_report(
            self.y_train, y_pred_train, output_dict=True
        )).transpose()
        self.report_test = pd.DataFrame(classification_report(
            self.y_test, y_pred_test, output_dict=True
        )).transpose()

        return self.report_train, self.report_test

    def save_model(self, model_name="base_model"):
        self.model.save(f"{model_name}.h5")
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(f"history_{model_name}.csv", index=False)
        self.report_train.to_csv(f"report_train_{model_name}.csv", index=False)
        self.report_test.to_csv(f"report_test_{model_name}.csv", index=False)


if __name__ == '__main__':
    model_a = SentimentModel(
        dataset="evilspirit05/tweet-gpt"
    )
    print(f"The data is in: {model_a.data_path}")
    print(model_a.data_value_count())
    model_a.preprocess_data()
    model_a.plot_distribution()
    model_a.convert_to_tensor()
    print(model_a.X_test_tf)
    model_a.create_vectorizer()
    model_a.plot_longitud_secuencias()
    model_a.build_model()
    model_a.train_model()
    model_a.evaluate_model()
    model_a.save_model()

import pandas as pd
from tokenizerFile import Tokenizer
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
nltk.download('punkt')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from collections import defaultdict
from matplotlib import pyplot as plt

def main():

    # import the CSV file with data and transform it into a pandas dataframe
    CSV_data = pd.read_csv('cleaned_data.csv')
    #print(data.head())

    # tokenize data
    myTokenizer = Tokenizer()
    test_data = CSV_data.iloc[0:4]      # test data to tokanize with (first 5 rows)
    tokenized_data:pd.DataFrame = myTokenizer.tokenize_dataframe(CSV_data)
    tokenized_data.to_csv('tokenized_data.csv', index=False)

    # load the tokenized data
    data = pd.read_csv('tokenized_data.csv')

    # build a vocabulary index from tokenized texts, index 0 for padding
    def build_vocab(tokenized_texts):
        vocab = defaultdict(lambda: len(vocab))
        _ = vocab['<PAD>']
        for tokens in tokenized_texts:
            for token in tokens:
                _ = vocab[token]
        return dict(vocab)

    vocab = build_vocab(data['Text'])
    vocab_size = len(vocab)

    # convert tokens to sequences, replaces each token with its corresponding integer index
    def tokens_to_sequences(tokenized_texts, vocab):
        return [[vocab[token] for token in tokens if token in vocab] for tokens in tokenized_texts]

    data['Text'] = tokens_to_sequences(data['Text'], vocab)

    # pad sequences to ensure they all have the same length
    MAX_SEQUENCE_LENGTH = 200
    padded_sequences = pad_sequences(data['Text'], maxlen=MAX_SEQUENCE_LENGTH)

    # labels
    labels = data['label'].values

    # split data into training and test sets (80/20 for now)
    train_size = int(len(data) * 0.8)
    X_train = padded_sequences[:train_size]
    X_test = padded_sequences[train_size:]
    y_train = labels[:train_size]
    y_test = labels[train_size:]

    # model
    model = Sequential()
    model.add(Embedding(vocab_size, 100,))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # if validation loss does not decrease for 3 epochs, stop training
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # train
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # evaluate and save
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    # model.save('fake_news_model.keras')

    # plot training & validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

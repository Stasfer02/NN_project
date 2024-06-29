import pandas as pd
from tokenizerFile import Tokenizer
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sns

# making sure punkt tokenizer is downloaded
nltk.download('punkt')


def main():

    # import the CSV file with data and transform it into a pandas dataframe
    csv_data = pd.read_csv('cleaned_data.csv')

    # combine title and text into one column, but make titles uppercase
    csv_data['combined'] = csv_data['title'].str.upper() + " " + csv_data['Text']

    # Tokenize data
    my_tokenizer = Tokenizer()
    tokenized_data = my_tokenizer.tokenize_dataframe(csv_data[['combined', 'label']])
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

    vocab = build_vocab(data['combined'])
    vocab_size = len(vocab)

    # convert tokens to sequences, replaces each token with its corresponding integer index
    def tokens_to_sequences(tokenized_texts, vocab):
        return [[vocab[token] for token in tokens if token in vocab] for tokens in tokenized_texts]

    data['combined'] = tokens_to_sequences(data['combined'], vocab)

    # pad sequences to ensure they all have the same length
    MAX_SEQUENCE_LENGTH = 200
    padded_sequences = pad_sequences(data['combined'], maxlen=MAX_SEQUENCE_LENGTH)

    # labels
    labels = data['label'].values

    # k-fold cv
    kf = KFold(n_splits=10, shuffle=True)
    fold_no = 1

    # confusion matrix
    total_cm = np.array([[0, 0], [0, 0]])

    # lists to store accuracy/loss for each fold
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for train_index, test_index in kf.split(padded_sequences):
        print(f'Training fold {fold_no}...')

        X_train, X_test = padded_sequences[train_index], padded_sequences[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # model
        model = Sequential()
        model.add(Embedding(vocab_size, 100))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # if validation loss does not decrease for 4 epochs, stop training
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

        # train
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        # predict on test set
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        # calculate confusion matrix and add it to total confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        total_cm += cm

        train_accuracies.append(history.history['accuracy'])
        val_accuracies.append(history.history['val_accuracy'])
        train_losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])

        fold_no += 1

    # ensure same length by padding with None
    max_len = max(len(lst) for lst in train_accuracies)
    for i in range(len(train_accuracies)):
        if len(train_accuracies[i]) < max_len:
            train_accuracies[i] = train_accuracies[i] + [None] * (max_len - len(train_accuracies[i]))
            val_accuracies[i] = val_accuracies[i] + [None] * (max_len - len(val_accuracies[i]))
            train_losses[i] = train_losses[i] + [None] * (max_len - len(train_losses[i]))
            val_losses[i] = val_losses[i] + [None] * (max_len - len(val_losses[i]))
    # convert lists to numpy arrays
    train_accuracies = np.array(train_accuracies, dtype=np.float32)
    val_accuracies = np.array(val_accuracies, dtype=np.float32)
    train_losses = np.array(train_losses, dtype=np.float32)
    val_losses = np.array(val_losses, dtype=np.float32)
    # calculate average loss and accuracy
    avg_train_accuracy = np.nanmean(train_accuracies, axis=0)
    avg_val_accuracy = np.nanmean(val_accuracies, axis=0)
    avg_train_loss = np.nanmean(train_losses, axis=0)
    avg_val_loss = np.nanmean(val_losses, axis=0)

    # plot the total confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Aggregated Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # model.save('fake_news_model.keras')

    # plot training & validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_accuracy, label='Train Accuracy')
    plt.plot(avg_val_accuracy, label='Val Accuracy')
    plt.title('Model Accuracy Average')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Val Loss')
    plt.title('Model Loss Average')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

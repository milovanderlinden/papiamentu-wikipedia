"""
Taken from https://towardsdatascience.com/neural-machine-translation-with-python-c2f0a34f7dd
"""

import os
import glob
import numpy as np
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def load_data(lang):
    """
    Load dataset
    """
    out = []
    files = sorted(
        glob.glob('data/testament-nobo/san-mateo/*.' + lang + '.txt'))
    for f in files:
        with open(f, 'r') as infile:
            data = infile.read()
            lines = data.split('\n')

            out = out + lines
    print(out)
    return out


def tokenize(x):
    """
    Tokenize

    Turn each sentence into a sequence of words ids using Keras’s Tokenizer function. 
    Use this function to tokenize first_sentences and second_sentences.
    """
    x_tk = Tokenizer(char_level=False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    """
    Padding

    Make sure all the English sequences have the same length and all the Dutch sequences
    have the same length by adding padding to the end of each sequence using Keras’s
    pad_sequences function.
    """
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):
    """
    Pre-process Pipeline

    Implement a pre-process function
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = pad(preprocess_y)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Ids Back to Text

    The neural network will be translating the input to words ids, 
    which isn’t the final form we want. We want the Dutch translation. 
    The function logits_to_text will bridge the gab between the logits 
    from the neural network to the Dutch translation. 
    We will use this function to better understand the output of the neural network.
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def simple_model(input_shape, output_sequence_length, first_vocab_size, second_vocab_size):
    """
    Model 1: RNN

    We are creating a basic RNN model which is a good baseline for sequence data
    that translate English to Dutch.
    """
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(second_vocab_size))(rnn)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model


def embed_model(input_shape, output_sequence_length, first_vocab_size, second_vocab_size):
    """
    Model 2: Embedding

    An embedding is a vector representation of the word that is close to similar
    words in n-dimensional space, where the n represents the size of the embedding
    vectors. We will create a RNN model using embedding.
    """
    learning_rate = 1e-3
    rnn = GRU(64, return_sequences=True, activation="tanh")

    embedding = Embedding(first_vocab_size, 64, input_length=input_shape[1])
    logits = TimeDistributed(Dense(second_vocab_size, activation="softmax"))

    model = Sequential()
    # em can only be used in first layer --> Keras Documentation
    model.add(embedding)
    model.add(rnn)
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model


def bd_model(input_shape, output_sequence_length, vocab_size):
    """
    Model 3: Bidirectional RNNs
    """
    learning_rate = 1e-3
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.1),
                            input_shape=input_shape[1:]))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def encdec_model(input_shape, output_sequence_length, vocab_size):
    """
    Model 4: Encoder-Decoder

    The encoder creates a matrix representation of the sentence. 
    The decoder takes this matrix as input and predicts the translation as output.
    """
    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape[1:], return_sequences=False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def model_final(input_shape, output_sequence_length, first_vocab_size, second_vocab_size):
    """
    Model 5: Custom

    Create a model_final that incorporates embedding and a bidirectional RNN into one model.

    At this stage, we need to do some experiments such as changing GPU parameter to 256, 
    changing learning rate to 0.005, training our model for more (or less than) 20 epochs etc.
    """
    model = Sequential()
    model.add(Embedding(input_dim=first_vocab_size,
                        output_dim=128, input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(second_vocab_size, activation='softmax')))
    learning_rate = 0.005

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model


def final_predictions(sentence, x, y, x_tk, y_tk):
    """
    Gets predictions using the final model

    :param x: Preprocessed First language data
    :param y: Preprocessed Second language data
    :param x_tk: First language tokenizer
    :param y_tk: Second language tokenizer
    """
    # TODO: Train neural network using model_final

    tmp_X = pad(x)
    model = model_final(tmp_X.shape,
                        y.shape[1],
                        len(x_tk.word_index)+1,
                        len(y_tk.word_index)+1)

    model.fit(tmp_X, y,
              batch_size=1024, epochs=17, validation_split=0.2)

    # Final predictions
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))




def load_report(first_lang, second_lang, first_sentences, second_sentences):
    for sample_i in range(2):
        print('{} sentences Line {}:  {}'.format(
            first_lang, sample_i + 1, first_sentences[sample_i]))
        print('{} sentences Line {}:  {}'.format(
            second_lang, sample_i + 1, second_sentences[sample_i]))

    first_words_counter = Counter(
        [word for sentence in first_sentences for word in sentence.split()])
    second_words_counter = Counter(
        [word for sentence in second_sentences for word in sentence.split()])

    print('{} {} words.'.format(
        first_lang, len([word for sentence in first_sentences for word in sentence.split()])))
    print('{} {} unique words.'.format(first_lang, len(first_words_counter)))
    print('10 Most common words in the {} dataset:'.format(first_lang))
    print(
        '"' + '" "'.join(list(zip(*first_words_counter.most_common(10)))[0]) + '"')

    print()

    print('{} {} words.'.format(
        second_lang, len([word for sentence in second_sentences for word in sentence.split()])))
    print('{} {} unique words.'.format(second_lang, len(second_words_counter)))
    print('10 Most common words in the {} dataset:'.format(second_lang))
    print(
        '"' + '" "'.join(list(zip(*second_words_counter.most_common(10)))[0]) + '"')


def preprocess_report(first_lang, second_lang, max_first_sequence_length, max_second_sequence_length, first_vocab_size, second_vocab_size):
    print('Data Preprocessed')
    print("First sequence max sentence length:", max_first_sequence_length)
    print("Second sequence max sentence length:", max_second_sequence_length)
    print("First vocabulary size:", first_vocab_size)
    print("Second vocabulary size:", second_vocab_size)


def run_simple_rnn(preproc_first_sentences, preproc_second_sentences, max_sequence_length, first_vocab_size, second_vocab_size, tokenizer):
    tmp_x = pad(preproc_first_sentences, max_sequence_length)
    tmp_x = tmp_x.reshape((-1, preproc_second_sentences.shape[-2], 1))
    # Train the neural network
    model = simple_model(
        tmp_x.shape,
        max_sequence_length,
        first_vocab_size,
        second_vocab_size + 1)
    model.fit(tmp_x, preproc_second_sentences, batch_size=1024,
              epochs=10, validation_split=0.2)
    # Print prediction(s)
    print(logits_to_text(model.predict(
        tmp_x[:1])[0], tokenizer))
    return model


def run_embedded_rnn(preproc_first_sentences, preproc_second_sentences, max_sequence_length, first_vocab_size, second_vocab_size, tokenizer):
    tmp_x = pad(preproc_first_sentences, max_sequence_length)
    tmp_x = tmp_x.reshape((-1, preproc_second_sentences.shape[-2]))
    model = embed_model(
        tmp_x.shape,
        max_sequence_length,
        first_vocab_size + 1,
        second_vocab_size + 1)
    model.fit(tmp_x, preproc_second_sentences,
              batch_size=1024, epochs=10, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(model.predict(
        tmp_x[:1])[0], tokenizer))
    return model


def run_bidirectional_model(preproc_first_sentences, preproc_second_sentences, first_tokenizer, second_tokenizer):
    tmp_x = pad(preproc_first_sentences, preproc_second_sentences.shape[1])
    tmp_x = tmp_x.reshape((-1, preproc_second_sentences.shape[-2], 1))
    model = bd_model(
        tmp_x.shape,
        preproc_second_sentences.shape[1],
        len(second_tokenizer.word_index)+1)
    model.fit(tmp_x, preproc_second_sentences,
              batch_size=1024, epochs=20, validation_split=0.2)
    # Print prediction(s)
    print(logits_to_text(model.predict(tmp_x[:1])[0], second_tokenizer))
    return model


def run_encdec_model(preproc_first_sentences, preproc_second_sentences, first_tokenizer, second_tokenizer):
    tmp_x = pad(preproc_first_sentences)
    tmp_x = tmp_x.reshape((-1, preproc_first_sentences.shape[1], 1))
    model = encdec_model(
        tmp_x.shape,
        preproc_second_sentences.shape[1],
        len(second_tokenizer.word_index)+1)

    model.fit(tmp_x, preproc_second_sentences,
              batch_size=1024, epochs=20, validation_split=0.2)
    print(logits_to_text(model.predict(tmp_x[:1])[0], second_tokenizer))
    return model


def main(sentence, first_lang, second_lang):
    first_sentences = load_data(first_lang)
    second_sentences = load_data(second_lang)
    load_report(first_lang, second_lang, first_sentences, second_sentences)

    # Preprocess
    preproc_first_sentences, preproc_second_sentences, first_tokenizer, second_tokenizer =\
        preprocess(first_sentences, second_sentences)

    # Calculate
    max_first_sequence_length = preproc_first_sentences.shape[1]
    max_second_sequence_length = preproc_second_sentences.shape[1]
    first_vocab_size = len(first_tokenizer.word_index)
    second_vocab_size = len(second_tokenizer.word_index)
    preprocess_report(first_lang, second_lang, max_first_sequence_length,
                      max_second_sequence_length, first_vocab_size, second_vocab_size)

    # Run Simple RNN
    # simple_rnn_model = run_simple_rnn(
    #     preproc_first_sentences,
    #     preproc_second_sentences,
    #     max_second_sequence_length,
    #     first_vocab_size,
    #     second_vocab_size,
    #     second_tokenizer)

    # Run Embedded model
    # embedded_model = run_simple_rnn(
    #     preproc_first_sentences,
    #     preproc_second_sentences,
    #     max_second_sequence_length,
    #     first_vocab_size,
    #     second_vocab_size,
    #     second_tokenizer)
    # Run Bidirectional model
    # bidirectional_model = run_bidirectional_model(
    #     preproc_first_sentences,
    #     preproc_second_sentences,
    #     first_tokenizer,
    #     second_tokenizer)

    # Run encoder-decoder
    # encdec_model = run_encdec_model(
    #     preproc_first_sentences,
    #     preproc_second_sentences,
    #     first_tokenizer,
    #     second_tokenizer)
        
    # Run final predictions
    final_predictions(sentence, preproc_first_sentences, preproc_second_sentences, first_tokenizer, second_tokenizer)

if __name__ == "__main__":
    main("jezus christus", 'nl', 'pap')

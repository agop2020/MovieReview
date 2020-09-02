import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000) #only take the most 10,000 most frequently said words
#words represented by integers in training data in list form

word_index = data.get_word_index() #tuples w/integers and the words they correspond to

#+3 due to 3 special character keys added
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 #unknown
word_index["<UNUSED>"] = 3

#swaps keys and values so now we have values first and keys second
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#neural network layer length has to be known and constant, so making them all 250(adding padding or trimming)
#can check len of train_data and test_data beforehand
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)



#returns all of the human-readable words
def decode_review(text):
    return "".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[0]))

#model
'''
model = keras.Sequential()
#16 vectors means ax + by + cz ... up to 16 variables
model.add(keras.layers.Embedding(88000, 16)) #great and good example - embedding layer groups similar words, 16 dimensions for the vector of each word
model.add(keras.layers.GlobalAveragePooling1D()) #averages out the vector values for each word in the input layer, passes to next dense layer, scales down data's dimension
model.add(keras.layers.Dense(16, activation="relu")) #classifies word vectors into a positive or negative review
model.add(keras.layers.Dense(1, activation="sigmoid")) #output neuron (0 or 1, good or bad review), sigmoid puts values between 0 and 1 (20% a bad review, 80% good)

model.summary()

#binary loss function, outputs are between 0 and 1, calculates diff between output neuron answer(0.2) and answer(0)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#validation data - see how model is performing on train_data
#validation data checks accuracy after learning from the train data
x_val = train_data[:10000] #25,000 reviews total
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

#batch_size, how many movie reviews to load in at once
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")
'''

#if the word is known, add it to the encoded review, if not, add an unknown tag
def review_encode(s):
    encoded = [1] #setting starting tag due to above start=1

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("model.h5")

#don't have to close the file afterwards if use 'with'
with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)  # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
'''
test_review = test_data[0]
predict = model.predict([test_review])
print("review: ")
print(decode_review(test_review))
print("prediction: " + str(predict[0]))
print("actual: " + str(test_labels[0]))
#print(results)
'''


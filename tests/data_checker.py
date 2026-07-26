import data_preprossor

data_train, data_test = data_preprossor.load_data()
data_train = data_preprossor.preprocess_data(data_train)

diff = set()

for text in data_train.data:
    text: str
    # print(text.split())
    diff.update(text.split())


words = list(diff)
words.sort()

# NOTE: There a lot of words only appearing once in the training set,
# which will be dropped by classifiers later.

print(f"Total unique words: {len(words)}")
print(words[15000:15100])

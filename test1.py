

train = [
     ('I love this sandwich.', 'pos'),
     ('this is an amazing place!', 'pos'),
     ('I feel very good about these beers.', 'pos'),
     ('this is my best work.', 'pos'),
     ("what an awesome view", 'pos'),
     ('I do not like this restaurant', 'neg'),
     ('I am tired of this stuff.', 'neg'),
     ("I can't deal with this", 'neg'),
     ('he is my sworn enemy!', 'neg'),
     ('my boss is horrible.', 'neg')

]


test = [
     ('the beer was good.', 'pos'),
     ('I do not enjoy my job', 'neg'),
     ("I ain't feeling dandy today.", 'neg'),
     ("I feel amazing!", 'pos'),
     ('Gary is a friend of mine.', 'pos'),
     ("I can't believe I'm doing this.", 'neg')
 ]

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
cl = NaiveBayesClassifier(train)
print(cl.classify("This is an amazing library!"))
prob_dist = cl.prob_classify("This one's a doozy.")
print(prob_dist.max())
print(round(prob_dist.prob("pos"), 2))
print(round(prob_dist.prob("neg"), 2))


blob = TextBlob("The beer is good. But the hangover is horrible.", classifier=cl)
blob.classify()
for s in blob.sentences:
     print(s)
     print(s.classify())
print(cl.accuracy(test))
print(cl.show_informative_features(5))
new_data = [('She is my best friend.', 'pos'),
             ("I'm happy to have a new friend.", 'pos'),
             ("Stay thirsty, my friend.", 'pos'),
             ("He ain't from around here.", 'neg')]
print(cl.update(new_data))
print(cl.accuracy(test))
def end_word_extractor(document):
	tokens = document.split()
	first_word, last_word = tokens[0], tokens[-1]
	feats = {}
	feats["first({0})".format(first_word)] = True
	feats["last({0})".format(last_word)] = False
	return feats
features = end_word_extractor("I feel happy")
assert features == {'last(happy)': False, 'first(I)': True}
cl2 = NaiveBayesClassifier(test, feature_extractor=end_word_extractor)
blob = TextBlob("I'm excited to try my new classifier.", classifier=cl2)
print(blob.classify())

from textblob import TextBlob


with open('train.json', 'r') as fp:     
	cl = NaiveBayesClassifier(fp, format="json")

clasif=cl.classify("This is an amazing library!")
prob_dist.max()
round(prob_dist.prob("pos"), 2)
round(prob_dist.prob("neg"), 2)
blob = TextBlob("The beer is good. But the hangover is horrible.", classifier=cl)
blob.classify()
for s in blob.sentences:
	    print(s)
		print(s.classify())
cl.accuracy(test)
print(cl.show_informative_features(5))
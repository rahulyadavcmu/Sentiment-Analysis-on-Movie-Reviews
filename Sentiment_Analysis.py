#Create and fit the classifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def main():
	#Declare the pipeline and parameters for grid search
	pipeline = Pipeline([('vect', TfidfVectorizer(stop_words='english')), ('clf', LogisticRegression())])
	parameters = {
	'vect__max_df' : (0.25, 0.5),
	'vect__ngram_range' : ((1,1), (1,2)),
	'vect__use_idf' : (True, False),
	'clf__C' : (0.1, 1, 10)
	}
	#Load the data from the csv file using pandas
	df = pd.read_csv('train.tsv', header=0, delimiter='\t')
	Phrases, Sentiments = df['Phrase'], df['Sentiment'].as_matrix()
	Phrases_train, Phrases_test, Sentiments_train, Sentiments_test = train_test_split(Phrases, Sentiments, train_size=0.5)
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1, scoring='accuracy')
	grid_search.fit(Phrases_train, Sentiments_train)
	print 'Best score: %0.3f' %grid_search.best_score_
	print 'Best parameters set:'
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print '\t%s: %r' %(param_name, best_parameters[param_name])
	predictions = grid_search.predict(Phrases_test)
	print 'Accuracy: ', accuracy_score(Sentiments_test, predictions)
	print 'Confusion Matrix: ', confusion_matrix(Sentiments_test, predictions)
	print 'Classification Report: ', classification_report(Sentiments_test, predictions)


if __name__ == '__main__':
	main()
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import nltk
import re
import pickle
import warnings
from sklearn.exceptions import UndefinedMetricWarning
data = pd.read_csv(r"D:\pythonProject\Sentiment_Analysis\Data_set_x - Copy.csv", encoding='utf-8')
df = data.head(10)


#data.drop(index=[231,232,233,234,235,236,237,238,239,240,241,242,243,244,245],inplace=True)
#print(data.tail(20))

print("\n Before Data preprocessing  sample data \n")

sample_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 941,950,1035,1084]
for j in sample_data:
    print(data.Comment[j],data.Tag[j])

def remove_punctuation(Comment):
  pattern = '[a-zA-Z0-9।!?%০-৯,:;()-.]'
  for punctuation in re.finditer(pattern, Comment):
    Comment = Comment.replace(punctuation.group(), '')
  return Comment


data['cleaned'] = data['Comment'].apply(remove_punctuation)

print("\n After  Data preprocessing  sample data \n")

for j in sample_data:
    print(data.cleaned[j],data.Tag[j])

for j in sample_data:
    print(data.cleaned[j],data.Tag[j])
print('\n')
print(data['Emotion'].unique())
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
label_encoder = LabelEncoder()
label_encoder.fit(data['Tag'])
data['label']=label_encoder.transform(data['Tag'])
X = data['cleaned']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=20)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=20)

print("\nDataset Distribution:\n")
print("\tSet Name", "\t\tSize")
print("\t========\t\t======")

print("\tFull\t\t\t",data['Comment'].shape[0],
      "\n\tTraining\t\t", X_train.shape[0],
      "\n\tTest\t\t\t", X_test.shape[0])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

tf = TfidfVectorizer(max_features=5000,ngram_range=(1,2),stop_words=None, lowercase=True)
X_train_tfidf = tf.fit_transform(X_train)
X_test_tfidf = tf.transform(X_test)


dt_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
rf_model = RandomForestClassifier(n_estimators=100, criterion ='entropy', random_state = 0)
mnb_model = MultinomialNB(alpha=0.15)
lsvm_model = SVC(kernel = 'linear',C = 0.2, probability=True, random_state = 0)
knn_model = KNeighborsClassifier(n_neighbors=3, metric = 'minkowski')
lr_model = LogisticRegression(max_iter=1000)


mnb_model .fit(X_train_tfidf, y_train)
y_pred = mnb_model.predict(X_test_tfidf)

mnb_model .fit(X_train_tfidf, y_train)
y_pred1 = mnb_model.predict(X_test_tfidf)

lsvm_model.fit(X_train_tfidf, y_train)
y_pred2 = lsvm_model.predict(X_test_tfidf)

# for Multinomial Nb
accuracy = accuracy_score(y_test, y_pred)*100
Precision = round(precision_score(y_test, y_pred,zero_division=0,average='weighted'), 4) * 100
Recall = round(recall_score(y_test, y_pred,average='weighted'), 4) * 100
F1_Score = round(f1_score(y_test, y_pred,average='micro'), 4) * 100
unique_labels = np.unique(y_test)
confusion = confusion_matrix(y_test, y_pred, labels=unique_labels)

# for random Forest
accuracy1 = accuracy_score(y_test, y_pred1)*100
Precision1 = round(precision_score(y_test, y_pred1,zero_division=0,average='weighted'), 4) * 100
Recall1 = round(recall_score(y_test, y_pred1,average='weighted'), 4) * 100
F1_Score1 = round(f1_score(y_test, y_pred1,average='micro'), 4) * 100
unique_labels1 = np.unique(y_test)
confusion1 = confusion_matrix(y_test, y_pred1, labels=unique_labels)

# for lsvm
accuracy2 = accuracy_score(y_test, y_pred2)*100
Precision2 = round(precision_score(y_test, y_pred2,zero_division=0,average='weighted'), 4) * 100
Recall2 = round(recall_score(y_test, y_pred2,average='weighted'), 4) * 100
F1_Score2 = round(f1_score(y_test, y_pred2,average='micro'), 4) * 100
unique_labels2 = np.unique(y_test)
confusion2 = confusion_matrix(y_test, y_pred2, labels=unique_labels)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=0)

print("\n Multinomial NB\n ")
print(confusion)
print(f"Accuracy: {accuracy:.2f}")
print(f"precision: {Precision:.2f}")
print(f"Recall: {Recall:.2f}")
print(f"F1_Score: {F1_Score:.2f}")
print(report)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(mnb_model, model_file)

with open('model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

new_text = 'দারুন কালেকশ'
cleaned_text = remove_punctuation(new_text)
text_tfidf = tf.transform([cleaned_text])
predicted_label = mnb_model.predict(text_tfidf)
predicted_sentiment = label_encoder.inverse_transform(predicted_label)
print(f"Predicted Sentiment: {predicted_sentiment[0]}")

report1 = classification_report(y_test, y_pred1, labels=unique_labels1, zero_division=0)
print("\n random forest\n ")
print(confusion1)
print(f"Accuracy: {accuracy1:.2f}")
print(f"precision: {Precision1:.2f}")
print(f"Recall: {Recall1:.2f}")
print(f"F1_Score: {F1_Score1:.2f}")
print(report1)

report2 = classification_report(y_test, y_pred2, labels=unique_labels1, zero_division=0)
print("\n Lsvm\n ")
print(confusion2)
print(f"Accuracy: {accuracy2:.2f}")
print(f"precision: {Precision2:.2f}")
print(f"Recall: {Recall2:.2f}")
print(f"F1_Score: {F1_Score2:.2f}")
print(report2)


metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [86.55, 85.89, 86.55, 86.55]
values1 = [86.55, 85.89, 86.55, 86.55]
values2 = [84.35,84.72, 84.35, 84.35]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color='skyblue')
plt.ylim(0, 100)
plt.title("Performance Metrics in Multinomial NB")
plt.xlabel("Performance Metrics")
plt.ylabel("Value")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(metrics, values1, color='skyblue')
plt.ylim(0, 100)
plt.title("Performance Metrics in Random Forest")
plt.xlabel("Performance Metrics")
plt.ylabel("Value")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(metrics, values2, color='skyblue')
plt.ylim(0, 100)
plt.title("Performance Metrics in LSVM")
plt.xlabel("Performance Metrics")
plt.ylabel("Value")
plt.show()

#plt.savefig('Performance Metrics in LSVM')

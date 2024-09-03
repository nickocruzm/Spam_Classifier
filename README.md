# Spam Classifier

Using NLP to create a spam filter

## Dataset 
[SMSSpamCollection](https://doi.org/10.24432/C5CC84.), Almeida,Tiago and Hidalgo,Jos. (2012). SMS Spam Collection. UCI Machine Learning Repository. 

### Dataset Attributes

- (.tsv) Tab Seperated Values
- 5572 Instances
- label column := ham or spam
- 4825 instances are ham
- 747 instances are spam


## Histogram

### Ham or Spam

![Ham or Spam](images/histogram.png)

Data above illustrates:
- Majority of the ham messages have a length between 50 to 100.
- Majority of the spam messages occur closer to the 150 message length.


```python
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
```


```python
%matplotlib inline
```


```python
messages = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])

print("HEAD:\n\t")
print(messages.head())

print("\n DESCRIBE:\n\t")
print(messages.describe())

print("\n GROUPED BY LABEL: \n\t")
print(messages.groupby('label').describe())
```

    HEAD:
    	
      label                                            message
    0   ham  Go until jurong point, crazy.. Available only ...
    1   ham                      Ok lar... Joking wif u oni...
    2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
    3   ham  U dun say so early hor... U c already then say...
    4   ham  Nah I don't think he goes to usf, he lives aro...
    
     DESCRIBE:
    	
           label                 message
    count   5572                    5572
    unique     2                    5169
    top      ham  Sorry, I'll call later
    freq    4825                      30
    
     GROUPED BY LABEL: 
    	
          message                                                               
            count unique                                                top freq
    label                                                                       
    ham      4825   4516                             Sorry, I'll call later   30
    spam      747    653  Please call our customer service representativ...    4



```python
messages['length'] = messages['message'].apply(len)
histogram = messages.hist(column='length', by='label', bins=35,figsize=(12,4))

min_length = messages['length'].min()
max_length = messages['length'].max()

colors = ['skyblue', 'lightgreen']
for ax,color in zip(histogram.flatten(), colors):
    ax.set_xlabel("Message Length")
    ax.set_ylabel("Frequency")
    # ax.set_xlim(min_length, max_length)
    #ax.set_xticks(np.arange(0, max_length, 50))
    for patch in ax.patches:
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    
    histogram[0].set_xticks(np.arange(0, messages['length'].max(), 50))

```


    
![histogram](images/histogram.png)
    



```python
def process_txt(mess):
    """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
    """
    
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
```

`CountVectorizer`: Used to convert a collection of text documents to a matrix of tokens. 
        
        I WANT TO KNOW WHY IT IS CALLED "COUNT"VECTORIZER. WHAT IS BEING COUNTED?

`.fit()`
         method processes the input text, identifies all unique words, and creates a vocabulary dictionary that maps each word to a unique index. This dictionary is then used to transform the text data into a numerical format that machine learning models can work with.

## What does "fit" mean?

In the context of machine learning and data processing, the term "fit" refers to the process of learning or training on the data. Specifically, it means that the model or algorithm is being trained on the input data to understand its structure, patterns, or characteristics.


### Analogy

- Imagine you have a book and you're making a list of all the unique words in the book. When you’re done, you have a list (vocabulary) that you can use to check which words are in the book and how often they appear.



This code processes text data, builds a vocabulary of unique tokens using the defined analyzer 'process_txt', and then prints the total number of tokens and ouputs 10 item pairs in vocabulary.


```python

# bag of words = bow
bow_transformer = CountVectorizer(analyzer=process_txt).fit(messages['message'])

# .vocabulary_ is a dictionary {key: words, val: feature index}
print(f"vocabulary size:  {len(bow_transformer.vocabulary_)} \n\t")

# show the first 10 items in vocabulary_
count = 0
for k,v in bow_transformer.vocabulary_.items():
    if(count == 10): break
    print(f"\t{k} : {v}")
    count += 1

```

    vocabulary size:  11425 
    	
    	Go : 2060
    	jurong : 7555
    	point : 8917
    	crazy : 5769
    	Available : 1110
    	bugis : 5218
    	n : 8336
    	great : 6937
    	world : 11163
    	la : 7668


### Code Below:

Produces a Sparse Matrix.

`(document_index, word_index) freq`





```python
messages_bow = bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
```

    Shape of Sparse Matrix:  (5572, 11425)
    Amount of Non-Zero occurences:  50548



This code is used to understand how much of the matrix is filled with actual data (non-zero values) as opposed to zeros.

## Sparsity 
typically refers to the proportion of zero elements in a matrix. The higher the sparsity, the more zeros the matrix contains.

In this case, the code calculates the percentage of non-zero elements, which is technically the matrix's "density." Therefore, if this value is high, the matrix is less sparse (contains more non-zero values); if it's low, the matrix is more sparse.


```python
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))
```

    sparsity: 0


## Term Frequency - Inverse Document Frequency (TF - IDF)

### Term Frequency (TF):

- This measures how frequently a term appears in a document relative to the total number of terms in that document
    
### Inverse Document Frequency (IDF):
- This measures how important a word is by looking at how many documents contain the word.
- Words that appear in many documents get lower IDF values (common words like "the," "and," etc.), while words that appear in fewer documents get higher IDF values.
    
### TF-IDF:
- The TF-IDF score is the product of TF and IDF. It gives a high score to words that are frequent in a document but not in others, making them more relevant for identifying the content of that document.


```python
tfidf_transformer = TfidfTransformer().fit(messages_bow)
```


`tfidf_tranformer.idf_[index]`: returns the `IDF value` of word found at index in `dictionay: vocabulary_`.

The greater the `IDF value`, the more significat/rare the word.





```python
# finds the index of the token, 'u' in the vocabulary.
index = bow_transformer.vocabulary_['u']
idf_u = tfidf_transformer.idf_[index]

index = bow_transformer.vocabulary_['university']
idf_university = tfidf_transformer.idf_[index]

best = max(idf_u,idf_university)

if(best == idf_u):
    x = 'u'
    y = 'university'
else:
    x = 'university'
    y = 'u'
    
    
print(f"u: {idf_u}, university: {idf_university}")

msg = f'{y} is a more common word found in the documents than {x}.'
print(msg)


```

    u: 3.2800524267409408, university: 8.527076498901426
    u is a more common word found in the documents than university.



```python
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
```

    (5572, 11425)


## Key Components:

1.	MultinomialNB():
- This is an instance of the MultinomialNB class from the sklearn.naive_bayes module.
- MultinomialNB is a type of Naive Bayes classifier that is particularly suited for classification with discrete features, such as word counts or term frequencies in text data.
2.	.fit(messages_tfidf, messages['label']):
- The .fit() method trains the Naive Bayes model on the provided data.
- messages_tfidf: This is the feature matrix where each row represents a message, and each column represents a term’s TF-IDF score in that message.
- messages['label']: This is the target vector, which contains the labels for each message (e.g., “spam” or “ham”). These labels are what the model is trying to predict.


```python
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
```

## What the Code Does:

- Model Initialization: MultinomialNB() initializes a Naive Bayes model.
- Training the Model: .fit(messages_tfidf, messages['label']) trains this model on the data.
- The model learns to associate the TF-IDF features (word importance scores) of each message with its corresponding label (“spam” or “ham”).
- After training, the model can predict whether new messages are spam or ham based on their TF-IDF features.

## Why This is Important:

- Spam Detection: This line of code is crucial in developing a spam detection system. The trained model can be used to automatically classify incoming messages as either spam or not spam based on the patterns it has learned during training.

In summary, this code is creating and training a Naive Bayes classifier to detect spam by learning from a dataset of messages represented by their TF-IDF scores and corresponding labels.


```python
all_predictions = spam_detect_model.predict(messages_tfidf)
print(classification_report(messages['label'], all_predictions))

```

                  precision    recall  f1-score   support
    
             ham       0.98      1.00      0.99      4825
            spam       1.00      0.85      0.92       747
    
        accuracy                           0.98      5572
       macro avg       0.99      0.92      0.95      5572
    weighted avg       0.98      0.98      0.98      5572
    


## Key Components:

### train_test_split:

- This function is imported from sklearn.model_selection.
- It splits arrays or matrices into random train and test subsets. It’s used to divide the dataset into a training set and a testing set.

### messages['message']:

- This is the feature data, which contains the actual text messages.


### messages['label']:

- This is the target data, which contains the labels (e.g., “spam” or “ham”) associated with each message.
- 
### test_size=0.2:

- This parameter specifies the proportion of the dataset to include in the test split.
- test_size=0.2 means 20% of the data will be used for testing, and the remaining 80% will be used for training.
- 
### msg_train, msg_test, label_train, label_test:

- The function returns four sets of data:
- msg_train: The training set of messages (80% of the original messages).
- msg_test: The testing set of messages (20% of the original messages).
- label_train: The training set of labels corresponding to msg_train.
- label_test: The testing set of labels corresponding to msg_test.
- 
### print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test)):

- This prints the lengths of the training and testing sets, followed by the total number of messages (which should be the same as the original dataset size).



```python
msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)
print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
```

    4457 1115 5572


## What the Code Does:

- Splitting the Data: The code splits the dataset into a training set and a testing set, with 80% of the data used for training the model and 20% reserved for evaluating the model’s performance.
- Output: The print statement provides a quick check to ensure that the split was performed correctly by showing the number of items in the training and testing sets and confirming that their sum equals the total number of items in the original dataset.

## Why This is Important:

- Model Evaluation: By splitting the data into training and testing sets, you can train the model on one subset (training) and evaluate its performance on another (testing) that the model hasn’t seen before. This helps to assess how well the model generalizes to new, unseen data.
- Preventing Overfitting: Using a separate test set ensures that the model is not just memorizing the training data but is learning patterns that apply to new data as well.

In summary, this code is setting up the data for training and testing by splitting it into two parts, which is a fundamental step in building a reliable machine-learning model.

# Data Pipeline

## ('tfidf', TfidfTransformer()):

- TfidfTransformer: This converts the integer counts from the previous step into TF-IDF scores. TF-IDF (Term Frequency-Inverse Document Frequency) scales the raw counts so that more common words are down-weighted, while rarer, more informative words are up-weighted.
- 
- Purpose: This step refines the word counts by considering how important each word is in the context of the entire dataset, making the model more sensitive to the significance of words in different messages.

## ('classifier', MultinomialNB())

- Purpose: This step trains a Naive Bayes model on the TF-IDF features generated in the previous step, allowing the model to learn patterns that distinguish different classes (e.g., spam vs. ham).



```python
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_txt)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
```

## What the Code Does:

- End-to-End Process: The pipeline integrates multiple steps into a single object. When you fit the pipeline to your data, it will:
	- 1.	Convert the text data into a bag-of-words representation using CountVectorizer.
	- 2.	Transform those counts into TF-IDF scores using TfidfTransformer.
	- 3.	Train the Naive Bayes classifier on the TF-IDF scores.
- Predicting with New Data: When you pass new data through the pipeline, it will automatically apply all these steps and output predictions.

## Why This is Important:

- Simplifies Workflow: The pipeline abstracts away the complexity of manually applying each transformation and then fitting the model, making the workflow more straightforward and less error-prone.
- Consistency: It ensures that the same sequence of transformations is applied to both the training data and any new data you want to predict, maintaining consistency in the data processing steps.


```python
pipeline.fit(msg_train,label_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;bow&#x27;,
                 CountVectorizer(analyzer=&lt;function process_txt at 0x151045280&gt;)),
                (&#x27;tfidf&#x27;, TfidfTransformer()),
                (&#x27;classifier&#x27;, MultinomialNB())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;bow&#x27;,
                 CountVectorizer(analyzer=&lt;function process_txt at 0x151045280&gt;)),
                (&#x27;tfidf&#x27;, TfidfTransformer()),
                (&#x27;classifier&#x27;, MultinomialNB())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">CountVectorizer</label><div class="sk-toggleable__content"><pre>CountVectorizer(analyzer=&lt;function process_txt at 0x151045280&gt;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">TfidfTransformer</label><div class="sk-toggleable__content"><pre>TfidfTransformer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">MultinomialNB</label><div class="sk-toggleable__content"><pre>MultinomialNB()</pre></div></div></div></div></div></div></div>



## What the Code Does:

### Training the Pipeline:

- The fit method starts by passing the msg_train data through the pipeline:
	- 1.	CountVectorizer: Converts the raw text messages in msg_train into a bag-of-words representation (i.e., a matrix of token counts).
	- 2.	TfidfTransformer: Converts the bag-of-words matrix into a TF-IDF matrix, which scales the word counts by their importance.
	- 3.	MultinomialNB: Trains the Naive Bayes classifier on the TF-IDF matrix using the corresponding labels from label_train.

### Result:

- After this process, the pipeline’s model is trained and ready to make predictions. The model has learned patterns in the training data that it can use to classify new, unseen messages.


```python
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))
```

                  precision    recall  f1-score   support
    
             ham       1.00      0.96      0.98      1009
            spam       0.74      1.00      0.85       106
    
        accuracy                           0.97      1115
       macro avg       0.87      0.98      0.92      1115
    weighted avg       0.98      0.97      0.97      1115
    


## Why This is Important:

- Model Training: This step is crucial because it’s where the model learns from the training data. The quality of this training will determine how accurately the model can classify new messages.

- Automated Workflow: The use of a pipeline ensures that all preprocessing steps (like vectorization and TF-IDF transformation) are consistently applied to the training data, which is essential for building a robust model.


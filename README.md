# Spam Classifier

Using NLP to create a spam filter

<!--## Tools -->

<!-- <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg" style="width: 100px; height=200px;" alt="scikit">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original-wordmark.svg" style="width: 100px; height=200px;" alt=""/> -->


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



`CountVectorizer`: Used to convert a collection of text documents to a matrix of tokens. 
        


`CountVectorizer.fit()`
         method processes the input text, identifies all unique words, and creates a vocabulary dictionary that maps each word to a unique index. This dictionary is then used to transform the text data into a numerical format that machine learning models can work with.

## What does "fit" mean?

In the context of machine learning and data processing, the term "fit" refers to the process of learning or training on the data. Specifically, it means that the model or algorithm is being trained on the input data to understand its structure, patterns, or characteristics.


### Analogy

- Imagine you have a book and you're making a list of all the unique words in the book. When you’re done, you have a list (vocabulary) that you can use to check which words are in the book and how often they appear.



This code processes text data, builds a vocabulary of unique tokens using the defined analyzer 'process_txt', and then prints the total number of tokens and ouputs 10 item pairs in vocabulary.




```sh
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
```



## Sparse Ma

Shape of Sparse Matrix:  (5572, 11425)
Amount of Non-Zero occurences:  50548



This code is used to understand how much of the matrix is filled with actual data (non-zero values) as opposed to zeros.

## Sparsity 
typically refers to the proportion of zero elements in a matrix. The higher the sparsity, the more zeros the matrix contains.

In this case, the code calculates the percentage of non-zero elements, which is technically the matrix's "density." Therefore, if this value is high, the matrix is less sparse (contains more non-zero values); if it's low, the matrix is more sparse.

    sparsity: 0


## Term Frequency - Inverse Document Frequency (TF - IDF)

### Term Frequency (TF)

- This measures how frequently a term appears in a document relative to the total number of terms in that document
    
### Inverse Document Frequency (IDF)
- This measures how important a word is by looking at how many documents contain the word.
- Words that appear in many documents get lower IDF values (common words like "the," "and," etc.), while words that appear in fewer documents get higher IDF values.
    
### TF-IDF
- The TF-IDF score is the product of TF and IDF. It gives a high score to words that are frequent in a document but not in others, making them more relevant for identifying the content of that document.


```python
tfidf_transformer = TfidfTransformer().fit(messages_bow)
```


`tfidf_tranformer.idf_[index]`: returns the `IDF value` of word found at index in `dictionay: vocabulary_`.

The greater the `IDF value`, the more significat/rare the word.






u: 3.2800524267409408, university: 8.527076498901426
u is a more common word found in the documents than university.



## Key Components

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


- Model Initialization: MultinomialNB() initializes a Naive Bayes model.
- Training the Model: .fit(messages_tfidf, messages['label']) trains this model on the data.
- The model learns to associate the TF-IDF features (word importance scores) of each message with its corresponding label (“spam” or “ham”).
- After training, the model can predict whether new messages are spam or ham based on their TF-IDF features.

## Why This is Important

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
    




### train_test_split

- This function is imported from sklearn.model_selection.
- It splits arrays or matrices into random train and test subsets. It’s used to divide the dataset into a training set and a testing set.

### messages['message']

- This is the feature data, which contains the actual text messages.


### messages['label']

- This is the target data, which contains the labels (e.g., “spam” or “ham”) associated with each message.
- 
### test_size=0.2

- This parameter specifies the proportion of the dataset to include in the test split.
- test_size=0.2 means 20% of the data will be used for testing, and the remaining 80% will be used for training.
- 
### msg_train, msg_test, label_train, label_test

- The function returns four sets of data:
- msg_train: The training set of messages (80% of the original messages).
- msg_test: The testing set of messages (20% of the original messages).
- label_train: The training set of labels corresponding to msg_train.
- label_test: The testing set of labels corresponding to msg_test.


### print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

- This prints the lengths of the training and testing sets, followed by the total number of messages (which should be the same as the original dataset size).



```python
msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)
print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
```



- Splitting the Data: The code splits the dataset into a training set and a testing set, with 80% of the data used for training the model and 20% reserved for evaluating the model’s performance.
- Output: The print statement provides a quick check to ensure that the split was performed correctly by showing the number of items in the training and testing sets and confirming that their sum equals the total number of items in the original dataset.


- Model Evaluation: By splitting the data into training and testing sets, you can train the model on one subset (training) and evaluate its performance on another (testing) that the model hasn’t seen before. This helps to assess how well the model generalizes to new, unseen data.
- Preventing Overfitting: Using a separate test set ensures that the model is not just memorizing the training data but is learning patterns that apply to new data as well.

In summary, this code is setting up the data for training and testing by splitting it into two parts, which is a fundamental step in building a reliable machine-learning model.



## ('tfidf', TfidfTransformer())

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

- End-to-End Process: The pipeline integrates multiple steps into a single object. When you fit the pipeline to your data, it will:
	- 1.Convert the text data into a bag-of-words representation using CountVectorizer.
	- 2.Transform those counts into TF-IDF scores using TfidfTransformer.
	- 3.Train the Naive Bayes classifier on the TF-IDF scores.

- Predicting with New Data: When you pass new data through the pipeline, it will automatically apply all these steps and output predictions.

- Simplifies Workflow: The pipeline abstracts away the complexity of manually applying each transformation and then fitting the model, making the workflow more straightforward and less error-prone.
- Consistency: It ensures that the same sequence of transformations is applied to both the training data and any new data you want to predict, maintaining consistency in the data processing steps.


```python
pipeline.fit(msg_train,label_train)
```


### Training the Pipeline

- The fit method starts by passing the msg_train data through the pipeline:
- 1.CountVectorizer: Converts the raw text messages in msg_train into a bag-of-words representation (i.e., a matrix of token counts).
- 2.TfidfTransformer: Converts the bag-of-words matrix into a TF-IDF matrix, which scales the word counts by their importance.
- 3.MultinomialNB: Trains the Naive Bayes classifier on the TF-IDF matrix using the corresponding labels from label_train.

### Result

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
    



- Model Training: This step is crucial because it’s where the model learns from the training data. The quality of this training will determine how accurately the model can classify new messages.

- Automated Workflow: The use of a pipeline ensures that all preprocessing steps (like vectorization and TF-IDF transformation) are consistently applied to the training data, which is essential for building a robust model.


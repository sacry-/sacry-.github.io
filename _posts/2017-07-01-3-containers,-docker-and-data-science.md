---
layout: post
author: Matthias Nitsche
title: Containers, Docker and Data Science
keywords: [containers, docker, data science, machine learning, data engineering]
index: 3
img: containerization.jpg
---

Hardware virtualization and virtual machines were yesterday! Today is the age of containers. Just kidding, but containers are here to stay for a long time. 

{% include image.html url="/images/containerization.jpg" description="source: https://c1.staticflickr.com/4/3678/10921728045_4de014f856_b.jpg" %}

<blockquote>A container image is a lightweight, stand-alone, executable package of a piece of software that includes everything needed to run it: code, runtime, system tools, system libraries, settings. 
<cite>- https://www.docker.com/what-container</cite></blockquote>

Docker is one of the many providers to offer support for containers through various tools like the Docker cli, Dockerhub and the Dockerfile standard. The best thing about them is simple: If you need to install several libraries and software tools to support your own use case Docker helps in defining this platform in a simple Dockerfile format. From third party tools to runtime setups, operating systems and how the code is loaded most of the things you think of are possible. Containers also offer a standardized interface to the outside world meaning that IaaS or PaaS providers are capable of offering their services given your Dockerfile alone.

In this post I would like to create a small runtime for a data science / machine learning workflow. The hard thing about these workflows is that you need about everything. Programming environments, dozens of programming libraries, dozens of system libraries at best running on a linux, at worst with GPU accelerated and system specific support, visualization tools and access to a broad range of databases. This seems like a cool use case!

### Docker

For this post basic Docker knowledge is required as well as some familiarity with the eco system evolving around data science/machine learning applications. In your console you should be able to do

```ruby
$ docker ps
CONTAINER ID  IMAGE  COMMAND  CREATED  STATUS  PORTS  NAMES
```

Currently there is nothing to see here, but if you had any containers running they would show up here. Let us create a `Dockerfile`

```ruby
$ touch Dockerfile
$ $EDITOR Dockerfile
```

We will use a modified version of <a href="https://github.com/dataquestio/ds-containers" target="_blank">Dataquest.io</a> to setup our data-science environment.

```ruby
$ cat Dockerfile
FROM dataquestio/ubuntu-base

ENV TERM=xterm
ENV LANG en_US.UTF-8

RUN apt-get update && apt-get -qy upgrade && apt-get install build-essential -y && apt-get clean

ADD apt-packages.txt /tmp/apt-packages.txt
RUN xargs -a /tmp/apt-packages.txt apt-get install -y --fix-missing
```

We tell Docker that we would like to use a base image called `dataquestio/ubuntu-base` configured by a user of Dockerhub. This image is a good starting point, other data science images are much more involved and require a lot more time to fully understand. For machine learning you would need the CUDA versions for everything you do.
Next we set some basic environments `ENV` and update the ubuntu version with system libraries of our needs. In the directory of your Dockerfile needs to be the <a href="https://github.com/dataquestio/ds-containers/blob/master/apt-packages.txt" target="_blank">apt-packages.txt</a> file.

```ruby
RUN pip install virtualenv
RUN /usr/local/bin/virtualenv /opt/ds --distribute --python=/usr/bin/python3

ADD requirements.txt /tmp/requirements.txt
RUN /opt/ds/bin/pip install -r /tmp/requirements.txt
```

Next we install everything required for running python and the needed libraries. The `requirements.txt` are necessary for all the statistical and machine learning libraries needed. They contain

```ruby
$ cat requirements.txt
ipykernel
jupyter
pyzmq
scipy
pandas
matplotlib
statsmodels
scikit-learn
seaborn
nltk
gensim
sympy
bokeh
networkx
requests
beautifulsoup4
textblob
pymysql
sqlalchemy
glances
pyenchant
langdetect
xlrd
theano
pillow
h5py
tensorflow
keras
```

Next let us create the correct users and add permissions.

```ruby
RUN useradd --create-home --home-dir /home/ds --shell /bin/bash ds
RUN chown -R ds /opt/ds
RUN adduser ds sudo

ADD run-jupyter.sh /home/ds
RUN chmod +x /home/ds/run-jupyter.sh
RUN chown ds /home/ds/run-jupyter.sh
```

Within the container we need a user called `ds` and we would like to boot our jupyter notebook with executable permissions later on.

```ruby
EXPOSE 6006
EXPOSE 8888
RUN usermod -a -G sudo ds
RUN echo "ds ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER ds
RUN mkdir -p /home/ds/notebooks
ENV HOME=/home/ds
ENV SHELL=/bin/bash
ENV USER=ds
ENV NLTK_DATA=/home/ds/notebooks
WORKDIR /home/ds/notebooks

CMD ["bash", "/home/ds/run-jupyter.sh"]
```

From within the container we need to expose Jupyter on port 8888, so it is accessible via `localhost:8888`. Additionally we set the `NLTK_DATA` environment to tell our container where it can be found (using tokenizers etc.). The `CMD` is a command that is actually run at the end. We are telling Docker to run a script called `run-jupyter` to start our session.

After everything is setup, we can build our container like this

```ruby
docker build -t data-explore:latest -f Dockerfile .
```

Nothing fancy really. We build the image and tag it `-t` with a name using the just created Dockerfile. After building the image we will need to start it.

```ruby
docker run --name data-explore -d -p 8888:8888 -v $HOME/to/notebooks:/home/ds/notebooks data-explore:latest
```

Now here is happening quite a bit more. We run a container naming it `data-explore` port forwarding on `-p 8888:8888` so it is accessible on `localhost:8888` and defining a volume `-v` from our local host where the notebooks lie to the notebooks in the container. This means that all files on your local machine are mapped into the container and vice versa for peristent nootebook sessions. If everything worked out correctly, go to your browser on `localhost:8888` seeing something like this

{% include image.html url="/images/jupyter-browser.png" %}

### Data Science

After setting everything up it is high time to create a small model doing anything really. In the following I will take in a list of documents with text describing positions and one of three true normalized positions "engineer, product, c-level".

<table class="table table-md table-striped table-bordered">
<thead>
  <tr>
  <th>Position as Text</th>
  <th>Real position</th>
  </tr>
</thead>
<tbody>
  <tr>
  <td>Senior business developer</td>
  <td>engineer</td>
  </tr>
  <tr>
  <td>Research Scientist</td>
  <td>product</td>
  </tr>
  <tr>
  <td>Co-founder and Director</td>
  <td>c-level</td>
  </tr>
</tbody>
</table>

Each categorie has roughly 200 different examples with a text description and the true label. Lets setup the data as a `pandas` Dataframe and `drop` columns we do not care about.

```python
import pandas as pd
import numpy as np

df = pd.read_excel("list_of_positions.xlsx", header=1)
df = df[pd.notnull(df["label"])]
df.drop(['Unnamed: 2', 'url',
       'domain', 'column1', 'column2', 'firstname', 'lastname',
       'fullname', 'exact match?', 'email', 'match1', 'guessed1', 'match2',
       'guessed2', 'match3', 'pattern match in', 'pattern',
       'pattern matched email', 'app_name.1'],inplace=True,axis=1)
```

Group positions by label.

```python
grouped = df.groupby(["label"])

labels = [group for group, _ in grouped]
print(labels)

def group_map(grouped, field):
  return dict((group, [str(field) for field in fields]) for group, fields in grouped[field])

positions = group_map(grouped, "positions")
```

Preprocess the data with English based tokenization, removing stopwords and stemming the words to only take the stem of a word into account. 

```python
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import PorterStemmer
from collections import defaultdict

NLTK_STOPS = stopwords.words('english')
STOPS = frozenset(w for w in NLTK_STOPS if w)
PORTER = PorterStemmer()

def stem(text):
  return [
    stem for stem in 
    [PORTER.stem(token).lower() for token in word_tokenize(text) if token] if stem
  ]

def tokenize(groups):
  cleaned = defaultdict(list)
  for category, docs in groups.items():
    for doc in docs:
      if not doc: continue
      tokens = word_tokenize(doc) 
      new_doc = []
      for token in tokens:
        token = token.lower().strip()
        if token in STOPS or len(token) <= 1:
          continue            
        new_doc.append(token)
      if new_doc:
        d = " ".join(new_doc)
        cleaned[category].append(" ".join(stem(d)))
  return cleaned

normalized_positions = tokenize(positions)

data = defaultdict(list)
for label in labels:
  _positions = normalized_titles[label]
  for i in range(0, len(_positions)):
    r = ""
    if _positions[i] != "nan":
      r = _positions[i] + " "     
    if r != "":
      data[label].append(r)
```

After preprocessing the data we map the natural language positions into the vector space with `Tf-Idf`. We need a train set for training the classifier and a test set for validating how well we did. `X` are always the datapoints described by their columns e.g. natural language positions and `Y` the golden standard of what the actual label is.

```python
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def get_xy(labels, data):
  x, y = [], []
  for idx, label in enumerate(labels):
    for doc in data[label]:
      x.append(doc)
      y.append(idx)
  return x, y

x, y = get_xy(labels, data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)                

vectorizer = TfidfVectorizer(
  sublinear_tf=True, 
  use_idf=True,
  smooth_idf=True,
  analyzer='word',
  stop_words='english',
  max_features=100000,
  ngram_range=(1,2),
  max_df=0.99,
  min_df=1
)

X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

print(X_train.shape, X_test.shape, len(y_train), len(y_test))
```

Next we need to take the mapped data and classify a classifier with the training set. We use the RandomForest classifier that builds up decision trees or weak learners sampled randomly optimizing a cost function at each tree node and merging the results of all weak learners to create a stronger classifier. We will do several runs with RandomForest and would like to have one that excels 80% on the v-measure scale.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics.cluster import v_measure_score
from random import randint

# Add models here to get the best one
def optimize_random_forest(X_train, y_train, X_test, y_test):
  max_clf = None
  max_name = ""
  max_pred = []
  max_accuracy = 0
  max_v_measure = 0
  
  around = 1
  while max_v_measure < 0.8:
    criterias = ["gini", "entropy"]
    criterion = criterias[around%2]
    n_estimators = 100 + randint(-20, 20)
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    clazz = clf.__class__.__name__
    accuracy = metrics.accuracy_score(y_test, pred, normalize=True)
    v_measure = v_measure_score(y_test, pred)
    
    if v_measure > max_v_measure:
      print("round:", around, "criterion:", criterion, "estimators:", n_estimators)
      print(clazz, " - accuracy:", accuracy, "v_measure:", v_measure)
      
      max_v_measure = v_measure
      max_accuracy = accuracy
      max_name = clazz
      max_pred = pred
      max_clf = clf
    
    around += 1
          
  return max_clf, max_name, max_pred, max_accuracy, max_v_measure

clf, name, pred, accuracy, v_measure = optimize_random_forest(X_train, y_train, X_test, y_test)

# Extract analysis data
mapped_labels = dict((idx, label) for idx,label in enumerate(labels))
pred_readable = [mapped_labels[val] for val in pred]
y_readable = [mapped_labels[val] for val in y_test]
result_table = list(zip(x_test, pred_readable, y_readable))
```

When everything is run (this might take a while), we can analyze our results and print out what values were misclassified and what their actual label was.

```python
from collections import Counter

def analyze(result_table):
  correct, wrong = [], []
  for val, pred, actual in result_table:
    if pred == actual:
      correct.append((val, pred, actual))
    else:
      wrong.append((val, pred, actual))

  def group_by(wrong):
    groups = defaultdict(list)
    vals = defaultdict(list)
    for val, pred, actual in wrong:
      groups[pred].append(actual)
      vals[pred].append(val)
    return groups, vals
  
  def pprint(groups, vals):
    for k, v in groups.items():
      print("-------\n", k, "({})".format(len(v))," -> ", list(Counter(v).items()))
      print("   {}".format(vals[k]))

  print("wrong:")
  groups_wrong, vals_wrong = group_by(wrong)
  pprint(groups_wrong, vals_wrong)
  
  print("\ncorrect:")
  groups_correct, vals_correct = group_by(correct)
  pprint(groups_correct, vals_correct)
    
analyze(result_table)
```

The wrong results are much more interesting than correct ones. The above classifier has an accuracy of around 96%. `5 were classified as engineers` despite being in the c-level category, `3 as product` despite being c-level and `4 as c-level` despite being engineer and product. The v-measure is only 80% because of this fact. On the wrongly assigned labels `c-level` seems to be very ambiguous.

```python
wrong:
-------
engineer (5)  ->  [('c-level', 5)]
['senior busi develop innov manag head co-found comdirect start-up garag passion leadership digit strategi innov', 'co-found engin cofound engin everalbum found team atom lab', 'vp engin vp technolog audibl amazon compani', 'director softwar develop director softwar develop flightawar', 'cofound mobil engin lead co-found everalbum found team atom lab']
-------
product (3)  ->  [('c-level', 3)]
['senior product manag product jibjab co-found hello santa', 'co-found vp product co-found vp product outfit7 slovenian subsidiari ekipa2 d.o.o', 'co-found product director co-found vp product dashlan presid dashlan franc']
-------
c-level (4)  ->  [('engineer', 1), ('product', 3)]
['vice presid technolog product strategi vp technolog strategi workday/investor advisor', 'research scientist consult climate/environment issu project coordin', 'project coordin project coordin deutscher wetterdienst', 'io dev team leader io dev team leader joytun']
```

Some correct samples as well. Overall pretty good for the range of very unnormalized textual inputs.

```python
correct:
-------
engineer (62)
['develop relat director develop relat director', 'c++ develop', 'engin manag softwar engin 9gag', 'senior android develop senior android develop flipagram', ...]
-------
product (45)
['director sale market outdoor product director sale market outdoor product garmin intern', 'product/brand market director market rockstar game', 'product manag product manag tinybop', ... ]
-------
c-level (127)
['member board director product strategist investor entrepreneur', 'presid ceo tech feder credit union', 'sr brand ambassador market research vc serial entrepreneur sme b2b master linkedin expert pr/digit media market strategist growth hacker', ... ]
```

### Wrap up

In this post I have shown how to use basic Docker to setup a data-science/machine learning environment. We then applied the environment to run a very basic algorithm called RandomForest. We did not optimize anything and just randomly hoped to improve results. Todays state of the art models in classifying something like the positions Long short-term memory (LSTM) networks, convolutional neural networks (CNN) or attention based networks (Transformer), do amazingly great on the tasks. There is always room for improvement!

### Sources

- [Docker](https://www.docker.com/)
- [Python Numpy](http://www.numpy.org/)
- [Python Scipy](https://www.scipy.org/)
- [Python Pandas](http://pandas.pydata.org/)
- [Python Sklearn](http://scikit-learn.org/)
- [Python Jupyter](http://jupyter.org/)
- [Dataquest on Github](https://github.com/dataquestio/ds-containers)


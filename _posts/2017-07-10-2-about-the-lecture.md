---
layout: post
author: Matthias Nitsche
title: About the Lecture
keywords: [tti, lecture summary, cloud computing]
index: 2
img: cloud-computing.jpg
---

The overall topic in the TTI lecture can be summarized with the two words "cloud computing". Create distributed architectures, deploy and provision services horizontally and scale them as needed. In the following I will summarize the lecture and make remarks to the BPaaS project/assigment of the team I worked in. In the following I will skim through concepts that were most interesting to me.

{% include image.html url="/images/cloud-computing.jpg" description="source: https://c1.staticflickr.com/8/7404/16389741442_45374e44a1_b.jpg" %}

### Cloud Computing

<blockquote>Cloud computing is a model for enabling convenient, on-demand network access to a shared pool of configurable computing resources (e.g., networks, servers, storage, applications, and services) that can be rapidly provisioned and released with minimal management effort or service provider interaction. <cite>- NIST</cite></blockquote>

For me the most striking point of cloud computing is that by definition the setup and usability should be "convenient", "rapid", with "minimal management effort" working on "shared resources" that should be replaceable e.g. "configurable" and "on-demand". Services like Heroku are prime examples of what this means. For deploying a service with Heroku and Github you only need to do
    
    cd app
    heroku login # authenticate here
    git init
    git add .
    git commit -m "initial commit"
    heroku create
    git push heroku master

If your use case is simple (e.g. small application on a few workers) you are done. Congratulations! Depending on your requirements and framework you are using e.g. Ruby on Rails, Django or Node.js you need to provide a way to start the service. If the Heroku defaults are insufficient, tools like Docker help in making your services deployable. This is achieved by specific file formats like the <mark>Dockerfile</mark> configuring exactly what is needed to run your service. Heroku has gained a lot of experience within the last years and knows how the defaults of a lot of different frameworks look a like.

<div class="panel panel-default">
  <div class="panel-heading"><b>BPaaS project</b></div>
  <div class="panel-body">
  For the project we used a barebone server with Docker installed. This is sufficient for some cases but not for all. In retrospec we should have standardized this early to see how it behaves and works in production. Solutions like Kubernetes would have been a good choice instead of e.g. Heroku or barebone servers to deploy services and workers. The major problem was that we did not assign anybody on this particular task untill th semester was over.
  </div>
</div>

### Virtualization

The concept <b>virtualization</b> basically means that one can easily abstract away different layers of a computer. Most commonly we speak about <b>hardware virtualization</b> by creating virtual software that runs on different operating systems. The advantage is that the virtualization software can be used on any platform and with changing hardware requirements emulating other operating systems with their dedicated software tools. For IaaS providers this means that they can use a dedicated set of barebone servers with their default configuration and operating system using VMs to make the best use of their resources.

{% include image.html url="/images/docker-architecture.svg" description="source: https://docs.docker.com/engine/article-img/architecture.svg" class="small" %}

With emerging tools like <b>Docker</b>, Containers are a new way to think about virtualization. <b>Containers</b> are immutable structures with an operating system and a handful of commands/software that can be run within it. All dependecies can be easily installed trough standardized file formats like the Dockerfile. Special registries are used such as <b>Dockerhub</b> to create images of software components or operating systems that can be reused by other users. To run a linux container with a [Redis](https://redis.io/) database running on port `6379` simply do

    docker run --name my-redis -d redis redis-server -p 6379

The advantage? You did not have to configure anything on your localhost machine. It simply "works".

<div class="panel panel-default">
  <div class="panel-heading"><b>BPaaS project</b></div>
  <div class="panel-body">
  For the project we used docker to configure the application. Besides a Dockerfile nothing more was required. The orchestration could be achieved through networks referencing from one to another container.
  </div>
</div>

### IaaS and BPaaS

<b>Infrastructure as a Service</b> boiled down means: I write software and have no clue about infrastracture, please give me a tool that abstracts away all the ceveats of hardware, permission, certificate etc. handling and just scale (horizontally) my application. This, of course, is exaggerated. It raises a point though: Do we care about how our software is deployed as software developers? Is there enough talent on the job market that can manage all these requirements without an IaaS? My claim is that you need both, the talent to handle infrastructure and an IaaS to manage the deployment of your software. Autoscaling (more hardware e.g. servers for more requests), health checks (e.g. is our software still running?), vital metrics for insights and 24/7 availability with an average response time of 20ms (very exaggerated)! All these goals can be achieved by a small error margin with tools like Amazon AWS, Google GCE or Microsoft Azure. No need to reinvent the wheel, it is extremely complicated - ask [Leslie Lamport](https://en.wikipedia.org/wiki/Leslie_Lamport)!

<div class="panel panel-default">
  <div class="panel-heading"><b>BPaaS project</b></div>
  <div class="panel-body">
    While an IaaS covers infrastructure there is a new trend to offer anything as a service. I would actually call it AaaS (Anything as a Service). <b>Business Processing as a Service</b> is occupied with delivering business use cases as a service. Upload your xmls or click through a BPMN editor and create services that directly run on the BPaaS platform handling individual uses cases for the company. Moreover it should be possible to offer such services to other users on the platform as well and offer them to different types of users, such as managers or analysts.

    {% include image.html url="/images/bpmn-wiki.png" description="source: https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/BPMN-1.svg/2000px-BPMN-1.svg.png" class="small" %}

    The goal of our small team was to build such a service platform. We managed to handle the basic business logic through <a href="https://www.activiti.org/" target="_blank">Activiti</a> with their Java API adapters provided through Swagger, but failed to deliver the prototype that creates costum services. Part of that failed requirement was how we understood the idea behind BPaaS. Me, as a software developer mostly interested in data engineering/machine learning/data science and solid software developement skills I could not care less about such an idea. In a stark contrast I currently work at a company that offer translation services to software companies world wide, we could just name it "Translations as a Service". Our mantra is to construct powerful and beautiful code that a.) gets done in a small amount of time and b.) gets done what it should. On the way it should be scalable and maintainable. I could not be further away to the idea of BPMN. My knowledge about the whole realm is small.
  </div>
</div>

### REST and SaaS

As a web developer Web Services and <b>RESTful design</b> are at the heart of my daily work. I would summarize REST with a contract to stateless actions where the context must be fully passed as parameters and standardized action names that are well known and behaved. As such web application typically offer CRUD (create, read, update, delete) interfaces via http verbs such as `get/post/put/patch/delete`. The actions are called with a variety of parameters (typically resources are identified by a `:resource_id`) that determine the kind of action rendering responses with standardized status codes and data formats (e.g. JSON, html, text).

<table class="table table-md table-striped table-bordered">
<thead>
<tr>
<th>CRUD</th>
<th>HTTP Verb</th>
<th>Path</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Read</td>
<td>GET</td>
<td>/users</td>
<td>display a list of all users</td>
</tr>
<tr>
<td>Read</td>
<td>GET</td>
<td>/users/new</td>
<td>return a HTML form for creating a new user</td>
</tr>
<tr>
<td>Create</td>
<td>POST</td>
<td>/users</td>
<td>create a new user</td>
</tr>
<tr>
<td>Read</td>
<td>GET</td>
<td>/users/:id</td>
<td>display a specific user</td>
</tr>
<tr>
<td>Read</td>
<td>GET</td>
<td>/users/:id/edit</td>
<td>return a HTML form for editing a user</td>
</tr>
<tr>
<td>Update</td>
<td>PATCH/PUT</td>
<td>/users/:id</td>
<td>update a specific user</td>
</tr>
<tr>
<td>Delete</td>
<td>DELETE</td>
<td>/users/:id</td>
<td>delete a specific user</td>
</tr>
</tbody>
</table>

<b>Software as a Service</b> is a business where problems are matched with software solutions and customers can integrate named solutions to match their specific use case. In case of translation services, a software company has x localization files and needs to translate their software into many languages with high quality and in no time. The SaaS product then needs to decide how to offer the service to a customer e.g. what interfaces are needed for their usecases (API, Web platform, Binary client/Desktop application) or how resources are shared (multiple databases, applications) etc.

<div class="panel panel-default">
  <div class="panel-heading"><b>BPaaS project</b></div>
  <div class="panel-body">
  In our project we build a REST service for our frontend, that was written in `Node.js with Grunt`. The REST interface was well designed via typical http verbs and responded with JSON documents. The backend itself was implemented in Java with Spring Boot.
  </div>
</div>

### Hadoop and MapReduce

<b>Apache Hadoop</b> is a distributed storage database that works with the MapReduce paradigma using the Hadoop Distributed File System (HDFS). It is well known for scaling and handling billions of rows in a small amount of time, making it ideal for large amount of data "Big data".

The <b>MapReduce</b> paradigma is mainly a model for parallelization of tasks on different services. The operation at hand must always be sequential and is mapped from a domain A to a domain B via some finite processing steps (mapper, filter..). 

{% include image.html url="/images/map-reduce.png" description="source: TTI Foliensatz 5" class="small" %}

<div class="panel panel-info">
<div class="panel-heading"><b>MapReduce in Haskell</b></div>
<div class="panel-body">
<p>I am a huge fan of <a href="https://wiki.haskell.org/Learn_Haskell_in_10_minutes" target="_blank">Haskell</a>.</p>

<p><b>A mapper</b> takes some sequence and applies a transformation function to output some new sequence (with a different or the same type)</p>

<pre><code>map: (a -> b) -> [a] -> [b]
# signature: (Int -> Int) -> [Int] -> [Int]
map (+ 1) [1,2,3] # [2,3,4]
</code></pre>

<p>
<b>A filter</b> pertains the type but eventually drops some elements of the sequence by some condition</p>

<pre><code>filter: (a -> Bool) -> [a] -> [a]
# signature: (Int -> Bool) -> [Int] -> [Int]
filter (> 2) [1,2,3] # [3]
</code></pre>

<b>A reducer</b> takes a sequence, an accumulator and a transformation function to produce the type of the accumulator (which can be a sequence again)

<pre><code>foldr: (a -> b -> b) -> b -> [a] -> b
# signature: (Int -> Int -> Int) -> Int -> [Int] -> Int
foldr (\a,b -> a + b) 0 [1,2,3] # 6
</code></pre>

Of course in reality you will need to figure out given N servers or nodes how to distribute the data and compute some function(s) x. The reducer then gets all the sequential results of all computation servers and joins them into a final result (this can happen via multiple hierarchical reducers as well). As long as all functions in the filter and mapper are sequential e.g. do not depend on a computation of all datapoints at once it can be mapped/filtered and reduced. As such nested data is not suitable.
</div>
</div>

### NoSQL

<b>NoSQL</b> is a paradoxical word. It refers to the aspect that databases in this realm are not relational databases. SQL itself however is a Structured Query Language which can be well defined on NoSQL databases as well. It happens to be the standard for relational databases such as MySQL or Postgresql. 

{% include image.html url="/images/nosql-general.gif" description="source: https://upload.wikimedia.org/wikipedia/commons/c/c1/Nosql.gif" class="small" %}

The following attributes often hold true for NoSQL databases

  - Non relational
  - Distributed and horizontally scalable
  - No transactions (BASE instead of ACID)

<a href="http://www.cs.berkeley.edu/~brewer/cs262b-2004/PODC-keynote.pdf" target="_blank">Brewer</a> had some good ideas what a NoSQL system means and established the <b>BASE</b> principle:

  - Basic Availability
  - Soft-state
  - Eventual consistency

in contrast <b>ACID</b>:

  - Atomic: transaction is sucessful or completely rolled back.
  - Consistent: Database cannot be in an inconsistent state.
  - Isolated: A transaction is indepedant of another.
  - Durable: Translationcts always happen if scheduled (also in case of server restarts etc.)

It is easy to see that NoSQL without any transaction is in a stark contrast to a relational ACID model where everything is wrapped around transactions. In BASE systems we do not care that everything is in sync 100% of the time. It is more convenient to be almost always available and having a state that is not fully consistent at the moment but maybe later e.g. it might happen that 2 users see 2 different truths at times. We gain a lot of speed and conveniences with BASE, however the software that is in such an environment must handle the inconsistent states. 

<div class="panel panel-info">
  <div class="panel-heading"><b>Brewers CAP theorem</b></div>
  <div class="panel-body">
  <b>Brewers CAP</b> theorem states that it is impossible for a NoSQL database to guarantee all of the following attributes at the same time: Consistency, Availability and Partition tolerance. This is explained by the fact that in distributed systems, network failures must be expected.

  {% include image.html url="/images/brewer-cap.png" class="small" %}

  As a result either availability or constistency will suffer. This is also depicted in the above image. There are trade offs to the other either choosing <code>CP, AP or CA</code>.
  </div>
</div>

Typical NoSQL Databases you have to know of

<table class="table table-md table-striped table-bordered">
<thead>
<tr>
<th>Names</th>
<th>Kind</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td>Elasticsearch, MongoDB</td>
<td>Document stores</td>
<td>Extremely good at persisting textual, nested and JSON content. Document text search/retrieval is very fast.</td>
</tr>
<tr>
<td>Redis, memcached, Dynamo</td>
<td>Key-Value stores</td>
<td>O(log(n)) retrieval operation on keys for their values. Often in RAM databases and extraordinary fast. Very simple structures and hard to query on text unlike document stores.</td>
</tr>
<tr>
<td>HBase, Cassandra</td>
<td>Column stores</td>
<td>Can work on billions of rows easily.</td>
</tr>
<tr>
<td>Neo4J, InfiniteGraph</td>
<td>Graph databases</td>
<td>Persists records as vertices and relations between them as weights on edges. Complex graph algorithms and queries can be run on it.</td>
</tr>
</tbody>
</table>

### Web Search - Information Retrieval

When a user searches for content in natural language or with a boolean query we commonly speak of web search and information retrieval. Information retrieval is a discipline of organizing documents in a way that can be queried and ranked by some scoring function. The relevance of how well a retrieval matches can be evaluated by precision and recall or more generally measures of entropy (e.g. more information means more disortion in the data). 

{% include image.html url="/images/precision-recall.png" description="source: TTI Foliensatz 9" class="small" %}

Some companies like Google or WolframAlpha are exceptionally good at interpreting the query a user sends to them. It does not matter if the query is natural language or a specialized query. This is no small feat and is closely connected to huge linguistic databases and algorithms. As of today most web search companies will make use of deep learning algorithms that map a query to possible retrievals via some non linear learned weights that go into the billions. Nobody has a clue why it works so well. It just does.

<div class="panel panel-info">
  <div class="panel-heading"><b>Journey into vector space</b></div>
  <div class="panel-body">
  Typically a document is mapped into vector space or more commonly into term-document matrices and inversed-term-document matrices. With this, simple algorithms such as <b>Tf-Idf</b> can be used for relevance scoring.

  {% include image.html url="/images/vsm.png" class="small" %}

  They are said to be <b>i.i.d.</b> (independent and identically distributed) and the <b>order of words does not matter</b>, really reducing the search to some kind of word term and word inversed term matching by vector space <b>metrics such as cosine or euclidean</b>. 

  <h4>Advanced concepts</h4>

  The best that can be achieved is a combination of Tf-Idf with <b>Latent Semantic Indexing</b> or <b>Latent Dirichlet Allocation</b> to convert count based word methods into a higher feature space that overlap concepts and assigns words/documents real-valued hidden vectors representing them. Queries are then mapped into the hidden feature space and closeness is evaluated on all word/document vectors. Intrinsically testing these structures often shows some kind of semantic resemblance such as that <mark>train, car and commuting</mark> are close to one another. <b>Word2Vec</b> or <b>Paragraph2Vec</b> are natural extensions coming from neural language models.
  </div>
</div>

While algorithms provide a way to rank the content for the user, behind the scences web search needs to be organized in large hierarchical trees due to the enourmous amount of existing websites and new ones. The process behind organizing this is commonly referred to as indexing. Words and sentences are therefore mapped into numerical domains that pertains the structure. It therefore becomes much faster to resolve incoming queries to a subset of the results. On these results high quality algorithms can be run that would take too much time when applied to the whole dataset.

### Wrap up

The lecture covered a wide variety of topics from business architectures to NoSQL, from information retrieval to big data applications, from cloud solutions to cluster mangement software. Two of the biggest drivers in these realms are virtualization with improving abstractions for deploying and scaling software as well as storage solutions that can handle petabyte of data if configured and setup correctly. New software ideas grow quickly and IaaS, PaaS or SaaS solutions come up everyday now, in turn helping other ideas to come to fruit. With a little money on the side software deployment and continuous delivery solutions can be written in 50 lines of code and almost zero configuration while being much better than solutions 5-10 years ago.

### Sources

- [Heroku Setup](https://devcenter.heroku.com/articles/git){:target="_blank"}
- [Redis](https://redis.io/){:target="_blank"}
- [Activiti](https://www.activiti.org/){:target="_blank"}
- [Haskell](https://wiki.haskell.org/Learn_Haskell_in_10_minutes){:target="_blank"}

### Papers

- Jeffrey Dean and Sanjay Ghemawat - 2004, „MapReduce: Simplified data processing on large clusters“. Proceedings of the 6th Symposium on Operating Systems Design and Implementation (OSDI ’04), S. 137–150.
- Brewer - 2000, „Towards robust distributed systems“. Proceedings of the Annual ACM Symposium on Principles of Distributed Computing http://www.cs.berkeley.edu/~brewer/cs262b-2004/PODC-keynote.pdf

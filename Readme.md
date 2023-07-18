# Intelligent Research Assistant with an Interactive Visualization 
---

<details>
<summary>Table of Contents</summary>
 
1. [Projet Description](#project-description)

>  - [Purpose of the Project](#purpose)
>  - [Main Collars](#main-collar)
>  - [Prerequisites](#prerequisites)

2. Installation

3. Usage

4. Codes Explanations
</details>

### Project Description
---
<p> In this project, we want to build a simple and interactive visualization approach built based on Machine Learning Algorithms that can help researchers explore a general topic, such as Machine Learning in Health Care. Let's discuss the project by a simple example. Assume that a researcher wants to gain some knowledge in a rather new field, but as we know, it is not easy for a beginner to find the best directions to move forward. We generally spend a decent amount of time looking for an appropriate starting point, and then we can start thinking about narrowing the general problem down to more specific topics.</p> 
Typically, we start our journey by reading blogs, papers, books, etc., and it makes sense, but the whole process is time consuming, and it can be a nightmare specially if the topic is very broad, and it has multiple aspects. In these cases, having an adviser, a mentor or an experienced colleague can be tremendously helpful, but this is not always the case! Just note that I am not speaking only about academic research, but I also intend to expand the project to industrial research settings.

Hopefully, I have been able to motivate you by now. In this project, we will try to make life easier by providing a rather intelligent framework that can advise people in choosing an appropriate research direction. However, I must confess that this project is a small yet important step towards our target, and on its own, it will not address the problem completely. 

The main concepts of the project can be enumerated as follows:

- Collect a decent number of papers about the topic
- Get the titles or abstracts and use NLP techniques to clean and transform them to numerical values
- Cluster the titles into some groups
- Try to figure out the topic of each group either automatically or by scrutinizing the papers falling inside each cluster
- Finalize and visualize all clusters so that people can explore, read and familiarize themselves with the topics and their associated papers
#### Purpose
<p style="color:dark gray"> There are two main purposes that we are trying to reach by doing this project. Firstly, we are going to go through the details of how we can deal with the <a style='color:rgb(216,118,0)'>challenges</a> faced when working with a real problem. The challenges are both technical and practical, and I will try to introduce the challenge, discuss it and try one or more solutions. We will see how a seemingly simple yet fruitful Machine Learning procedure can be put in practice and used in a production environment which is typically called $\color{rgb(216,118,0)}\large{productionization}$ or <a style='color:rgb(216,118,0)'>modularization</a></p>
<p style="color:dark gray">
 Moreover, the codes, visualization and the logic of the project can be a starting point for many other similar projects in the field; as a result, it might provide some ideas for those interested in a similar domain.  
</p>

#### Main Collars

To work on this project, we will walk through or scratch the surface of the following subjects: 

1. Regular expressions
2. Basic NLP techniques: stopwords, stemming, tokenization and ngrams
3. More advanced NLP techniques: word-to-vec transforms, sentence transforms, topic detection, and topic label generation
4. K-Means clustering
5. Plotly visualization and Python dash 

#### Prerequisites

In terms of methodologies, you need to be familiar with ML and NLP techniques. For the required packages, everything is included in the [requirements.txt](/requirements.txt) file. 

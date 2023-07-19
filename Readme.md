# Intelligent Research Assistant with an Interactive Visualization 
 <a name="readme-top"></a>
<details>
<summary>Table of Contents</summary>
 
1. [Projet Description](#project-description)

>  - [Purpose of the Project](#purpose)
>  - [Main Collars](#main-collars)
>  - [Prerequisites](#prerequisites)

2. [Installation](#installation)

3. [Structure](#Structure)

4. [Codes Explanations](#codes-explanations)
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
There are two main purposes that we are trying to reach by doing this project. Firstly, we are going to go through the details of how we can deal with the challenges faced when working with a real problem. The challenges are both technical and practical, and I will try to introduce the challenge, discuss it and try one or more solutions. We will see how a seemingly simple yet fruitful Machine Learning procedure can be put in practice and used in a production environment which is typically called $\color{rgb(216,118,0)}\large\textrm{productionization}$ or $\color{rgb(216,118,0)}\large\textrm{modularization}$

Moreover, the codes, visualization and the logic of the project can be a starting point for many other similar projects in the field; as a result, it might provide some ideas for those interested in a similar domain.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>


#### Main Collars

To work on this project, we will walk through or scratch the surface of the following subjects: 

1. Regular expressions
2. Basic NLP techniques: stopwords, stemming, tokenization and ngrams
3. More advanced NLP techniques: word-to-vec transforms, sentence transforms, topic detection, and topic label generation
4. K-Means clustering
5. Plotly visualization and Python dash 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Prerequisites

In terms of methodologies, we will need to be familiar with basic ML and NLP techniques. For the required packages, everything is included in the [requirements.txt](/requirements.txt) file which is laid out as follows:

> - pandas
> - numpy
> - scikit-learn
> - plotly 
> - nltk 
>   - Our basic NLP techniques can be done using nltk
> - fasttext 
>   - you might face some problems when installing this package. Instead, you could install fasttext-wheel
> - fasttext-wheel
>   - fasttext or fastext-wheel will be used to detect the language of the text
> - transformers 
>   - We will use this package for topic label generation which does not work very well in our case
    - Also, we will use it to apply a sentence transformation model which will be useful to compute the semantic distance among the titles
> - torch 
>   - torch is required in case we use Hugging Face

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation
---

1. You need to clone the repository on your local machine:

```sh
git clone https://github.com/yzereh/Data_Analytics.git
```
2. To install the packages, you can either try installing each package separately or install all of them using this command:

```sh							
pip install -r your_path/requirements.txt
```

Replace "your_path" with the path leading to the folder where you keep the requirements.txt file. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Structure
---

The project contains four main modules. 

1. The [support.py](/support.py) module, as the name says, contains the majority of required functions supporting the main object of the project located in the [cluster.py](/cluster.py) module. Moreover, the constant variables and the paths to the essential files can be found in this module. We will go through the details in the [Codes Explanations](#codes-explanations) section.
2. The cluster.py module has the main object that can be used to perform the clustering. In brief, all the concepts pointed out in [Project Description](#project-description) section are executed in this module.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Intelligent Research Assistant with an Interactive Visualization 
 <a name="readme-top"></a>
<details>
<summary>Table of Contents</summary>
 
1. [Project Description](#project-description)

>  - [Purpose of the Project](#purpose)
>  - [Main Collars](#main-collars)
>  - [Prerequisites](#prerequisites)

2. [Installation](#installation)

3. [Structure](#structure)

4. [Detailed Project Explanation](#detailed-project-explanation)

5. [References](#references)
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

### Detailed Project Explanation

---

In this section, we will try to go through the details of all the necessary steps taken to complete the project. I will try to reasonably discuss the options that we have at every step and why we need to select one over the others. 

As you know, when dealing with text data, there are several stages, but everything starts with cleaning. Depending on the context, cleaning can significantly vary. In our case, it consists of removing the signs, removing the numbers, removing the stopwords, and stemming each word. I tried to leave the stopwords removal and stemming optional since we do not need to do these operations in every application. Especially that we will have additional frequent words which do not necessarily help us cluster the titles. 

To initiate the process, the ```process_clean_the_text()``` method is used which can be found in the [cluster.py](/cluster.py) module. Let's break this function into some components and take a deeper look at each one.

- **Load the data into memory**

    > We have a simple function in [support.py](/support.py) module, which is called ```load_data()```, and it is defined as:

```sh
def load_data(path_to_the_data: str = MAIN_DIRECTORY, file_name: str = DATA_FILE_NAME):
	with open(os.path.join(path_to_the_data, file_name), encoding='UTF-8') as titles:
		read_lines_original = titles.readlines()
	return read_lines_original
```
> - $\color{rgb(216,118,0)}\large\textbf{params}$:

 >  >  >  **path_to_the_data**: it is the path to the folder containing the data. 
 >  >  >  it is a string and the default value is the constant variable MAIN_DIRECTORY which is defined in the [support.py](/support.py) module as

```sh
MAIN_DIRECTORY = "path_to_your_main_dir_followed_by_a_slash"
```

 >  >  >  **Note**: Do not forget to add a $\color{rgb(216,118,0)}\normalsize{slash}$ to the end of the path.       
 >  >  >  **file_name**: this is the name of the file containing your data. In this project, it is a text file with each line equivalent to a title [health_care_titles.txt](/health_care_titles.txt).

> - $\color{rgb(216,118,0)}\large\textbf{return}$: the function returns a list whose elements are the paper titles.


- **Non-English Titles**

    > After reading the data and striping each title, we need to make sure that we are keeping only english titles. That is because our purpose is to cluster the papers based on their semantic distance or lexicographical similarity; consequently, all titles should preferably be written in a unique language. To do this, I am going to use the [fasttext language identification](https://fasttext.cc/docs/en/language-identification.html) model. The model was trained using Wikipedia, Tatoeba and SETimes, and it can recognize 176 languages [[1](#references), [2](#references)]. 
  
	> the ```is_the_language_english()``` function takes a string or a sequence of strings and evaluate the language of each text. 

```sh	
def is_the_language_english(documents: Union[str, Sequence[str], map]) -> List[str]:
	try:
		import fasttext
		path_to_lang_model = get_fasttext_lang_detect_path()
		language_identification_model = fasttext.load_model(path_to_lang_model)
	except ModuleNotFoundError:
		raise Exception('The fasttext or fasttext-wheel library must be installed')
	except ValueError:
		raise Exception('Please provide a valid path for fasttext language identification model')
	return [text if language_identification_model.predict(text, 1)[0][0][-2:] == 'en' else None for text in documents]
```

> - $\color{rgb(216,118,0)}\large\textbf{params}$:

 >  >  >  **documents**: a string, a map or a sequence of strings. In our case, it is the list of titles. 

> - $\color{rgb(216,118,0)}\large\textbf{return}$: A list with the same length as that of the documents. If the title is in English, it will be kept; otherwise, it will be replaced by None. 

 >  >  >  **Note 1**: the get_fasttext_lang_detect_path() function is added to the [support.py](/support.py) module to make the model loading more flexible to potential changes. 
 
 >  >  >  **Note 2**: I left this step as an optional argument which can be selected by the analyst. The ```drop_non_english``` argument in the ```process_clean_the_text()``` method is a Boolean variable, and the default value is True.

  
```sh	 
def get_fasttext_lang_detect_path(main_directory: str = MAIN_DIRECTORY, model_name: str =  FASTTEXT_LANGUAGE_DETECT_NAME) -> str:
	return os.path.join(main_directory, model_name) 
```

>  >  >  The model_name parameter is set to ```FASTTEXT_LANGUAGE_DETECT_NAME = 'lid.176.bin'``` which is the language identification model, and it can be downloaded from [here](https://fasttext.cc/docs/en/language-identification.html). 


- **Clean the Titles**

    > Cleaning consists of three simple steps:

     > - Change to uppercase
     > - Remove the signs
     > - Remove the numbers
  
```sh
read_lines = list(map(lambda each_line: each_line.upper() if each_line else None, read_lines))
``` 

```sh
read_lines = list(map(lambda each_line: re.sub(r"[^\w\s]", "", each_line) if each_line else None, read_lines))
```
 
```sh
read_lines = list(map(lambda each_line: " ".join(re.findall("[A-Za-z]+", each_line)) if each_line else None, read_lines))
```

- **Stemming**

    > This is another optional step which I found useful in this application. As you know, it is a very standard operation used in mant text analytics applications. An example would clarify the situation to a large extent. Consider the words prediction, predictive, predict, predicting and predictory. Does it make sense to assume that each of these words can be the subject of a new cluster? In some cases, the answer is yes. But in this case, we tend to cluster the papers based on more generic topics. For instance, one cluster might be text mining and the other can be disease treatments, etc. So in our application, it might make better sense to stem all these words and replace them by $\color{rgb(216,118,0)}\large\textbf{predict}$. There are several stemming algorithms, but I chose to use the [Snowball Stemmer](https://snowballstem.org/). 

```sh
if stem_the_words:
    print('Stemming the words.')
    stemmer = SnowballStemmer("english")
    read_lines = \
        list(map(lambda each_line: " ".join(stemmer.stem(each_word).upper()
                                            for each_word in each_line.split()) if each_line else None, read_lines))
```


- **Stopwords and Frequent words Removal**

    > The standard English stopwards can be simply removed and they exist in almost all standrad NLP packages. However, based on my experience working with text and depending on the context, I typically find a group of frequently used words which does not serve the purpose of the project. Let's take a deeper look at our case. If you go throug our papers, you will find words, such as machine, learning, data, mining, hospital, patient, records, settings, software, app, healthcare, etc., are used very frequently. 
    
	> If we leave these words, they will probably dominate the topics of some clusters. But, does it make sense to have a cluster with the topic "machine learning" and another with the topic "patients"? Most likely, I would say no, because these are the most general and the initial search topics. Recall that our initial search was "Machine Learning in Healthcare". As a result, I would say that we can usually find these words depending on the application and get rid of them. I buuilt a JSON document [additional_frequent_words.json](/additional_frequent_words.json) where you can load and take a look the frequent words that I found. Offcourse, you can add or remove words based on your purpose and application.  


### References

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models



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

To facilitate the potential usage of the project in the future, we tried to build the ClusterTitles class with two major methods inside the [cluster.py](/cluster.py) module. The first method which performs the cleaning and preprocessing is called  ```process_clean_the_text()```, and the second one is ```cluster_the_titles()``` method, and its function is to apply ```process_clean_the_text()``` method on the loaded data, transform the cleaned data to vectors, and cluster the titles. <a href="#figure1">Figure 1</a> lays out the structure of the project and the interrelations of modules, classes, methods and functions.   

<img src="/programming_structure.png" name="figure1" title="Figure 1">
<p align="center"><a>Figure 1. The programming structure of the project</a></p>

I will try to break down ```process_clean_the_text()``` method into its building blocks and take a deeper look at each one.

- **Load the data into memory**

    > We have a simple function in [support.py](/support.py) module, which is called ```load_data()```, and it is defined as:

```sh
def load_data(path_to_the_data: str = MAIN_DIRECTORY, file_name: str = DATA_FILE_NAME):
	with open(os.path.join(path_to_the_data, file_name), encoding='UTF-8') as titles:
		read_lines_original = titles.readlines()
	return read_lines_original
```
> - $\color{rgb(216,118,0)}\large\textrm{params}$:

 >  >  >  **path_to_the_data**: it is the path to the folder containing the data. 
 >  >  >  it is a string and the default value is the constant variable MAIN_DIRECTORY which is defined in the [support.py](/support.py) module as

```sh
MAIN_DIRECTORY = "path_to_your_main_dir_followed_by_a_slash"
```

 >  >  >  **Note**: Do not forget to add a $\color{rgb(216,118,0)}\normalsize{slash}$ to the end of the path.       
 >  >  >  **file_name**: this is the name of the file containing your data. In this project, it is a text file with each line equivalent to a title [health_care_titles.txt](/health_care_titles.txt).

> - $\color{rgb(216,118,0)}\large\textbf{return}$: the function returns a list whose elements are the paper titles.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

> - $\color{rgb(216,118,0)}\large\textrm{params}$:

 >  >  >  **documents**: a string, a map or a sequence of strings. In our case, it is the list of titles. 

> - $\color{rgb(216,118,0)}\large\textbf{return}$: A list with the same length as that of the documents. If the title is in English, it will be kept; otherwise, it will be replaced by ```None```. 

 >  >  >  **Note 1**: the get_fasttext_lang_detect_path() function is added to the [support.py](/support.py) module to make the model loading more flexible to potential changes. 
 
 >  >  >  **Note 2**: I left this step as an optional argument which can be selected by the analyst. The ```drop_non_english``` argument in the ```process_clean_the_text()``` method is a Boolean variable, and the default value is True.

  
```sh	 
def get_fasttext_lang_detect_path(main_directory: str = MAIN_DIRECTORY, model_name: str =  FASTTEXT_LANGUAGE_DETECT_NAME) -> str:
	return os.path.join(main_directory, model_name) 
```

>  >  >  The model_name parameter is set to ```FASTTEXT_LANGUAGE_DETECT_NAME = 'lid.176.bin'``` which is the language identification model, and it can be downloaded from [here](https://fasttext.cc/docs/en/language-identification.html). 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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
<p align="right">(<a href="#readme-top">back to top</a>)</p>

- **Stemming**

    > This is another optional step which I found useful in this application. As you know, it is a very standard operation used in mant text analytics applications. An example would clarify the situation to a large extent. Consider the words prediction, predictive, predict, predicting and predictory. Does it make sense to assume that each of these words can be the subject of a new cluster? In some cases, the answer is yes. But in this case, we tend to cluster the papers based on more generic topics. For instance, one cluster might be text mining and the other can be disease treatments, etc. So in our application, it might make better sense to stem all these words and replace them by $\color{rgb(216,118,0)}\large\textbf{predict}$. There are several stemming algorithms, but I chose to use the [Snowball Stemmer](https://snowballstem.org/). 

```sh
if self.stem_the_words:
    print('Stemming the words.')
    stemmer = SnowballStemmer("english")
    read_lines = \
        list(map(lambda each_line: " ".join(stemmer.stem(each_word).upper()
                                            for each_word in each_line.split()) if each_line else None, read_lines))
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

- **Stopwords and Frequent words Removal**

    > The standard English stopwards can be simply removed and they exist in almost all standrad NLP packages. However, based on my experience working with text data and depending on the context, I typically find a group of frequently-used words which do not serve the purpose of the project. Let's take a deeper look at our case. If you go throug our papers, you will find words, such as machine, learning, data, mining, hospital, patient, records, settings, software, app, healthcare, etc., thar are used very frequently. 
    
	> If we leave these words, they will probably dominate the topics of some clusters. But, does it make sense to have a cluster with the topic "machine learning" and another with the topic "patients"? Most likely, I would say no, because these are the most general and the initial search topics. Recall that our initial search was "Machine Learning in Healthcare". As a result, I would say that we can usually find these words depending on the application and get rid of them. I built a JSON document [additional_frequent_words.json](/additional_frequent_words.json) where you can take a look at the frequent words that I found. Offcourse, you can add or remove some words based on your purpose and the application that you are dealing with. The file contains four categories and their associated frequent words. The categories are: **health**, **analytics**, **general**, and **information technology**. One can choose "all", one or a subset of categories to remove their corresponding words from the text. 
	
	> At this point, you might think why we do not use the TF-IDF vectorizer instead of spending some time finding these frequent words brining no insight to our clustering. I agree with you! We can definitely use this vectorizer and skip the process of finding frequent words, but I have two arguments against this method for our application. 
	
	> 1. What if we want to use another type of word-to-vec transforms, like Embeddings?
	> 2. I used TF-IDF, and I found that manually finding and removing frequent words yield better results, for our documents are small, and we are looking for more accurate clusters. 
	
	> Note that both these arguments might be completely wrong for another application, and you might find TF-IDF very effective in your work especially if you are working with very large documents.
 
 > **Suggestion**: to keep track of all the methods, functions and their relations, take a look at <a href="#figure1">Figure 1</a> every once and a while. 
   
```sh
if self.remove_stop_words:
    stop_words = get_stop_words(self.add_extra_frequent_words, self.remove_stop_word_signs,
                                self.extra_frequent_words_categories, self.stem_the_words)
    read_lines = list(map(
        lambda each_line: " ".join(each_word for each_word in each_line.split()
                                   if each_word
                                   not in stop_words) if each_line else None, read_lines))
self.titles_stopwords_removed = read_lines
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

> > > Again, the ```remove_stop_words``` is a Boolean variable and can be set by the analyst. Note that the major finction is ```get_stop_words()``` which can be found in the [support.py](/support.py) module. It takes some information from the user and returns the list of stopwprds to be removed from the text. 

```sh
def get_stop_words(add_extra_frequent_words: bool, extra_frequent_words_categories, remove_stop_word_signs: bool, stem_the_words):
    extra_frequent_words = None
    if add_extra_frequent_words:
        path_to_extra_frequent_words = get_extra_frequent_words_path()
        extra_frequent_words = get_frequent_words(path_to_extra_frequent_words,
                                                  extra_frequent_words_categories)
    stop_words = download_load_stop_words(remove_stop_word_signs, extra_frequent_words)
    stop_words = list(map(lambda x: x.upper(), stop_words))
    if stem_the_words:
        stemmer = SnowballStemmer("english")
        stop_words = list(map(lambda x: stemmer.stem(x).upper(), stop_words))
    return stop_words
```

> - $\color{rgb(216,118,0)}\large\textrm{params}$:

 > > > The first two arguments of this function are fed to another function called ```get_frequent_words()``` which gets the path and the categories of the frequent words and returns the list of frequent words associated with the given categories.

```sh
def get_frequent_words(path_to_additional_frequent_words: str,
                       categories: Union[str, Sequence[str]] = 'all') -> List:
    frequent_words_by_category = json_load(path_to_additional_frequent_words)
    list_frequent_words = []
    if categories == 'all':
        all_frequent_words = reduce(lambda x, y: x + y, frequent_words_by_category.values())
        return all_frequent_words
    else:
        if isinstance(categories, Sequence):
            for each_category in categories:
                if each_category in frequent_words_by_category.keys():
                    list_frequent_words.extend(frequent_words_by_category.get(each_category))
                else:
                    raise Exception(f"Couldn't find the key. Please choose from one of these categories: "
                                    f"{FREQUENT_WORDS_CATEGORIES}")
            return list_frequent_words
        else:
            if categories in frequent_words_by_category.keys():
                return frequent_words_by_category.get(categories)
            else:
                raise Exception(f"Couldn't find the key. Please choose from one of these categories: "
                                f"{FREQUENT_WORDS_CATEGORIES}")
```

 > - $\color{rgb(216,118,0)}\large\textrm{params}$:

  >  >  >  **path_to_additional_frequent_words**: a string showing the path to [additional_frequent_words.json](/additional_frequent_words.json) file, and it is obtained using the function 
  ```get_extra_frequent_words_path()```.

  >  >  >  **categories**: a string or a sequence of strings giving the categories of the additional frequent words to be removed from the text. The default is "all" meaning that the frequent words for all categories 
  will be removed. If someone wants to remove the frequent words associated with the "health" and "general" categories only, this can be specified as ```categories = ["health", "general"]```. 

>  >  > The third argument of the function ```get_stop_words()``` (```remove_stop_word_signs```) is given to the ```download_load_stop_words``` function which downloads and extends the stopwords and removes the signs 
from the stopwords if needed. The final output would be the list of stopwords and the optional frequent words.
 
```sh
def download_load_stop_words(remove_signs: bool = True, extra_frequent_words: List[str] = None) -> List:
    try:
        stop_words = stopwords.words("english")
    except LookupError:
        import nltk
        nltk.download("stopwords")
        stop_words = stopwords.words("english")

    if extra_frequent_words:
        stop_words.extend(extra_frequent_words)

    if remove_signs:
        stop_words = list(map(lambda x: re.sub(r"[^\w\s]", "", x), stop_words))

    return stop_words
```

 > - $\color{rgb(216,118,0)}\large\textrm{params}$:

 >  >  >  **remove_stop_word_signs**: a Boolean specifying whether or not to remove the signs from stopwords.
 
 >  >  >  **extra_frequent_words**: the list of frequent words to be removed. It is defined within the ```get_stop_words()``` function as:

```sh
extra_frequent_words = get_frequent_words(path_to_extra_frequent_words, extra_frequent_words_categories)
```

>  >  > The last argument of the ```get_stop_words()``` function is **stem_the_words** which is a Boolean variable determining whether or not the stopwords must be stemmed. 


>  >  > **Note**: When removng the stopwords, some of the titles might be removed completely. In this case, we will have an empty string ```''``` which will be replaced by ```None``` at the last step. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

- **Duplicates**

	> Since our data were originally scraped from the google scholar website, chances are high that it contains some duplicates. In this step, we would like to find the duplicates, keep the first one and replace the rest by ```None```. At the end, we will deal with these ```None```, but for now, we would like to just spot them. To this end, we added the ```find_duplicates()``` function to the support.py module. The function takes the collection of titles (a sequence or map) as its input and returns a tuple ```tuple[dict[str, list[int]], int]``` containing two elements. The first element is a dictionary of duplicates with the duplicated titles as keys and their associated location in the original data as values. The second element is an integer showing the number of duplicates in the data. 

```sh
def find_duplicates(collection_of_titles: Union[Sequence[str], map]) -> tuple[dict[str, list[int]], int]:
    index = -1
    duplicates_count = 0
    collection_of_titles_by_indices = {}
    duplicated_with_indices = {}
    for each_title in collection_of_titles:
        index += 1
        if collection_of_titles_by_indices.get(each_title, 'not_here') != 'not_here':
            collection_of_titles_by_indices[each_title].append(index)
            if each_title is not None and each_title != '':
                duplicates_count += 1
                try:
                    duplicated_with_indices[each_title].append(index)
                except KeyError:
                    duplicated_with_indices[each_title] = [collection_of_titles.index(each_title)]
                    duplicated_with_indices[each_title].append(index)
        else:
            collection_of_titles_by_indices[each_title] = [index]
    return duplicated_with_indices, duplicates_count
```

 > - $\color{rgb(216,118,0)}\large\textrm{params}$:

 >  >  >  **collection_of_titles**: the sequence of cleaned titles. 
 
 > - $\color{rgb(216,118,0)}\large\textrm{return}$: a tuple including a dictionary of duplicates with their locations and the duplicates count
 
> > Next, the output of the function is used to locate all the duplicates and replace them with ```None```.

```sh
self.duplicated_with_indices, self.duplicates_count = find_duplicates(read_lines)
if self.duplicates_count > 0:
    print('Finding the duplicates.')
    logging.warning(f'There are {self.duplicates_count} duplicates in the titles. '
                    f'All duplicates will be replaced by None.')
    for each_list_of_indices in self.duplicated_with_indices.values():
        for each_index in each_list_of_indices[1:]:
            read_lines[each_index] = None
```  

> > In the last step in the ```process_clean_the_text()``` method; as pointed out in the previous section, the empty string ```''``` will be replaced by ```None```. 

```sh
read_lines = list(map(lambda each_line: None if each_line == '' and each_line is not None else each_line, read_lines))
```

$\color{rgb(216,118,0)}\large\textrm{return}$: ```process_clean_the_text()``` method returns a dictionary with two keys: 1. the collection of original titles that are not even touched. These are going to be used to retrieve the original titles for the final visualization, and 2. The cleaned titles that can be used in the subsequent steps.

```sh
print('##### Done with cleaning and preprocessing #####')
return {support.CLEANED_TITLES_NAME: read_lines, support.ORIGINAL_TITLES_NAME: self.read_lines_original}
```    

Now, we are ready to move to the next major method in the ```ClusterTitles``` class, the ```cluster_the_titles()``` method. Similar to the previous method, we need to look at each important step within this method to have a better grasp of it. 

- **Get the Original and Cleaned Titles**

>> First, we get the cleaned and the initial, untoached titles from the ```process_clean_the_text()``` method. 

```sh
self.dictionary_of_processed_titles = self.process_clean_the_text()
cleaned_data, original_data = self.dictionary_of_processed_titles[support.CLEANED_TITLES_NAME], self.dictionary_of_processed_titles[support.ORIGINAL_TITLES_NAME]
```

>> Next, we find the location of ```Nones``` and remove them from both the original and cleaned titles. To find the location of ```Nones```, we use the ```find_none_indices()``` function:

```sh
def find_none_indices(sequence_of_values: Sequence):
    if not sequence_of_values:
        raise Exception('Please provide a valid sequence')
    index = -1
    none_indices = []
    for each_element in sequence_of_values:
        index += 1
        if each_element is None:
            none_indices.append(index)
    return none_indices
```

>> This function simply gets a sequence of values and returns the list of indexes showing the locations of ```Nones``` in the input sequence.

```sh
self.indices_of_nones = find_none_indices(cleaned_data)
self.original_data_without_nones = [original_data[none_index] for none_index in range(len(original_data))
                                            if none_index not in self.indices_of_nones]
self.cleaned_data_without_nones = [each_element for each_element in cleaned_data if each_element is not None]
```
- **Word/Sentence-to-Vector Transformation and Embeddings**

>> The subsequent step would be to transform the words or the sentences to numeric vectors. The word_to_vec transformations are abundant and the literature is very rich. Even the sentence transformations which transform a sentence to a vector directly are numerous now. It is a nice opportunity to have a brief introduction to some of the transformations which I have personally taken the most advantage of:

>> **Note**: this section can be skipped if you are not interested in the details of word to vectors transformations. 

>>> - the simple **count vectorizer** which counts the frequency of each word in every title and builds a matrix with each row representing a title and every word giving a column. The term-document matrix would be sparse since the titles are short in our case, and chances are high that the words do not appear in many titles. You can find the Python documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

>>> - TF-IDF vecorizer which is similar to CountVectorizer, but it also downplays the importance of the words that are very frequent, and it tries to capture the the technical jargon of a sprcific context in a corpus. See Python [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) for further information.  

>>> - Pre-trained GLoVe [\[3\]](#references) is a vector representation method which transforms each word to a 25d, 50d, 100d, 200d or 300d vector with d standing for dimensional here. The idea behind GloVe is very interesting, and I suggest you to read their [paper](#https://nlp.stanford.edu/pubs/glove.pdf). 

>>>>> First, a word on word co-occurrence matrix is constructed using a large corpus. What is a co-occurrence matrix? It is a matrix whose rows and columns are represented by words. In the literature, the rows are simply called words, and the columns are called contexts, but generally, we can say that words and contexts are interchangeable. For instance, if "surgery" is a word and "health" is the context, then we can switch the roles and assume that "surgery" is the context and "health" is the word.

>>>>> What are the entries of this matrix? the entries are usually the number of times a word appears in a context. So in our example, we are looking for the number of times the word "surgery" appears in the "health" context. Now that we know what a co-occurrence matrix is, we can go through more the details of the GLoVe method. Based on this matrix, the co-occurrence probability matrix is built. The probabilities are the conditional probability that a word occurs given a specific context. Let's say $\mathbb p(surgery|health)$ gives the probability that "surgery" occurs in the "health" context. Further assume that $\mathbb p(surgery|politics)$ is the probability that the word "surgery" appears in the "politics" context. Which one is supposed to be bigger? Since we are talking about "surgery", it is more likely to observe it in the "health" context rathar than the "politics" context. In other words, the $\frac{\mathbb p(surgery|health)}{\mathbb p(surgery|politics)}$ ratio must be a large number, and a ratio, such as $\frac{\mathbb p(filibuster|health)}{\mathbb p(filibuster|politics)}$ must be a small one since it is more probable to see "filibuster" in the "politics" context.

>>>>> Consequently, it makes a perfect sense to look at this ratio a distinctive feature that can tell us about the semantic similarity of two words. Now, how can we relate this to some vectors? Here is the beauty of what Pennington et al. (2014) are suggesting. I will try to briefly talk about the main idea. We need to find some $d$ dimensional vectors representing the words and contexts by incorporating the aforementioned probability ratios which can be also presented in a more generic format as $\frac{\mathbb p(Word_i|Context_k)}{\mathbb p(Word_j|Context_k)}$.

>>>>> Let's assume that the $w_i$, $w_j$ and $w_k$ give the $d$ dimensional vectors representing $Word_i$, $Word_j$ and $Context_k$, respectively. Our purpose would be to estimate $w_i$, $w_j$ and $w_k$ considering the aforementioned probability ratio. To this end, in the paper, they suggest Eq.(1) relating the global vectors to the probability ratios:

<p align="center">$\mathbb F(w_i, w_j, w_k) = \frac{\mathbb p(Word_i|Context_k)}{\mathbb p(Word_j|Context_k)}$ $\mathbb (1)$</p>

>>>>> They; afterwards, make some assumptions about the function $\mathbb F$ to simplify the estimation. After appying all these assumptions, the relationship in Eq. (1) simplifies to:

<p align="center">$\mathbb w_i^Tw_k + b_i + b_k = log(x_ik)$ $\mathbb (2)$</p>

>>>>> where, $b_i$ and $b_k$ are some bias terms to be estimated, and they appear here to restore the symmetry of the co-oocurence matrix (the interchangeability of words and contexts), and $x_ik$ is the number of times the word $i$ appears in context $k$. Finally, the problem can be formulated as a weighted least squares problem with an objectibe function defined as:

<p align="center">$\mathbb \sum_{i,j=1}^{V} \omega(ij)(log(x_ij) - w_i^Tw_j - b_i - b_j)^2$ $\mathbb (3)$</p>

>>>>> where, V is the number of words in the corpus, and $\omega(ij)$ is a weight function assigned to every word-context pair to avoid the dominance of very frequent and infrequent words.

>>>>> The final solution is obtained by minimizing the objective function in Eq. (3). 

>>> - Pre-trained Fastext model developed by Mikolov has some major differences with  et al. (2017) [\[4\]](#references) the pervious word representations enumerated below:
>>>>> - Instead of finding merely the word representations, it finds the chraracter n-gram representations. What does this mean? Let's take a look at a simple example. Consider the word "healthcare". The list of trigrams can be given as:

<p align="center">$(hea, eal, alt, lth, thc, hca, car, are)$</p>

>>>>>>> plus the n-gram of the word itself

<p align="center">$(healthcare)$</p>

>>>>>>> so, instead of having a $d$ dimensional vector $w_i$ for $Word_i$, we will have a set of $d$ dimensional vectors $\nu_j; j = 1, 2, ..., N$, where $N$ is the number of character n-grams including the word itself. Then, the final word representation would be given as:

<p align="center">$\nu_w + \frac{\sum{j=1, j \neq w}^{N}\nu_j}{N}$</p>
 
>>>>>>> What is the advantage of this? it has two advantages. First, we can find the words representations for out-of the vocabulary words. Second, the words that are infrequent, will not be underrepresented.

>>>>> - The position of the words is considered and incorporated as some vectors weighting the word representations. I will not go through the details, but you can find the original paper [here](#https://proceedings.neurips.cc/paper/2013/file/db2b4182156b2f1f817860ac9f409ad7-Paper.pdf). 
>>>>
 > - $\color{rgb(216,118,0)}\large\textrm{params}$:
 >  >  >  **tokenizer_pattern**: in case we want to tokenize the text, what pattern we need to use. The default is ```support.TOKEN_PATTERN = r"\b[^\d\W]+\b"```.
 >  >  >  **transform_method**: which method we want to use to transform the words (sentences) to vectors. I tried several embedding and classical methods and decided to stick to these two methods:
 
 >  >  > 1. the basic CountVectorizer() method w
 
 >  >  > 2. 
   
### References

[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models

[3] J. Pennington, R. Socher, and C. D. Manning. 2014. GloVe: Global Vectors for Word Representation. ]



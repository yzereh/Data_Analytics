This document briefly discusses some of the frequently used word to vector transformations and word representations
> - **count vectorizer**: it counts the frequency of each word in every title and builds a matrix with each row representing a title and every word giving a column. The term-document matrix would be sparse since the titles are short in our case, and chances are high that the words do not appear in many titles. You can find the Python documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).

> - **TF-IDF vecorizer**: it is similar to CountVectorizer, but it also downplays the importance of the words that are very frequent, and it tries to capture the the technical jargon of a sprcific context in a corpus. See Python [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) for further information.  

> - **Pre-trained GLoVe** [\[1\]](#references): it is a vector representation method which transforms each word to a 25d, 50d, 100d, 200d or 300d vector with d standing for dimensional here. The idea behind GloVe is very interesting, and I suggest you to read their [paper](#https://nlp.stanford.edu/pubs/glove.pdf). 

>>> First, a word on word co-occurrence matrix is constructed using a large corpus. What is a co-occurrence matrix? It is a matrix whose rows and columns are represented by words. In the literature, the rows are simply called words, and the columns are called contexts, but generally, we can say that words and contexts are interchangeable. For instance, if "surgery" is a word and "health" is the context, then we can switch the roles and assume that "surgery" is the context and "health" is the word.

>>> What are the entries of this matrix? the entries are usually the number of times a word appears in a context. So in our example, we are looking for the number of times the word "surgery" appears in the "health" context. Now that we know what a co-occurrence matrix is, we can go through more the details of the GLoVe method. Based on this matrix, the co-occurrence probability matrix is built. The probabilities are the conditional probability that a word occurs given a specific context. Let's say $\mathbb p(surgery|health)$ gives the probability that "surgery" occurs in the "health" context. Further assume that $\mathbb p(surgery|politics)$ is the probability that the word "surgery" appears in the "politics" context. Which one is supposed to be bigger? Since we are talking about "surgery", it is more likely to observe it in the "health" context rathar than the "politics" context. In other words, the $\frac{\mathbb p(surgery|health)}{\mathbb p(surgery|politics)}$ ratio must be a large number, and a ratio, such as $\frac{\mathbb p(filibuster|health)}{\mathbb p(filibuster|politics)}$ must be a small one since it is more probable to see "filibuster" in the "politics" context.

>>> Consequently, it makes a perfect sense to look at this ratio a distinctive feature that can tell us about the semantic similarity of two words. Now, how can we relate this to some vectors? Here is the beauty of what Pennington et al. (2014) are suggesting. I will try to briefly talk about the main idea. We need to find some $d$ dimensional vectors representing the words and contexts by incorporating the aforementioned probability ratios which can be also presented in a more generic format as $\frac{\mathbb p(Word_i|Context_k)}{\mathbb p(Word_j|Context_k)}$.

>>> Let's assume that the $w_i$, $w_j$ and $w_k$ give the $d$ dimensional vectors representing $Word_i$, $Word_j$ and $Context_k$, respectively. Our purpose would be to estimate $w_i$, $w_j$ and $w_k$ considering the aforementioned probability ratio. To this end, in the paper, they suggest Eq.(1) relating the global vectors to the probability ratios:

<p align="center">$\mathbb F(w_i, w_j, w_k) = \frac{\mathbb p(Word_i|Context_k)}{\mathbb p(Word_j|Context_k)}$ $\mathbb (1)$</p>

>>> They; afterwards, make some assumptions about the function $\mathbb F$ to simplify the estimation. After appying all these assumptions, the relationship in Eq. (1) simplifies to:

<p align="center">$\mathbb w_i^Tw_k + b_i + b_k = log(x_ik)$ $\mathbb (2)$</p>

>>> where, $b_i$ and $b_k$ are some bias terms to be estimated, and they appear here to restore the symmetry of the co-oocurence matrix (the interchangeability of words and contexts), and $x_ik$ is the number of times the word $i$ appears in context $k$. Finally, the problem can be formulated as a weighted least squares problem with an objectibe function defined as:

<p align="center">$\mathbb \sum_{i,j=1}^{V} \omega(ij)(log(x_ij) - w_i^Tw_j - b_i - b_j)^2$ $\mathbb (3)$</p>

>>> where, V is the number of words in the corpus, and $\omega(ij)$ is a weight function assigned to every word-context pair to avoid the dominance of very frequent and infrequent words.

>>> The final solution is obtained by minimizing the objective function in Eq. (3). 

> - **Pre-trained FastText**: this model is developed by Mikolov et al. (2017) [\[2\]](#references), and it has some differences with the pervious word representations enumerated below:
>>> - Instead of finding merely the word representations, it finds the chraracter n-gram representations. What does this mean? Let's take a look at a simple example. Consider the word "healthcare". The list of trigrams can be given as:

<p align="center">$(hea, eal, alt, lth, thc, hca, car, are)$</p>

>>>> plus the n-gram of the word itself

<p align="center">$(healthcare)$</p>

>>>> so, instead of having a $d$ dimensional vector $w_i$ for $Word_i$, we will have a set of $d$ dimensional vectors $\nu_j; j = 1, 2, ..., N$, where $N$ is the number of character n-grams including the word itself. Then, the final word representation would be given as:

<p align="center">$\nu_w + \frac{\sum{j=1, j \neq w}^{N}\nu_j}{N}$</p>
 
>>>> What is the advantage of this? it has two advantages. First, we can find the words representations for out-of the vocabulary words. Second, the words that are infrequent, will not be underrepresented. For more information you could take a look at the original paper by Bojanowski et al. (2017) [\[3\]](#references).

>>> - The position of the words is considered and incorporated as some vectors weighting the word representations. I will not go through the details, but you can find the original paper [here](#https://proceedings.neurips.cc/paper/2013/file/db2b4182156b2f1f817860ac9f409ad7-Paper.pdf) [\[4\]](#references).
>>> - In this model, a simple approach is used to see wether it is worth it to combine some wods together to create phrases. The phrsees, then, are used as a unigram and a word representation vector is obtained. For instance, the bigram "machine learning" is probably more likely to occure compared to the unigrams "machine" and "learning". In this case, these two unigrams would be combined to create a new unigram called "machine_learning" and a word representation is btained for the whole combined unigram. Doing so, will provide a model with richer information and closer to reality. You can find more details in this [paper](#https://arxiv.org/pdf/1310.4546.pdf) by Mikolov et al. (2013) [\[5\]](#references).

**References**

[1] J. Pennington, R. Socher, and C. D. Manning. 2014. GloVe: Global Vectors for Word Representation. 

[2] T. Mikolov, E. Grave, P. Bojanowski, C. Puhrsch, A. Joulin. Advances in Pre-Training Distributed Word Representations

[3] Bojanowski, P., Grave, E., Joulin, A., and Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5:135–146.

[4] Mnih, A. and Kavukcuoglu, K. (2013). Learning word embeddings efficiently with noise-contrastive estimation. In Advances in neural information processing systems, pages 2265–2273.

[5] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119.

import logging
import re
from functools import reduce
from nltk import SnowballStemmer
from support import find_none_indices, transform_word_to_vec, get_stop_words, find_duplicates, load_data, \
    is_the_language_english
from typing import List, Sequence, Dict, Union
from sklearn.cluster import KMeans
import collections
import support


class ClusterTitles:

    def __init__(self, number_of_clusters: int = 50, minimum_cluster_size: int = 2,
                 first_two_common_word_proportions: Sequence[float] = None):
        self.original_titles_by_cluster = None
        self.dictionary_of_processed_titles = None
        self.dictionary_of_titles = None
        self.titles_by_cluster = None
        self.duplicated_with_indices = None
        if first_two_common_word_proportions is None:
            first_two_common_word_proportions = [0.5, 0.2]
        self.first_two_common_word_proportions = first_two_common_word_proportions
        self.original_titles_by_cluster: Dict[int]
        self.titles_by_cluster: Dict[int]
        self.read_lines_original: Sequence[Union[str, None]] = [None]
        self.kmeans: KMeans() = KMeans(n_clusters=1, n_init='auto', random_state=0)
        self.transformed_titles_basic_model: Sequence[Sequence[float]] = [[0.]]
        self.cleaned_data_without_nones: Sequence[str] = ['']
        self.original_data_without_nones: Sequence[str] = ['']
        self.dictionary_of_titles: Dict
        self.indices_of_nones = Sequence[int]
        self.duplicates_count: int = 0
        self.duplicated_with_indices: Dict[str]
        self.titles_stopwords_removed = Sequence[str]
        self.stemmed_and_cleaned_titles = Sequence[str]
        self.cleaned_titles = Sequence[str]
        self.number_of_nones: int = 0
        self.minimum_cluster_size: int = minimum_cluster_size
        self.number_of_clusters: int = number_of_clusters
        self.read_lines_original: Sequence[str]
        self.keep_or_remove_cluster: List[str] = []

    def process_clean_the_text(self, remove_stop_words: bool = True,
                               remove_stop_word_signs: bool = True, add_extra_frequent_words: bool = True,
                               extra_frequent_words_categories: str or Sequence[str] = 'all',
                               stem_the_words: bool = True, drop_non_english=True) -> Dict:
        """
        Parameters:
                    remove_stop_word_signs: whether the signs must be removed from the stopwords or not (True/False).
                    add_extra_frequent_words: whether extra frequently used words must be added to the stopwords
                    (True/False).
                    extra_frequent_words_categories: a single category or a collection of categories to select the
                    most frequently used words from. Default value is 'all' meaning all categories.
                    remove_stop_words: whether stopwords must be removed or not.
                    stem_the_words: whether stemming must be done.
                    drop_non_english: drop if a title is not english.
        return: a dictionary with two keys, 1. cleand_title: the cleaned data, and
        2. original_titles: the initial untouched data

        """
        self.read_lines_original = load_data()
        read_lines = self.read_lines_original.copy()
        read_lines = list(map(lambda each_line: each_line.strip(), read_lines))

        if drop_non_english:
            read_lines = is_the_language_english(read_lines)

        self.number_of_nones = read_lines.count(None)
        if self.number_of_nones > 0:
            logging.warning(f'The provided data contain {self.number_of_nones} '
                            f'non-english titles which will be replaced by Nones.')

        read_lines = \
            list(map(lambda each_line: each_line.upper() if each_line else None, read_lines))
        read_lines = \
            list(map(lambda each_line: re.sub(r"[^\w\s]", "", each_line) if each_line else None,
                     read_lines))
        read_lines = \
            list(map(lambda each_line: " ".join(re.findall("[A-Za-z]+", each_line)) if each_line else None,
                     read_lines))
        self.cleaned_titles = read_lines.copy()

        if stem_the_words:
            stemmer = SnowballStemmer("english")
            read_lines = \
                list(map(lambda each_line: " ".join(stemmer.stem(each_word).upper()
                                                    for each_word in each_line.split()) if each_line else None,
                         read_lines))

        self.stemmed_and_cleaned_titles = read_lines
        if remove_stop_words:
            stop_words = get_stop_words(add_extra_frequent_words, remove_stop_word_signs,
                                        extra_frequent_words_categories, stem_the_words)
            read_lines = list(map(
                lambda each_line: " ".join(each_word for each_word in each_line.split()
                                           if each_word
                                           not in stop_words) if each_line else None, read_lines))

        self.titles_stopwords_removed = read_lines

        if read_lines.count('') == 1:
            logging.warning('After removing the stopwords, one of the titles is removed completely.\n'
                            'This is probably due to the addition of the extra frequent words to the stopwords.\n'
                            'The additional_frequent_words.json file can be edited in case you think some information'
                            'is lost.')

        if read_lines.count('') > 1:
            logging.warning(
                f"After removing the stopwords, {read_lines.count('')} of the titles are removed completely.\n"
                f"This is probably due to the addition of the extra frequent words to the stopwords.\n"
                f"The additional_frequent_words.json file can be edited in case you think some information "
                f"is lost.")

        self.duplicated_with_indices, self.duplicates_count = find_duplicates(read_lines)
        if self.duplicates_count > 0:
            logging.warning(f'There are {self.duplicates_count} duplicates in the titles. '
                            f'All duplicates will be replaced by None.')

            for each_list_of_indices in self.duplicated_with_indices.values():
                for each_index in each_list_of_indices[1:]:
                    read_lines[each_index] = None
        read_lines = list(
            map(lambda each_line: None if each_line == '' and each_line is not None else each_line, read_lines))

        return {support.CLEANED_TITLES_NAME: read_lines, support.ORIGINAL_TITLES_NAME: self.read_lines_original}

    def cluster_the_titles(self, remove_stop_words: bool = True,
                           remove_stop_word_signs: bool = True, add_extra_frequent_words: bool = True,
                           extra_frequent_words_categories: Union[str, Sequence[str]] = 'all',
                           stem_the_words: bool = True, drop_non_english=True, original_data_remove_nones=None):
        self.dictionary_of_processed_titles = self.process_clean_the_text(remove_stop_words, remove_stop_word_signs,
                                                                          add_extra_frequent_words,
                                                                          extra_frequent_words_categories,
                                                                          stem_the_words, drop_non_english)
        cleaned_data, original_data = self.dictionary_of_processed_titles[support.CLEANED_TITLES_NAME], \
            self.dictionary_of_processed_titles[support.ORIGINAL_TITLES_NAME]
        self.indices_of_nones = find_none_indices(cleaned_data)
        self.original_data_without_nones = [original_data[none_index] for none_index in range(len(original_data))
                                            if none_index not in self.indices_of_nones]
        self.cleaned_data_without_nones = [each_element for each_element in cleaned_data if each_element is not None]
        self.transformed_titles_basic_model = transform_word_to_vec(self.cleaned_data_without_nones)
        self.kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0, n_init='auto'). \
            fit(self.transformed_titles_basic_model)

        cluster_labels = self.kmeans.labels_

        self.titles_by_cluster = {}
        self.original_titles_by_cluster = {}
        for cluster in range(max(cluster_labels)):
            cluster_size = sum(cluster_labels == cluster)
            titles_indices_within_the_cluster = \
                [(self.cleaned_data_without_nones[i], i) for i in range(len(self.cleaned_data_without_nones)) if
                 cluster_labels[i] == cluster]
            titles_within_the_cluster, indices_within_the_cluster = \
                list(map(lambda title_index_pair: title_index_pair[0], titles_indices_within_the_cluster)), \
                map(lambda title_index_pair: title_index_pair[1], titles_indices_within_the_cluster)
            combined_titles_within_the_cluster = reduce(
                lambda previous_title, next_title: next_title + ' ' + previous_title,
                titles_within_the_cluster)
            self.titles_by_cluster[cluster] = titles_within_the_cluster
            counts_by_tokens = collections.Counter(combined_titles_within_the_cluster.split())
            ten_most_common_tokens = counts_by_tokens.most_common(10)
            if cluster_size >= self.minimum_cluster_size and ten_most_common_tokens[0][1] / cluster_size > \
                    self.first_two_common_word_proportions[0] and \
                    ten_most_common_tokens[0][1] / cluster_size > self.first_two_common_word_proportions[1]:
                self.keep_or_remove_cluster.append('keep')
            else:
                self.keep_or_remove_cluster.append('remove')
            self.oiginal_titles_by_cluster[cluster] = [self.original_data_without_nones[i] for i in indices_within_the_cluster]

import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
from typing import List, Dict, Tuple, Sequence
import logging
import json
from functools import reduce
import os
from itertools import groupby

MAIN_DIRECTORY = "E:/Data_Playground/Analytics/"
DATA_FILE_NAME = "health_care_titles.txt"
ADDITIONAL_FREQUENT_WORDS_FILE_NAME = 'additional_frequent_words.json'
FREQUENT_WORDS_CATEGORIES = ['health', 'general', 'analytics', 'information technology']
FASTTEXT_LANGUAGE_DETECT_NAME = 'lid.176.bin'


def load_data():
    df = pd.read_csv(os.path.join(MAIN_DIRECTORY, DATA_FILE_NAME), encoding='UTF-8')
    return df


def get_extra_frequent_words_path() -> str:
    return os.path.join(MAIN_DIRECTORY, ADDITIONAL_FREQUENT_WORDS_FILE_NAME)


def get_fasttext_lang_detect_path() -> str:
    return os.path.join(MAIN_DIRECTORY, FASTTEXT_LANGUAGE_DETECT_NAME)


def is_the_language_english(documents: Sequence[str]) -> List[str]:
    try:
        import fasttext
        path_to_lang_model = get_fasttext_lang_detect_path()
        language_identification_model = fasttext.load_model(path_to_lang_model)
    except ModuleNotFoundError:
        raise Exception('The fasttext or fasttext-wheel library must be installed')
    except ValueError:
        raise Exception('Please provide a valid path for fasttext language identification model')
    return [text if language_identification_model.predict(text, 1)[0][0][-2:] == 'en' else None for text in documents]


def get_stop_words(add_extra_frequent_words: bool, remove_stop_word_signs: bool,
                   extra_frequent_words_categories, stem_the_words):
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


def find_duplicates(collection_of_titles: Sequence[str]) -> (Dict, int):
    """
    param collection_of_titles: a non-empty collection of titles
    return: a dictionary of duplicated titles as keys and their associated indices in the collection as values
    """
    counter = -1
    duplicates_count = 0
    collection_of_titles_by_indices = {}
    duplicated_with_indices = {}
    for each_title in collection_of_titles:
        counter += 1
        if collection_of_titles_by_indices.get(each_title, 'not_here') != 'not_here':
            collection_of_titles_by_indices[each_title].append(counter)
            if each_title is not None and each_title != '':
                duplicates_count += 1
                try:
                    duplicated_with_indices[each_title].append(counter)
                except KeyError:
                    duplicated_with_indices[each_title] = [collection_of_titles.index(each_title)]
                    duplicated_with_indices[each_title].append(counter)
        else:
            collection_of_titles_by_indices[each_title] = [counter]
    return duplicated_with_indices, duplicates_count


def process_clean_the_text(remove_stop_words: bool = True,
                           remove_stop_word_signs: bool = True, add_extra_frequent_words: bool = True,
                           extra_frequent_words_categories: str or Sequence[str] = 'all',
                           stem_the_words: bool = True, drop_non_english=True) -> Sequence:
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
    return: a sequence of cleaned titles

    """

    with open(os.path.join(MAIN_DIRECTORY, DATA_FILE_NAME), encoding='UTF-8') as titles:
        read_lines_original = titles.readlines()

    read_lines = read_lines_original.copy()
    read_lines = map(lambda each_line: each_line.strip(), read_lines)

    if drop_non_english:
        read_lines = list(is_the_language_english(read_lines))

    number_of_nones = read_lines.count(None)
    if number_of_nones > 0:
        logging.warning(f'The provided data contain {number_of_nones} '
                        f'non-english titles which will be removed.')

    read_lines = list(map(lambda each_line: each_line.upper() if each_line else None, read_lines))
    read_lines = map(lambda each_line: re.sub(r"[^\w\s]", "", each_line) if each_line else None, read_lines)
    read_lines = \
        map(lambda each_line: " ".join(re.findall("[A-Za-z]+", each_line)) if each_line else None, read_lines)

    if stem_the_words:
        stemmer = SnowballStemmer("english")
        read_lines = map(lambda each_line: " ".join(stemmer.stem(each_word).upper()
                                                    for each_word in each_line.split()) if each_line else None,
                         read_lines)

    if remove_stop_words:
        stop_words = get_stop_words(add_extra_frequent_words, remove_stop_word_signs,
                                    extra_frequent_words_categories, stem_the_words)
        read_lines = map(lambda each_line: " ".join(each_word for each_word in each_line.split()
                                                    if each_word
                                                    not in stop_words) if each_line else None, read_lines)
    read_lines = list(read_lines)

    if read_lines.count('') == 1:
        logging.warning('After removing the stopwords, one of the titles is removed completely.\n'
                        'This is probably due to the addition of the extra frequent words to the stopwords.\n'
                        'The additional_frequent_words.json file can be edited in case you think some information'
                        'is lost.')

    if read_lines.count('') > 1:
        logging.warning(f"After removing the stopwords, {read_lines.count('')} of the titles are removed completely.\n"
                        f"This is probably due to the addition of the extra frequent words to the stopwords.\n"
                        f"The additional_frequent_words.json file can be edited in case you think some information "
                        f"is lost.")

    duplicated_with_indices, duplicates_count = find_duplicates(read_lines)
    if duplicates_count > 0:
        logging.warning(f'There are {duplicates_count} duplicates in the titles. '
                        f'All duplicates will be replaced by None.')

        for each_list_of_indices in duplicated_with_indices.values():
            for each_index in each_list_of_indices[1:]:
                read_lines[each_index] = None

    return read_lines


def download_load_stop_words(remove_signs: bool = True, extra_frequent_words: List[str] = None) -> List:
    """
    param remove_signs: whether to remove signs from stop words or not
    param add_external_stopwords: list of additional stop words based on the context
    return: list of english stop words
    """

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


def json_load(path: str) -> Dict:
    """
    Loads a json file into a dictionary
    """
    with open(path, "r") as file:
        dict_of_data = json.load(file)
    return dict_of_data


def get_frequent_words(path_to_additional_frequent_words: str,
                       categories: str or List[str] or Tuple[str] = 'all') -> List:
    frequent_words_by_category = json_load(path_to_additional_frequent_words)
    list_frequent_words = []
    if categories == 'all':
        all_frequent_words = reduce(lambda x, y: x + y, frequent_words_by_category.values())
        return all_frequent_words
    else:
        if isinstance(categories, List) | isinstance(categories, Tuple):
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

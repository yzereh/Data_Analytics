from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
from typing import List, Dict, Tuple, Sequence, Union
import logging
import json
from functools import reduce
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

MAIN_DIRECTORY = "path_to_your_main_dir_followed_by_a_slash"
DATA_FILE_NAME = "health_care_titles.txt"
ADDITIONAL_FREQUENT_WORDS_FILE_NAME = 'additional_frequent_words.json'
FREQUENT_WORDS_CATEGORIES = ['health', 'general', 'analytics', 'information technology']
FASTTEXT_LANGUAGE_DETECT_NAME = 'lid.176.bin'
TOKEN_PATTERN = r"\b[^\d\W]+\b"
PRETRAINED_EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
HUGGINGFACE_EMBEDDINGS_OUTPUT_NAME = 'last_hidden_state'
SENTENCE_TRANSFORMERS_MODELS_FOLDER_HUGGINGFACE = 'sentence-transformers/'
CLEANED_TITLES_NAME = 'cleaned_titles'
ORIGINAL_TITLES_NAME = 'original_titles'
BASIC_TRANSFORM_NAME = 'basic'


def get_extra_frequent_words_path() -> str:
    return os.path.join(MAIN_DIRECTORY, ADDITIONAL_FREQUENT_WORDS_FILE_NAME)


def get_fasttext_lang_detect_path() -> str:
    return os.path.join(MAIN_DIRECTORY, FASTTEXT_LANGUAGE_DETECT_NAME)


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


def find_duplicates(collection_of_titles: Union[Sequence[str], map]) -> tuple[dict[str, list[int]], int]:
    """
    param collection_of_titles: a non-empty collection of titles
    return: a dictionary of duplicated titles as keys and their associated indices in the collection as values
    """
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


def drop_nones_from_sequence(sequence_of_values: Sequence) -> Sequence:
    number_of_nones = sequence_of_values.count(None)
    if number_of_nones > 0:
        sequence_of_values = [each_element for each_element in sequence_of_values if each_element is not None]
        return sequence_of_values
    else:
        logging.warning('There are no Nones in the provided sequence')


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


def transform_word_to_vec(sequence_of_sentences: Union[Sequence[str], map], tokenizer_pattern: str = TOKEN_PATTERN,
                          method: str = 'basic', normalize_output: bool = True) -> Sequence[Sequence[float]]:
    """
    sequence_of_sentences: the collection of sentences to transform to vectors. Each element must be a non-empty
    string. None, '' and null values are not acceptable.
    tokenizer_pattern: a regex for tokenizing the text, this is required if method is set to 'basic'
    normalize_output: whether to normalize the transformed data
    method: type of words' transformation:
        1. basic: this is the simple word-count vectorizer which tries to find lexicographical similarity between
        phrases.
        2. embedding_all_mpnet: the 'all_mpnet_base_v2' sentence transformer is used to convert
        the sentences to vectors. This method can be potentially used for sentiment search.
    output: a collection with len(list_of_sentences) elements. Each element is a collection of vectorized words
    associated with a sentence.
    """
    if any(map(lambda each_sentence: each_sentence is None, sequence_of_sentences)):
        raise Exception('The sequence_of_sentences argument must contain valid sentences not Nones')

    if method == BASIC_TRANSFORM_NAME:
        word_vectorizer = CountVectorizer(token_pattern=tokenizer_pattern)
        transformed_vectors = word_vectorizer.fit_transform(sequence_of_sentences)
        if normalize_output:
            return normalize(transformed_vectors).toarray().tolist()
        else:
            return transformed_vectors.toarray().tolist()

    elif method == PRETRAINED_EMBEDDING_MODEL_NAME:
        transformed_vectors = huggingface_sentence_transform(sequence_of_sentences, normalize_output)
        return transformed_vectors.tolist()
    else:
        raise Exception('Please select either the basic or embedding_all_mpnet method')


def huggingface_sentence_transform(sequence_of_sentences: Union[Sequence[str], str], normalize: bool = False):
    tokenizer = \
        AutoTokenizer.from_pretrained(
            os.path.join(SENTENCE_TRANSFORMERS_MODELS_FOLDER_HUGGINGFACE, PRETRAINED_EMBEDDING_MODEL_NAME))
    sentence_transformer_model = \
        AutoModel.from_pretrained(
            os.path.join(SENTENCE_TRANSFORMERS_MODELS_FOLDER_HUGGINGFACE, PRETRAINED_EMBEDDING_MODEL_NAME))
    tensor_of_indexed_tokens = tokenizer(sequence_of_sentences,
                                         padding=True,
                                         truncation=True, return_tensors='pt')
    with torch.no_grad():
        token_and_pooled_embeddings = sentence_transformer_model(**tensor_of_indexed_tokens)
    token_embeddings = token_and_pooled_embeddings[HUGGINGFACE_EMBEDDINGS_OUTPUT_NAME]
    attention_mask = tensor_of_indexed_tokens['attention_mask']
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled_sentence_transforms = torch.sum(token_embeddings * attention_mask_expanded, 1) / torch.clamp(
        attention_mask_expanded.sum(1), min=1e-9)
    if normalize:
        return torch.nn.functional.normalize(pooled_sentence_transforms, p=2, dim=1)
    else:
        return pooled_sentence_transforms


def load_data(path_to_the_data: str = MAIN_DIRECTORY, file_name: str = DATA_FILE_NAME):
    with open(os.path.join(path_to_the_data, file_name), encoding='UTF-8') as titles:
        read_lines_original = titles.readlines()
    return read_lines_original

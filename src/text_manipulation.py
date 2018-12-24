"""Text manipulation methods and attributes used to parse TREC data

Attributes:
    EXTRA_STOP_WORDS (list of str): Manually added stopwords for STOP_WORDS
    HTML_UNESCAPE_TABLE (dict): HTML Character references to unescape
    PUNCT_REGEX (re): Regex that removes punctuation
    STEMMER (stemmer): Porter stemmer
    STOP_WORDS (list of str): Stopwords to remove
"""
import collections
import re
import string
# import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

HTML_UNESCAPE_TABLE = {
    "&lt;": " ",
    "&sect;": " ",
    "&gt;": " ",
    "&hyph;": " ",
    "&apos;": " ",
    "&quot;": " ",
    "&amp;": " ",
    "&blank;": " ",
    "&pacute;": "p",
    "&racute;": "r",
    "&mu;": " ",
    "&cacute;": "c",
    "&ccedil;": "c",
    "&acirc;": "a",
    "&cent;": "c",
    "&sect;": " ",
    "&ntilde;": "n",
    "&ge;": " ",
    "&agrave;": "a",
    "&para;": " ",
    "&ugrave;": "u",
    "&ograve;": "o",
    "&iuml;": "i",
    "&ocirc;": "o",
    "&lacute;": "l",
    "&auml;": "a",
    "&racute;": "r",
    "&euml;": "e",
    "&bull;": " ",
    "&ncirc;": "n",
    "&reg;": " ",
    "&ouml;": "o",
    "&times;": " ",
    "&eacute;": "e",
    "&utilde;": "u",
    "&uacute;": "u",
    "&sacute;": "s",
    "&iacute;": "i",
    "&oacute;": "o",
    "&egrave;": "e",
    "&igrave;": "i",
    "&cir;": " ",
    "&rsquo;": " ",
    "&aacute;": "a",
    "&uuml;": "u",
}

PUNCT_REGEX = re.compile('[%s]' % re.escape(string.punctuation + "1234567890"))

EXTRA_STOP_WORDS = [
    "the",
    "also",
    "said",
    "say",
    "well"
]

STEMMER = PorterStemmer()

def html_unescape(text):
    """Unescapes HTML character references from HTML_UNESCAPE_TABLE

    Args:
        text (str): Text
    
    Returns:
        str: Text with referenced characters
    """
    rep = dict((re.escape(k), v) for k, v in HTML_UNESCAPE_TABLE.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)


def remove_punctuation(text):
    """Removes all punctuation from input in a single pass
    
    Args:
        text (str): Text
    
    Returns:
        str: Text without punctuation
    """
    text.replace(".", "")
    return PUNCT_REGEX.sub(' ', text)

# STOP_WORDS = [remove_punctuation(w) for w in stopwords.words("english")] + EXTRA_STOP_WORDS
STOP_WORDS = ['the']

def bag_of_words(text):
    """Converts input to lowercase list of terms
    
    Args:
        text (str): Text to be converted
    
    Returns:
        list of str: Terms
    """
    return word_tokenize(text.lower())

def stop_and_stem(bow):
    """Removes stopwords and stems terms in input using Porter
    
    Args:
        bow (list of str): Terms to be processed
    
    Returns:
        list of str: Stemmed terms without stopwords
    
    Deleted Parameters:
        bag_of_words (list of str): Terms to be stopped and stemmed
    """
    return [STEMMER.stem(t) for t in bow if t not in STOP_WORDS]

def process_text(text):
    """Converts input text to list of processed terms
    
    Args:
        text (str): Text to be processed
    
    Returns:
        list of str: Processed terms
    """
    return stop_and_stem(bag_of_words(remove_punctuation(html_unescape(text))))

def to_frequency_pairs(terms):
    """Converts terms to frequency pairs
    
    Args:
        terms (list of str): Terms to be counted
    
    Returns:
        list of (str, int): Pairs of (term, frequency count) of the input list
    """
    counter = collections.Counter(terms)
    return list(zip(counter.keys(), counter.values()))

def get_blocks(text, tag, extra_space=1):
    """Converts TREC XML-like data to blocks surrounded by tag
    
    Args:
        text (str): XML-like TREC data
        tag (str): Tag surrounding blocks
        extra_space (bool, optional): Whether to remove an extra space around tags
    
    Returns:
        str: All blocks in text surrounded by tag
    """
    offset = 2 + extra_space
    all_blocks = re.findall(r'(<' + tag + r'>((?!<\/DOC>)[\s\S])*<\/' + tag + '>)', text)
    return [b[0][len(tag)+offset:-len(tag)-offset-1] for b in all_blocks]

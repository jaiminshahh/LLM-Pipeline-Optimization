import html
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

import string

# Initialize resources once, outside of the functions
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # Load stopwords once


def tweet_cleaner(tweet):
    # Decode HTML entities
    tweet = html.unescape(tweet)

    # Remove links (starting with http/https/ftp etc.)
    tweet = re.sub(r'http\S+|www\S+|ftp\S+', '', tweet)
    
    # Remove emojis using a regex pattern
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # Flags
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"  # Additional symbols
                               "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the tweet and remove stopwords and words shorter than 3 characters
    words = word_tokenize(tweet)
    filtered_words = [word for word in words if word.lower() not in stop_words and len(word) >= 3]
    
    # Join the filtered words back into a single string
    cleaned_tweet = ' '.join(filtered_words)

    return cleaned_tweet


# Function to convert nltk POS tags to WordNet POS tags
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# Function to lemmatize the cleaned tweet
def lemmatize_tweet(tweet):
    tweet = tweet_cleaner(tweet)
    nltk_tagged = nltk.pos_tag(word_tokenize(tweet))  # POS tagging

    # Map words to their WordNet POS tag
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_tweet.append(word)  # No tag available, append as is
        else:
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))  # Lemmatize
    return " ".join(lemmatized_tweet)
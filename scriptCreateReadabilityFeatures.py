## use this script to process the datasets for READABILITY FEATURES EXTRACTION 
# run the rogram with EXACTLY 5 argruments as shown below:
#       python <script> <input-file-name> <output-CSV-file> <#skipInitialRows> <#rowsToProcess> <#rowsAfterWhichToShowPrintMessageTracker>
#
#       #skipInitialRows
#               is an integer. It is the index value in the url columns from which the processing should start.
#       #rowsToProcess
#               is an integer. If = -1 then will read in all the rows after skipping as per above parameter.
#               Otherwise, will read in the number specified.

import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
import en_core_web_lg
import en_core_web_sm
from contractions import contractions_dict
from textstat.textstat import textstatistics, easy_word_set, legacy_round
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import MinMaxScaler

import csv
import sys
from datetime import datetime
import logging

global dataIpFile, csvOpFile, skipInitialRows, rowsToProcess, printMessageFreq

def ProcessCommandLineArgs():
    global dataIpFile, csvOpFile, skipInitialRows, rowsToProcess, printMessageFreq
    ## check number of arguments passed, should be 5
    if len(sys.argv) != 6:
        print(f"ERROR: Expected EXACTLY 6 argruments as shown below:\n")
        print(f"python <script> <input-file-name> <output-CSV-file> <#skipInitialRows> <#rowsToProcess> <#rowsAfterWhichToShowPrintMessageTracker>")
        print(f"But you passed {len(sys.argv)} arguments.")
        print(f"ERROR: Exiting program with RC = 100")
        exit(100)
    ## pick up and assign the arguments
    try:
        dataIpFile = sys.argv[1]
        csvOpFile = sys.argv[2]
        skipInitialRows = int(sys.argv[3])
        rowsToProcess = int(sys.argv[4])
        printMessageFreq = int(sys.argv[5])
    except:
        print(f"ERROR: Some parameters are wrong...recheck and run again.\nExiting the program with RC = 110")
        exit(110)
    
    return(0)

##
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
##
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
##
def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    expanded_text = "%s%s" % (expanded_text[0].upper(), expanded_text[1:])

    return expanded_text
#expand_contractions("You all can't expand contractions I'd think", contractions_dict)
##
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text
##
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
##
def stopWords(text):
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(text)
    wordCount = word_count(text)
    stopWordsCount = len(stopWords)
    return ((stopWordsCount/wordCount)*100)
##
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
##
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=False, 
                     text_lemmatization=False, special_char_removal=False, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc, contractions_dict)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
##
# Returns Number of Words in the text 
def word_count(text): 
    sentences = break_sentences(text) 
    words = 0
    for sentence in sentences: 
        words += len([token for token in sentence]) 
    return words
##
# Returns the number of sentences in the text 
def sentence_count(text): 
    sentences = break_sentences(text) 
#     return len(sentences) 
#     return sentences.count()
    return sum(1 for i in sentences)
##
def capitalized_words(text):
#     len(t) = re.findall(r'(\b[A-Z]([a-z])*\b)',text)
    return len(re.findall(r'(\b[A-Z]([a-z])*\b)',text))
##
def character_count(text):
    return len(text)
##
def break_sentences(text): 
    nlp = en_core_web_lg.load()
#     nlp = spacy.load('en') 
    doc = nlp(text) 
    return doc.sents 
##
# Returns average sentence length 
def avg_sentence_length(text): 
    words = word_count(text) 
    sentences = sentence_count(text) 
    average_sentence_length = float(words / sentences) 
    return average_sentence_length 
##
# Returns the average number of syllables per 
# word in the text 
def avg_syllables_per_word(text): 
    syllable = syllables_count(text) 
    words = word_count(text) 
    ASPW = float(syllable) / float(words) 
    return legacy_round(ASPW, 1) 
##
# Textstat is a python package, to calculate statistics from  
# text to determine readability,  
# complexity and grade level of a particular corpus. 
# Package can be found at https://pypi.python.org/pypi/textstat 
def syllables_count(word): 
    return textstatistics().syllable_count(word) 
##
def flesch_reading_ease(text): 
    """ 
        Implements Flesch Formula: 
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW) 
        Here, 
          ASL = average sentence length (number of words  
                divided by number of sentences) 
          ASW = average word length in syllables (number of syllables  
                divided by number of words) 
    """
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - float(84.6 * avg_syllables_per_word(text)) 
    return legacy_round(FRE, 2) 
##
### this was left commented in the jupyter script of sanikas
# def flesch_reading_ease(text): 
#     return textstatistics().flesch_reading_ease(text)
##
# Return total Difficult Words in a text 
def difficult_words(text): 
  
    # Find all words in the text 
    words = [] 
    sentences = break_sentences(text) 
    for sentence in sentences: 
        words += [str(token) for token in sentence] 
  
    # difficult words are those with syllables >= 2 
    # easy_word_set is provide by Textstat as  
    # a list of common words 
    diff_words_set = set() 
      
    for word in words: 
        syllables = syllables_count(word) 
        if word not in easy_word_set and syllables >= 2: 
            diff_words_set.add(word) 
  
    return len(diff_words_set)
##
def gunning_fog(text): 
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words) 
    return grade 
##
# A word is polysyllablic if it has more than 3 syllables 
# this functions returns the number of all such words  
# present in the text 
def poly_syllables_counts(text): 
    counts = 0
    words = [] 
    sentences = break_sentences(text)
    for sentence in sentences: 
        words += [token for token in sentence] 
      
    #if words is not None:
    for word in words: 
        syllables = syllables_count(word) 
#             syllables = textstatistics.syllable_count(word) 
        if syllables >= 3: 
            counts += 1
    return counts
##
def count_urls(text):
    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
##
def count_long_Words(s):
    words = s.split()
    return len([word for word in words if len(word)>4])
##
def linsear_write_formula(word): 
    return textstatistics().linsear_write_formula(word)
##
def flesch_kincaid_grade(word):
    return textstatistics().flesch_kincaid_grade(word)
##
def smog_index(word):
    return textstatistics().smog_index(word)
##
def automated_readability_index(word):
    return textstatistics().automated_readability_index(word)
##
def coleman_liau_index(word):
    return textstatistics().coleman_liau_index(word) 
##
def dale_chall_readability_score(word):
    return textstatistics().dale_chall_readability_score(word)
##
def lexicon_count(word):
    return textstatistics().lexicon_count(word)
##
def perform_readability_feature_extraction(data3, printFreqValue=100):

#     'Flesch_title_reading_ease''Flesch_title_kincaid_grade','Flesch_title_smog','Flesch_title_gunning_fog','Flesch_title_words',
#     'Flesch_title_lexicon','Flesch_title_syllables','Flesch_title_stop_words','Flesch_title_sentences','Flesch_title_linsear_write',
#     'Flesch_title_automated_readability','Flesch_title_coleman_liax','Flesch_title_difficult_words','Flesch_title_words_total'

#         ['title_Flesch_reading_ease','title_Flesch_kincaid_grade','title_smog','title_gunning_fog','title_words_per_sentence',
#         'title_capitalized_words','title_lexicon','title_syllables','title_stop_words','title_sentences','title_linsear_write', 
# 'title_complex_words','title_automated_readability','title_characters_total','title_coleman_liax','title_difficult_words',
# 'title_words_total',

#             'full_text_Flesch_reading_ease','full_text_Flesch_kincaid_grade','full_text_smog',
#             'full_text_gunning_fog','full_text_words_per_sentence','full_text_capitalized_words','full_text_lexicon','full_text_syllables','full_text_stop_words',
#             'full_text_sentences','full_text_linsear_write','full_text_complex_words','full_text_automated_readability',
#             'full_text_characters_total','full_text_coleman_liax','full_text_difficult_words','full_text_words_total']
    l = 0
    #while l < 1:#len(data3):
    while l < len(data3):
        t = str(data3.iloc[l].clean_full_text)
        data3.loc[l, ['full_text_Flesch_reading_ease']] = flesch_reading_ease(t)
        data3.loc[l, ['full_text_Flesch_kincaid_grade']] = flesch_kincaid_grade(t)
        data3.loc[l, ['full_text_smog']] = smog_index(t)
        data3.loc[l, ['full_text_gunning_fog']] = gunning_fog(t)
        # this is words per sentence
        data3.loc[l, ['full_text_words_per_sentence']] = avg_sentence_length(t)
        data3.loc[l, ['full_text_capitalized_words']] = capitalized_words(t)
        data3.loc[l, ['full_text_lexicon']] = lexicon_count(t)
        data3.loc[l, ['full_text_urls_counts']] = count_urls(t)
        data3.loc[l, ['full_text_long_words']] = count_long_Words(t)
        data3.loc[l, ['full_text_syllables']] = syllables_count(t)
        data3.loc[l, ['full_text_stop_words']] = stopWords(t)
        data3.loc[l, ['full_text_sentences']] = sentence_count(t)
        data3.loc[l, ['full_text_linsear_write']] = linsear_write_formula(t)
        #data3.loc[l, ['full_text_complex_words']] = poly_syllables_counts(t)
        data3.loc[l, ['full_text_automated_readability']] = automated_readability_index(t)
        data3.loc[l, ['full_text_characters_total']] = character_count(t)
        data3.loc[l, ['full_text_coleman_liax']] = coleman_liau_index(t)
        data3.loc[l, ['full_text_difficult_words']] = difficult_words(t)
        data3.loc[l, ['full_text_words_total']] = word_count(t)
        
        l = l + 1
        
        logging.warning(f"\n\t\tProcessed url = {data3.url[l-1]}\nat row number = {l-1}")
        if l % printFreqValue == 0:
            print(f"\nProcessed url = {data3.url[l-1]}\nat row number = {l-1}")
    
    return data3
######## main logic starts ##########
if ProcessCommandLineArgs() == 0:
    logFileName = sys.argv[0] + '_LOG_' + sys.argv[3] + '_' + sys.argv[4] + '_' + sys.argv[5] + '.log'
    logging.basicConfig(level=logging.WARNING, filename=logFileName,                  \
        filemode='w', format='%(asctime)s %(levelname)s:%(message)s')
    print(f"All command line arguments are valid....starting main processing")
    logging.warning(f"All command line arguments are valid....starting main processing.")

print(f'\nStart time: {datetime.now().strftime("%c")}')
logging.warning(f'\nStart time: {datetime.now().strftime("%c")}')

print(f"\n")
print(f"Processing with command line arguments as:")
print(f"1) dataIpFile       = {dataIpFile}")
print(f"2) csvOpFile        = {csvOpFile}")
print(f"3) skipInitialRows  = {skipInitialRows}")
print(f"4) rowsToProcess    = {rowsToProcess}")
print(f"5) printMessageFreq = {printMessageFreq}")
print(f"\n")
logging.warning(f"\n\n")
logging.warning(f"Processing with command line arguments as:")
logging.warning(f"1) dataIpFile       = {dataIpFile}")
logging.warning(f"2) csvOpFile        = {csvOpFile}")
logging.warning(f"3) skipInitialRows  = {skipInitialRows}")
logging.warning(f"4) rowsToProcess    = {rowsToProcess}")
logging.warning(f"5) printMessageFreq = {printMessageFreq}")
logging.warning(f"\n\n")

## read in the all the rows if rowsToProcess is -1 AFTER skipping the number of data rows specified.
if rowsToProcess == -1 :
    data = pd.read_csv(dataIpFile, skiprows = range(1, skipInitialRows + 1), sep = ',', low_memory=False)
else:
    data = pd.read_csv(dataIpFile, skiprows = range(1, skipInitialRows + 1), nrows = rowsToProcess, sep = ',', low_memory=False)

print(f"Input data read into dataframe.\n")
logging.warning(f"Input data read into dataframe.\n")

###########
## process Input File and Build Output Dataframe
###########

data1 = data.copy()
del(data)
data1['full_text'] = data1["title"].map(str)+ '. ' + data1["content"]
del data1['title']
del data1['content']

#### PENDING   #### PENDING   #### PENDING   #### PENDING   #### PENDING   #### PENDING
### remove any rows with non-English stuff
#### PENDING   #### PENDING   #### PENDING   #### PENDING   #### PENDING   #### PENDING

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
nlp = en_core_web_lg.load()

data1['clean_full_text'] = normalize_corpus(data1['full_text'])
print(f"\nNormalized the full_text into clean_full_text\n")
logging.warning(f"\n\nNormalized the full_text into clean_full_text\n")

del data1['full_text']

## shape should be (num. of urls, 3 columns) as only the domain, url and clean_full_text column should be present now
print(f"The shape should be = number of urls, 3 columns")
logging.warning(f"The shape should be = number of urls, 3 columns")
print(f"data1.shape = {data1.shape}")
logging.warning(f"data1.shape = {data1.shape}")

new_cols = [ 'full_text_Flesch_reading_ease', 'full_text_Flesch_kincaid_grade','full_text_smog', 'full_text_gunning_fog',      \
    'full_text_words_per_sentence', 'full_text_capitalized_words', 'full_text_lexicon', 'full_text_urls_counts',               \
    'full_text_long_words', 'full_text_syllables', 'full_text_stop_words', 'full_text_sentences', 'full_text_linsear_write',   \
    'full_text_complex_words', 'full_text_automated_readability', 'full_text_characters_total', 'full_text_coleman_liax',      \
    'full_text_difficult_words', 'full_text_words_total' ]
data1 = data1.assign(**dict.fromkeys(new_cols , -1.00))

## shape should be (num. of urls, 22 columns) as only the domain, url and clean_full_text column should be present now with 19 features
print(f"The shape should be = number of urls, 22 columns")
logging.warning(f"The shape should be = number of urls, 22 columns")
print(f"data1.shape = {data1.shape}")
logging.warning(f"data1.shape = {data1.shape}")

## actually start feature extraction

print(f'\n\nStarting the processing for readability features at:\n{datetime.now().strftime("%c")}\n\n')
logging.warning(f'\n\n\n\t\tStarting the processing for readability features at:\n\t{datetime.now().strftime("%c")}\n\n')

data1 = perform_readability_feature_extraction(data1, printFreqValue=printMessageFreq)

## not using this feature so deleting now
del data1['full_text_complex_words']

## shape should be (num. of urls, 21 columns) as only the domain, url and clean_full_text column should be present now with 18 features
print(f"The shape should be = number of urls, 21 columns, domain url, clean_text and 18 features.")
logging.warning(f"The shape should be = number of urls, 21 columns, domain url, clean_text and 18 features.")
print(f"data1.shape = {data1.shape}")
logging.warning(f"data1.shape = {data1.shape}")

## min max the features
dfOutTempNorm = data1.iloc[:, 3:]  ## not taking the first two columns for domain and url
scaler = MinMaxScaler()

dfOutTempNorm[['full_text_Flesch_reading_ease',   \
 'full_text_Flesch_kincaid_grade',   \
 'full_text_smog',   \
 'full_text_gunning_fog',   \
 'full_text_words_per_sentence',   \
 'full_text_capitalized_words',   \
 'full_text_lexicon',   \
 'full_text_urls_counts',   \
 'full_text_long_words',   \
 'full_text_syllables',   \
 'full_text_stop_words',   \
 'full_text_sentences',   \
 'full_text_linsear_write',   \
 'full_text_automated_readability',   \
 'full_text_characters_total',   \
 'full_text_coleman_liax',   \
 'full_text_difficult_words',   \
 'full_text_words_total']]   \
= \
scaler.fit_transform(dfOutTempNorm[['full_text_Flesch_reading_ease',   \
 'full_text_Flesch_kincaid_grade',   \
 'full_text_smog',   \
 'full_text_gunning_fog',   \
 'full_text_words_per_sentence',   \
 'full_text_capitalized_words',   \
 'full_text_lexicon',   \
 'full_text_urls_counts',   \
 'full_text_long_words',   \
 'full_text_syllables',   \
 'full_text_stop_words',   \
 'full_text_sentences',   \
 'full_text_linsear_write',   \
 'full_text_automated_readability',   \
 'full_text_characters_total',   \
 'full_text_coleman_liax',   \
 'full_text_difficult_words',   \
 'full_text_words_total']])

#################
## build the dfOut dataframes
#################

dfOut = data1.copy()
del dfOut['clean_full_text']
del(data1)
## now dfOut should have only two columns: domain and url

dfOut['norm_full_text_Flesch_reading_ease']   = dfOutTempNorm['full_text_Flesch_reading_ease']        
dfOut['norm_full_text_Flesch_kincaid_grade']  = dfOutTempNorm['full_text_Flesch_kincaid_grade']
dfOut['norm_full_text_smog']                  = dfOutTempNorm['full_text_smog']
dfOut['norm_full_text_gunning_fog']           = dfOutTempNorm['full_text_gunning_fog']
dfOut['norm_full_text_words_per_sentence']    = dfOutTempNorm['full_text_words_per_sentence']
dfOut['norm_full_text_capitalized_words']     = dfOutTempNorm['full_text_capitalized_words']
dfOut['norm_full_text_lexicon']               = dfOutTempNorm['full_text_lexicon']
dfOut['norm_full_text_urls_counts']           = dfOutTempNorm['full_text_urls_counts']
dfOut['norm_full_text_long_words']            = dfOutTempNorm['full_text_long_words']
dfOut['norm_full_text_syllables']             = dfOutTempNorm['full_text_syllables']
dfOut['norm_full_text_stop_words']            = dfOutTempNorm['full_text_stop_words']
dfOut['norm_full_text_sentences']             = dfOutTempNorm['full_text_sentences']
dfOut['norm_full_text_linsear_write']         = dfOutTempNorm['full_text_linsear_write']
dfOut['norm_full_text_automated_readability'] = dfOutTempNorm['full_text_automated_readability']
dfOut['norm_full_text_characters_total']      = dfOutTempNorm['full_text_characters_total']
dfOut['norm_full_text_coleman_liax']          = dfOutTempNorm['full_text_coleman_liax']
dfOut['norm_full_text_difficult_words']       = dfOutTempNorm['full_text_difficult_words']
dfOut['norm_full_text_words_total']           = dfOutTempNorm['full_text_words_total']
## now dfOut should have only two earlier columns: domain and url and the new 18 feature columns, so total 20 columns

try:
    ## write to output file
    dfOut.to_csv(csvOpFile, index=False)
    print(f"\n\nOutput file written to =\n{csvOpFile}")
    logging.warning(f"\n\nOutput file written to =\n{csvOpFile}")
except:
    print(f"Problem...error writing to output file")
    logging.warning(f"Problem...error writing to output file")

## print the summary info before exiting
print(f"Number of rows skipped   = {skipInitialRows}")
logging.warning(f"Number of rows skipped   = {skipInitialRows}")
print(f"Number of rows processed   = {rowsToProcess}")
logging.warning(f"Number of rows processed   = {rowsToProcess}")

print(f'\n\nEnd time: {datetime.now().strftime("%c")}')
logging.warning(f'\n\nEnd time: {datetime.now().strftime("%c")}')

exit(0)
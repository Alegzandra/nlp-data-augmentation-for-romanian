import rowordnet
from rowordnet import Synset
import stanza
import pandas as pd
from spacy_stanza import StanzaLanguage
from collections import Counter
from math import sqrt
import re
import json
import requests
from nltk.stem.snowball import SnowballStemmer

#stanza.download('ro')    #se ruleaza o singura data
wordnet = rowordnet.RoWordNet()
snlp = stanza.Pipeline(lang='ro')
nlp = StanzaLanguage(snlp)
stemmer = SnowballStemmer("romanian")

#citesc fisierul CSV
dataset_df = pd.read_csv('input_data.csv', encoding='utf8')
phrases = dataset_df['Phrases']
print("Numarul de propozitii in fisierul de intrare:", len(phrases))
augmented_data = {}

def word_similarity(word1, word2):
    """
    Returns the similarity between word1 and word2.
    Returns 1 if the stems are equal and 1/dist**2 if not, where dist is the shortest path between synsets
    corresponding to the words
    :return: similarity between 1,0
    """
    #if stemmer.stem(word1) == stemmer.stem(word2):
    #    return 1
    synsets1 = wordnet.synsets(literal=str(word1))
    if not synsets1:
        return 0
    synsets2 = wordnet.synsets(literal=str(word2))
    if not synsets2:
        return 0
    try:
        path_length = len(wordnet.shortest_path(synsets1[0], synsets2[1]))
    except:
        return 0
    return 1/path_length**2


def get_wordvector_similarity(nlp,replacements):
    """
    From the list of synonyms obtained from Wordnet, apply the
    similarity score to filter out non-relevant synonyms. The word pair who has similarity score less than
    THRESHOLD is neglected.
    """
    replacements_refined = {}
    THRESHOLD = 0.025     #empiric
    for key, values in replacements.items():
        key_vec = nlp(key.lower())
        synset_refined = []
        for each_value in values:
            value_vec = nlp(each_value.lower())
            if word_similarity(key_vec, value_vec) > THRESHOLD:
            #if key_vec.similarity(value_vec) > THRESHOLD:
                synset_refined.append(each_value)
        if len(synset_refined) > 0:
            replacements_refined[key] = synset_refined
    return replacements_refined

def convert(s):
    """
    Converts a list of characters into a word.
    """
    #initialization of the string
    new = ""
    #traverse the string
    for x in s:
        new += x
    return new

for current_sentence in phrases:
    print("\tPropozitia de la intrare:", current_sentence)
    doc = nlp(current_sentence)
    replacements = {}
    for token in doc:
        if token.pos_ == 'NOUN' and token.ent_type == 0:  # daca este substantiv si nu entitate (adica substantiv propriu)
            syns = wordnet.synsets(token.text, pos=Synset.Pos.NOUN) #returneaza toate ID-urile synseturilor substantiv din datele de intrare
            synonyms = []
            literals = set()
            good_literals = []
            for synset_id in syns:
                new_synset = wordnet(synset_id)
                new_synset.literals[0] = tuple(new_synset.literals[0])
                literals.add(convert(new_synset.literals[0]))
            for literal in literals:
                if literal.lower() != token.text.lower() and literal != token.lemma_:
                    #synonyms.append(literal.replace("_", " "))
                    synonyms.append(literal)
            if len(synonyms) > 0:
                replacements[token.text] = synonyms

        if token.pos_ == 'ADJ':  # if its an adjective
            """Augment the adjective with possible synonyms from RoWordnet"""
            syns = wordnet.synsets(token.text, pos=Synset.Pos.ADJECTIVE) # returneaza toate ID-urile synseturilor adjectiv din datele de intrare
            synonyms = []
            literals = set()
            good_literals = []
            for synset_id in syns:
                new_synset = wordnet(synset_id)
                new_synset.literals[0] = tuple(new_synset.literals[0])
                literals.add(convert(new_synset.literals[0]))
            for literal in literals:
                if literal.lower() != token.text.lower() and literal != token.lemma_:
                    #synonyms.append(literal.replace("_", " "))
                    synonyms.append(literal)
            if len(synonyms) > 0:
                replacements[token.text] = synonyms

        if token.pos_ == 'VERB':  # if its a verb
            """Augment the verb with possible synonyms from Wordnet"""
            syns = wordnet.synsets(token.text, pos=Synset.Pos.VERB) #returneaza toate ID-urile synseturilor verb din datele de intrare
            synonyms = []
            literals = set()
            good_literals = []
            for synset_id in syns:
                new_synset = wordnet(synset_id)
                new_synset.literals[0] = tuple(new_synset.literals[0])
                literals.add(convert(new_synset.literals[0]))
            for literal in literals:
                if literal.lower() != token.text.lower() and literal != token.lemma_:
                    #synonyms.append(literal.replace("_", " "))
                    synonyms.append(literal)
            if len(synonyms) > 0:
                replacements[token.text] = synonyms

    print("replacements", replacements)
    # print("Input(before filtering):\n",sum(map(len, replacements.values())))
    replacements_refined_all = get_wordvector_similarity(nlp,replacements)
    print("replacements refined", replacements_refined_all)

    generated_sentences = []
    generated_sentences.append(current_sentence)
    for key, value in replacements_refined_all.items():
        replaced_sentences = []
        for each_value in value:
            for each_sentence in generated_sentences:
                new_sentence = re.sub(r"\b%s\b" % key,each_value,each_sentence)
                replaced_sentences.append(new_sentence)
        generated_sentences.extend(replaced_sentences)
    augmented_data[current_sentence] = generated_sentences

print("Numarul total de variatii create:", sum(map(len, augmented_data.values())))

print("Salvez CSV...")

augmented_dataset = {'Phrases':[],'Paraphrases':[]}
phrases = []
paraphrases = []
for key,value in augmented_data.items():
    for each_value in value:
        phrases.append(key)
        paraphrases.append(each_value)
augmented_dataset['Phrases'] = phrases
augmented_dataset['Paraphrases'] = paraphrases
augmented_dataset_df = pd.DataFrame.from_dict(augmented_dataset)
augmented_dataset_df.to_csv("augmented_dataset.csv", encoding='utf8', index=False)

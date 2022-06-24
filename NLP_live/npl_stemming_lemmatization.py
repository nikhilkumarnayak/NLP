# Stemming and Lemmatization
## Defination:-
### Stemming : Process of Reducing Infected words to their word stem.
### Lemmatization : Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma .

## Stemming Vs Lemmatization
### 1. Stemming word representation is not have any meaning where as the Lemmatization's word representation have meaning.
### 2. Stemming took less time to process where as Lemmatization took more time to process to get the base word.
### 3. By Stemming we can do the gmail/spam classification,Sentiment classifier,+ve & -ve classifier as we just required to find the base word. By Lemmatization is used chatbot,Q&A.

import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords # remove the unrelated words

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career."""

sentences = nltk.sent_tokenize(paragraph,"english")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("======================================================== STOPWORDS of ENGLISH - START =================================================================")
print(stopwords.words('english'))
print("======================================================== STOPWORDS of ENGLISH - START =================================================================")

print("======================================================== STEMMING - START =================================================================")
#### Stemming
for i in range(len(sentences)):
    swords = nltk.word_tokenize(sentences[i])
    swords = [stemmer.stem(word) for word in swords if word not in set(stopwords.words('english'))]
    swords = ' '.join(swords)
    print(swords)

print("======================================================== STEMMING - END =================================================================")

print("======================================================== LEMMATIZATION - START =================================================================")
#### Lemmatization
for j in range(len(sentences)):
    lwords = nltk.word_tokenize((sentences[j]))
    lwords = [lemmatizer.lemmatize(word) for word in lwords if word not in set(stopwords.words('english'))]
    lwords =' '.join(lwords)
    print(lwords)
print("======================================================== LEMMATIZATION - END =================================================================")





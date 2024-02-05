import copy
from math import log
import itertools
import string
import numpy as np
import pickle
from transformers import pipeline

class BigramLM:
    
    def load_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        print("Count of sentences in corpus: "+str(len(lines)))
        return lines
    
    def tokenize_sentence(self, myLines):
        lines = copy.deepcopy(myLines)
        lines = [i.strip("''").split(" ") for i in lines] 
        print("No of sentences in Corpus: "+str(len(lines)))
        return lines
    
    def prep_data(self, myLines):
        lines = copy.deepcopy(myLines)
        for i in range(len(lines)):
            lines[i] = [''.join(c for c in s if c not in string.punctuation) for s in lines[i]] # remove punctuations
            lines[i] = [s for s in lines[i] if s] # removes empty strings
            lines[i] = [word.lower() for word in lines[i]] # lower case
            lines[i] += ['</s>'] # Append </s> at the end of each sentence in the corpus
            lines[i].insert(0, '<s>')  # Append <s> at the beginning of each sentence in the corpus
        print("No of sentences in Corpus: "+str(len(lines)))
        return lines
    
    def vocabulary(self, dataset):
        dataset_vocab = set(itertools.chain.from_iterable(dataset))
        # remove <s> and </s> from the vocabulary of the dataset
        dataset_vocab.remove('<s>')
        dataset_vocab.remove('</s>')
        dataset_vocab = list(dataset_vocab)
        dataset_vocab.append('<s>')
        dataset_vocab.append('</s>')
        return dataset_vocab
    
    def freq_of_unique_words(self, lines):
        bag_of_words = list(itertools.chain.from_iterable(lines)) # change the nested list to one single list
        corpus_word_count = 0 # No of words in the corpus excluding <s> and </s>.
        #count the no. of times a word repeats in the corpus
        count = {}
        for word in bag_of_words:
            if word in count :
                count[word] += 1
            else:
                count[word] = 1
            if word != '<s>' and word != '</s>':
                corpus_word_count +=1
                
        unique_word_count = len(count) - 2 # number of unique words in the corpus excluding <s> and </s>
        
        #print("!!! IT IS EXCLUDING <s> AND </s> !!!")
        print("No of unique words in corpus : "+ str(unique_word_count))
        print("No of words in corpus: "+ str(corpus_word_count))
        
        return count
    
    def compute_bigram_frequencies(self, vocab, lines):
        bigram_frequencies = dict()
        for w1 in vocab:
            for w2 in vocab:
                bigram_frequencies[(w1, w2)] = 0
        #unique_bigrams = set()
        for sentence in lines:
            given_word = None
            for word in sentence:
                if given_word != None:
                    bigram_frequencies[(given_word, word)] = bigram_frequencies.get((given_word, word),0) + 1
    #                 if(previous_word!='<s>' and word!='</s>'):
    #                     unique_bigrams.add((previous_word,word))
                given_word = word
        #The number of bigram_frquencies in the corpus       
        #print(len(bigram_frequencies))
        return bigram_frequencies
    
    def compute_bigram_probabilities(self, bigram_frequencies,count):
        bigram_probabilities = dict() 
        for key in bigram_frequencies:
            numerator = bigram_frequencies.get(key)
            denominator = count.get(key[0]) # count.get(key[0]) will get the frequency of "given word" in the corpus.
            if (numerator ==0 or denominator==0):
                bigram_probabilities[key] = 0
            else:
                bigram_probabilities[key] = float(numerator)/float(denominator)
        return bigram_probabilities
    
    def compute_updated_bigram_probabilities(self, dataset_vocab, bigram_frequencies,count,smoothing):
        if (smoothing==0):
            bigram_probabilities = dict() 
            for key in bigram_frequencies:
                numerator = bigram_frequencies.get(key)
                denominator = count.get(key[0]) # count.get(key[0]) will get the frequency of "given word" in the corpus.
                if (numerator ==0 or denominator==0):
                    bigram_probabilities[key] = 0
                else:
                    bigram_probabilities[key] = float(numerator)/float(denominator)
            return bigram_probabilities
        
        elif (smoothing==1):
            bigram_probabilities = dict() 
            for key in bigram_frequencies:
                numerator = bigram_frequencies.get(key) + 1
                denominator = count.get(key[0]) + len(count)
                if (numerator ==0 or denominator==0):
                    bigram_probabilities[key] = 0
                else:
                    bigram_probabilities[key] = float(numerator)/float(denominator)
            return bigram_probabilities
        
        elif (smoothing==2):
            n1 = 0
            testFreq = 1
            for key in count:
                if count[key] == testFreq:
                    n1 += 1
            n2 = 0
            testFreq = 2
            for key in count:
                if count[key] == testFreq:
                    n2 += 1
            d= float(n1)/float(n1+2*n2)
            countP = dict()
            countL = dict()
            for x in count:
                countP[x] = 0
                countL[x] = 0
            for x in count:
                for y in count:
                    if bigram_frequencies.get((x,y),0)> 0:
                        countP[x]+= 1 
                    if bigram_frequencies.get((y,x),0)> 0:
                        countL[x]+= 1
            Pcont = dict()
            z = dict()
            for w in dataset_vocab:
                # Pcont[w] = countL[w]/(len(bigram_frequencies))
                # z[w] = (d/len(bigram_frequencies))*countP[w]
                Pcont[w] = countL[w]/len(list(filter(lambda x: bigram_frequencies[x] > 0, list(bigram_frequencies.keys()))))
                z[w] = (d/count[w])*countP[w]

            bigram_probabilities = dict()
            for key in bigram_frequencies:
                # bigram_probabilities[key] = max(bigram_frequencies.get(key,0)-d, 0)/len(bigram_frequencies) + z[key[0]]*Pcont[key[1]]
                bigram_probabilities[key] = max(bigram_frequencies.get(key,0)-d, 0)/count[key[0]] + z[key[0]]*Pcont[key[1]]

            return bigram_probabilities
    
    def compute_new_prob_test_sentence(self, bigram_probabilities, sentence):
        test_sent_prob = 0
        given_word = None
        for word in sentence:
            if given_word!=None:
                if bigram_probabilities.get((given_word,word))==0 or bigram_probabilities.get((given_word,word))== None:
                    return 0
                else:
                    test_sent_prob+=log((bigram_probabilities.get((given_word,word),1)),10)
            given_word = word

        return 10**test_sent_prob

    def generate_sentence_from_bigrams(self, bigram_probabilities, length=10):
        sentence = ['<s>']  # Start the sentence with the start token
        for _ in range(length - 1):
            current_word = sentence[-1]
            # Filter bigram probabilities for the current word and extract next words and probabilities
            next_words_probs = {k[1]: v for k, v in bigram_probabilities.items() if k[0] == current_word}
            next_words = list(next_words_probs.keys())
            probabilities = list(next_words_probs.values())
            if not next_words:  # If there are no next words, break the loop
                break
            next_word = np.random.choice(next_words, p=probabilities / np.sum(probabilities))  # Sample a next word
            if next_word == '</s>':  # If the next word is the end token, end the sentence
                break
            sentence.append(next_word)
        return ' '.join(sentence[1:])  # Return the generated sentence without the start token
    
    def emotion_scores(self, classifier, sample): 
        emotion= classifier(sample)
        return emotion[0]

    def generate_emotion_sentence_from_bigrams(self, bigram_probabilities, classifier, emotion_label, length=10, top_k=3):
        sentence = ['<s>']  # Start the sentence with the start token
        for _ in range(length - 1):
            current_word = sentence[-1]
            # Filter bigram probabilities for the current word and extract next words and probabilities
            next_words_probs = {k[1]: v for k, v in bigram_probabilities.items() if k[0] == current_word}
            next_words = list(next_words_probs.keys())
            probabilities = list(next_words_probs.values())
            if not next_words:  # If there are no next words, break the loop
                break
            
            # Select top_k candidates based on probabilities for emotion scoring
            top_candidates_idx = np.argsort(probabilities)[-top_k:]
            for idx in top_candidates_idx:
                hypothetical_next_word = next_words[idx]
                # Construct a hypothetical next sentence
                hypothetical_sentence = ' '.join(sentence + [hypothetical_next_word])
                # Get emotion scores for the hypothetical sentence
                scores = self.emotion_scores(classifier, hypothetical_sentence)
                # Find the score for the specified emotion
                for score in scores:
                    if score['label'] == emotion_label:
                        emotion_score = score['score']
                        probabilities[idx] *= (1 + emotion_score)  # Adjust probability with emotion score
            
            next_word = np.random.choice(next_words, p=np.array(probabilities) / np.sum(probabilities))  # Sample a next word
            if next_word == '</s>':  # If the next word is the end token, end the sentence
                break
            sentence.append(next_word)
        return ' '.join(sentence[1:])  # Return the generated sentence without the start token

    def get_top_k_bigrams(self, bigram_probabilities, k):
        bigram_list = list(bigram_probabilities.items())
        
        sorted_bigrams = sorted(bigram_list, key=lambda x: x[1], reverse=True)
        
        top_k_bigrams = sorted_bigrams[:k]
        
        top_k_bigrams_dict = {bigram: prob for bigram, prob in top_k_bigrams}
        return top_k_bigrams_dict

    def learnModel(self, filename):
        smoothing = 1
        dataset = self.prep_data(self.tokenize_sentence(self.load_file(filename)))
        dataVocab = self.vocabulary(dataset)
        unique_word_frequency = self.freq_of_unique_words(dataset)
        bigram_frequencies = self.compute_bigram_frequencies(dataVocab, dataset)
        bigram_probabilities = self.compute_updated_bigram_probabilities(dataVocab, bigram_frequencies, unique_word_frequency, smoothing)
        
        # # save dictionary to a pickle file
        # with open('bigram_probabilities_KN.pkl', 'wb') as fp:
        #     pickle.dump(bigram_probabilities, fp)
        #     print('dictionary saved successfully to file')
        
        # print(bigram_probabilities)
        print("Top 5 bigrams:")
        print(self.get_top_k_bigrams(bigram_probabilities, 5))
        
        test_sentences = [['i feel so'],['i feel most unwelcome']]
        mySentence = self.prep_data(self.tokenize_sentence(test_sentences[0]))
        # mySentence = list(itertools.chain.from_iterable(mySentence))
        print(mySentence[0])
        test_sentence_probability = self.compute_new_prob_test_sentence(bigram_probabilities, mySentence[0])
        print("Test sentence probability:")
        print(test_sentence_probability)
        
        print("Generating vanilla sentences:")
        print(self.generate_sentence_from_bigrams(bigram_probabilities, 10))
        print(self.generate_sentence_from_bigrams(bigram_probabilities, 10))
        
        classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True,)
        # print(self.emotion_scores(classifier, "I am going to the market"))
        
        print("Generating sentences with emotion label:")
        for label in ['joy', 'love', 'sadness', 'surprise', 'fear', 'anger']:
            print("Label = " + label)
            print(self.generate_emotion_sentence_from_bigrams(bigram_probabilities, classifier, label, 10, 20))
            print(self.generate_emotion_sentence_from_bigrams(bigram_probabilities, classifier, label, 10, 20))
            print()
        
        
myModel = BigramLM()
myModel.learnModel("NLP-A1\Other\corpus.txt")
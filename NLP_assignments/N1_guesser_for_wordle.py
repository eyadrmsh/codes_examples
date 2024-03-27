from random import choice
import yaml
from rich.console import Console
from collections import Counter
import math
import pandas as pd

class Guesser:
    '''
        INSTRUCTIONS: This function should return your next guess. 
        Currently it picks a random word from wordlist and returns that.
        You will need to parse the output from Wordle:
        - If your guess contains that character in a different position, Wordle will return a '-' in that position.
        - If your guess does not contain thta character at all, Wordle will return a '+' in that position.
        - If you guesses the character placement correctly, Wordle will return the character. 

        You CANNOT just get the word from the Wordle class, obviously :)
    '''
    def __init__(self, manual):
        self.word_list = yaml.load(open('wordlist.yaml'), Loader=yaml.FullLoader)
        self._manual = manual
        self.console = Console()
        self._tried = []
        self.count = 0 
     

    def restart_game(self):
        self._tried = []
        self.count = 0 
    
        self.word_list = yaml.load(open('wordlist.yaml'), Loader=yaml.FullLoader)

        
    def get_guess(self, result):
        if self._manual=='manual':
            return self.console.input('Your guess:\n')
        else:
            if self.count ==  0:
                guess = 'salet'
                self.count +=1
                self._tried.append(guess)
                
                if guess in self.word_list:
                    self.word_list.remove(guess)
            elif self.count == 1:
                self.word_list = restricting_words(result, list(self._tried)[-1] , self.word_list)
                guess = guess_info(self.word_list)
                self.count +=1
                self._tried.append(guess)
                
                if guess in self.word_list:
                    self.word_list.remove(guess)                           
            elif self.count > 1:
                if create_dupl_dict(self._tried[-1]):
                    if resricting_words_double(result, self._tried[-1], self.word_list):
                        self.word_list = resricting_words_double(result, self._tried[-1], self.word_list)
                else:
                    if restricting_words(result, list(self._tried)[-1], self.word_list):
                        self.word_list = restricting_words(result, list(self._tried)[-1], self.word_list)
                try:
                    guess = guess_entropy(self.word_list)
                except:
                    guess = choice(self.word_list)
                self.count +=1
                self._tried.append(guess)
                
                if guess in self.word_list:
                    self.word_list.remove(guess) 
                self.console.print(guess)
            return guess
        
def create_dupl_dict(word):
    counter = Counter(word)
    duplicates = {element: count for element, count in counter.items() if count > 1}
    return duplicates

def same_letter_positions(word):
    return [i for i, letter in enumerate(word) if letter in [letter for letter, count in Counter(word).items() if count > 1]]

def sign_combo(w_output, positions):
    return ''.join([w_output[i] for i in positions])

def same_letter_positions(word):
    return [i for i, letter in enumerate(word) if letter in [letter for letter, count in Counter(word).items() if count > 1]]

def resricting_words_double( w_output, guess, word_list):
    pos = same_letter_positions(guess)
    w_output_list = list(w_output)
    w_output_list[pos[0]] = '#'
    w_output_list[pos[1]] = '#'
    w_output_mod = ''.join(w_output_list)
    word_list = restricting_words(w_output_mod, guess, word_list)
    w_output_adapted = ''.join(['=' if ch.isalpha() else ch for ch in w_output])
    combo = sign_combo(w_output_adapted, pos)
    if combo == '+=':
        word_list = [word for word in word_list if (word[pos[1]] == guess[pos[1]]) and (word[pos[1]] not in word[:pos[0]] + word[pos[0]+1:pos[1]] + word[pos[1]+1:])]   
    elif combo == '=+':
        word_list = [word for word in word_list if (word[pos[0]] == guess[pos[0]]) and (word[pos[0]] not in word[:pos[0]] + word[pos[0]+1:pos[1]] + word[pos[1]+1:])]    
    elif combo == '++':
        word_list = restricting_words(w_output, guess, word_list)
    elif combo == '-+':
        word_list = [word for word in word_list if guess[pos[0]] in word[:pos[0]]+word[pos[0]+1:pos[1]]+word[pos[1]+1:]]
    elif combo == '-+':
        word_list = [word for word in word_list if guess[pos[1]] in word[:pos[0]]+word[pos[0]+1:pos[1]]+word[pos[1]+1:]]    
    elif combo == '--':
        word_list = [word for word in word_list if Counter(word).most_common(1)[0] == (guess[pos[0]], 2)]  
    elif combo == '-=':
        word_list = [word for word in word_list if (word[pos[1]] == guess[pos[1]]) and (word[pos[1]] in word[:pos[0]] + word[pos[0]+1:pos[1]] + word[pos[1]+1:])] 
    elif combo == '=-':
        word_list = [word for word in word_list if (word[pos[0]] == guess[pos[0]]) and (word[pos[0]] in word[:pos[0]] + word[pos[0]+1:pos[1]] + word[pos[1]+1:])] 
    elif combo == '==':
        word_list = [word for word in word_list if (word[pos[1]] == guess[pos[1]]) and (word[pos[0]] == guess[pos[0]])]


def restricting_words(w_output, guess, word_list):

    dsame = {x:y for x, y in enumerate(w_output) if y.isalpha()==True}
    if dsame:
        word_list = [word for word in word_list if not dsame or all(word[pos] == i for pos, i in dsame.items())]
    
    dplus = {x: y for x, y in enumerate(w_output) if y == '+'}
    d = {}
    if dplus:
        for key in dplus.keys():
            d[key] = guess[key]
        word_list = [word for word in word_list if not dplus or all(i not in word for i in d.values())]

    dminus = {x: y for x, y in enumerate(w_output) if y == '-'}
    d = {}
    if dminus:
        for key in dminus.keys():
            d[key] = guess[key]
        word_list = [word for word in word_list if not d or all(word[pos] != i for pos, i in d.items())]
        word_list = [word for word in word_list if not d or all(i in word[0:pos]+word[pos+1:] for pos, i in d.items())]

    return word_list



def guess_info( word_list):

    ch_prob = {}
    position = {}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for i in alphabet:
        ch_prob[i] = 0
        position[i] = [0,0,0,0,0]

    ch_prob = {ch: sum(1 for word in word_list for letter in word if letter == ch) for ch in alphabet}
    position = {ch: [sum(1 for word in word_list if word[i] == ch) for i in range(5)] for ch in alphabet}
    ch_prob = {key: ch_prob[key] / len(word_list) * 5 for key in ch_prob}
    position = {key: [num / sum(position[key]) if sum(position[key]) > 0 else 0 for num in position[key]] for key in position}
    letter_like = {}
    letter_like = {letter: [ch_prob[letter] * position[letter][i] for i in range(5)] for letter in alphabet}
    pos = {}

    for i in range(5):
        pos[i] = {key: letter_like[key][i] for key in letter_like.keys()}

    for i in range(5):
        pos[i] = dict(sorted(pos[i].items(), key=lambda item: item[1], reverse=True))    

    word = ''
 
    for i in range(5):
        count = 0
        while len(word)<(i+1):
            if list(pos[0].keys())[count] not in word:
                word += list(pos[0].keys())[count]
            else:
                count += 1       
    return word
        

def guess_entropy(word_list):
    dict = {}
    for word in word_list:
        patterns = []
        initials = word_list.copy()
        initials.remove(word)
        for initial in initials:
            pattern = []
            for (init,gue) in zip(initial, word):
                if init == gue:
                    pattern += init
                else:
                    pattern += '+'
            for i, (init, gue) in enumerate(zip(initial, word)):
                if init != gue and gue in initial:
                    pattern[i] = '-'
            res = ''.join(pattern)
            patterns.append(res)
        counts = Counter(patterns)
        entropy = 0
        total = sum(counts.values())
        for c in counts:
            entropy+=(counts[c]/total)*math.log2(1/(counts[c]/total))
        dict.update({word:entropy})
    return max(dict,key=dict.get)

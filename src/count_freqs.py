# Makes one pass over our index to count total term occurrences for QL models

import _pickle as pickle
with open("../data/INVERTED_INDEX_NOSTOP.pkl", "rb") as file:
    inverted_index = pickle.load(file)

TERM_FREQ = {}

total_tokens = 0
for key in inverted_index:
    posting = inverted_index[key]
    term_freq = 0
    for item in posting:
    	total_tokens += item[1]
    	term_freq += item[1]
    TERM_FREQ[key] = term_freq


print(total_tokens)
pickle.dump(TERM_FREQ, open('../data/TERM_FREQ_NOSTOP.pkl', 'wb'))

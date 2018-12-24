# Project
Build an IR-system on the classic TREC CDs 4 and 5.

## Research Question
To what extent do different techniques improve the performance of an information retrieval system?
1. Which individual technique increases performance the most?
2. Which combination of techniques gives the best results?

## Design
Implementation will be done in Python 3.6.
We aim to implement to following:

1. Inverted index
2. TF-IDF weighting
3. Pivoted length normalization 
4. Query likelihood
5. Rank fusion

## Implementation ideas
Inverted index (one pass):
1. Stopping
2. Stemming
3. Hashmap (Python dictionary) for each term [(docID, freq)]
4. Document frequency can be added to this hashmap with pair (docFreq, [(docID, freq)])

TF-IDF weighting (Okapi BM25) and Pivoted length normalization:
1. Formula from Wikipedia (or textbook, Zhai)
2. Compute length of each document, and average document length, save to table (apply stopping before length computation?)
3. Variable k, and boolean for pivoted length normalization

Query likelihood:
1. Formula from textbook, Zhai
2. With and without smoothing, and different smoothing factors

We now have two retrieval formulas. Evaluate performance of both, and try to incorporate them into one somehow (BM25 has values [0,∞], while query likelihood has values [0,1]).

If we have time left to do query re-ranking, we need to work out the details.

## Evaluation
Evaluation is working! First `make` in the `trec_eval.9.0` folder, then run with:
```
./trec_eval.9.0/trec_eval -m map -m P.30 ./data/qrels ./outputs/<ranking file>
```

## Final Results
We managed to implement pretty much everything that we wanted to, and with good results.

With the simplest query likelihood (with zero-frequency problem) we get:
```
map                     all     0.0627
P_30                    all     0.0913
```

This is improved massively by adding smoothing.
With Dirichlet prior smoothing and mu=800 we get highest MAP, and for mu=900 we get the highest P30:
```
map                   	all	0.2445
P_30                  	all	0.3039
```

With Jelinek-Mercer smoothing and lambda=0.275 we get the highest MAP, and with lambda=0.2 we get highest P30:
```
map                   	all	0.2217
P_30                  	all	0.2797
```

With Okapi BM25 with k1=0.35 and b=0.625 we get the highest MAP, and with k1=0.4 and b=0.5 we get the highest P_30:
```
map                   	all	0.2448
P_30                  	all	0.3012
```

## Note
This repo contained many more files before when it was private, including the `topics` and `qrels` files, as well as all of our outputs. This made the repo too big, so we reduced it down to the bare mininum, the source code and evaluation software.
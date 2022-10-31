### Statistical Inference of Hidden Markov Models on High Frequency Quote Data


**Model Formulation**
- Construct Hidden Markov Models on the top of the book data
- Observation Vector at $t_i$
  - Bid Size 
  - Offer Size
  - OrderBook Imbalance
  - Spread
- Benchmarks PSG HMM_Normal against HMMLearn's Gaussian HMM which is solved heuristically and via the Viterbi algorithm

  
  Statistical Inference completed as a constrained optimization problem with starting values provided by Baum-Welch Algorithm

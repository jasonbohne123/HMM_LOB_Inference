### Statistical Inference of Hidden Markov Models on High Frequency Quote Data


**Model Formulation**
- Construct Hidden Markov Models on the top of the book data
- Observation Vector at $t_i$
  - Bid Size 
  - Offer Size
  - OrderBook Imbalance
  - Spread

  
  **Statistical Inference**
  - Baum-Welch Algorithm
  - Viterbi Dynamic Programming Algorithm

  **Numerical Results**
  - For each feature HMM was fitted on a single day of high-frequency top of book data 
  - Procedure repeated for entire month of Jan 2020
  - Fitted models using PSG and Hmmlearn were compared
  


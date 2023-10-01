# Exact and Approximate Inference in Bayesian Networks

In this work, we are focusing on inference methods and comparing their effectiveness on certain scenarios.
Inference methods can be categorized in two groups, exact inference and approximate inference methods. Here in
this project, our goal is to implement Likelihood Weighting, Gibbs Sampling and Metropolis Hastings methods,
compare these methods on their accuracy, run times and scaling properties on varying Bayesian Networks (BN).
To make this comparison, we tested these methods on several different BNs with different number of nodes and
network types.

## Usage of the functions

* .\exp.py runs the experiment for LW, GS and MH (with p values of .75, .85 and .95) without evidence. Outputs "experiment_results.txt".
* .\readResults.py needs "experiment_results.txt" file to read results for each method, calculate variances and writes the results to "read_results.txt"
* .\exp2.py runs the experiment for LW, GS and MH (with p values of .75, .85 and .95) with evidence *(randomly selected root and leaf nodes). Outputs "experiment_evidence_results.txt"
* .\readResultsEvidence.py needs "experiment_evidence_results.txt" file to read results for each method, calculate variances and writes the results to "read_evidence_results.txt"
* .\plotResults.py requires "read_results.txt" and "read_evidence_results.txt" files, creates and saves the plots for each test (5 in total) and saves to a .png file with method name.
* .\cpt.py is a helper function to calculate relative effectiveness when CPT Entries are close to 0/1. Averages over each methods for given nodes in BN.


Experiments can be conducted after the creation of 10p, 10d, 25p, 25d, 50p, 50d, 100p and 100d.json BNs with .\gen-bn provided in: https://github.com/TUmasters/gen-bn
Exact inference method is provided in: https://github.com/robger98/exact_inference

## gen-bn

Generates random probabilistic networks with Bernoulli random variables. (Modified with tim_polytree).

## Requirements

Python 3, numpy, (optional) networkx, matplotlib, json, multiprocessing


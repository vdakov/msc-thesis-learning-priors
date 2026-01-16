

## Agenda
- Updates and what I tried out 
- Experiments Discussion 
	- Baseline Comparisons
	- Our PFN's properties 
	- Sensitivities to configurations 
	- ML-II similarities
- Multiple vs. single datasets
- First Stage Evaluation
	- Deadline
	- Paper format 
	- Evaluation format
	- Thesis Commitee
- Next steps
- How do we do our meetings?
## Updates 
1. Got normal PFN running properly:
	- With comparison to nanoTabPFN, and debugging it
	- Initial issues about the noise, how context was passed, etc.
![[Pasted image 20260110222000.png]]
![[Pasted image 20260110222710.png]]
2. Prior-Learning PFN
	- Managed to solve the running issues from last time - the model trains now 
	- For a prior parameter, I create a token in the embedding space, that is appended to the test predictions at inference 
	- Capable of both doing multitask learning where we output 
3. Caching speeds up training double
	- For some reason GPU loading of the files is very fast and still speeds training
4. In the process of unit testing the entire codebase. There are some issues I will mention later, looking for bugs.
5. No DAIC yet, WIP, have been using Kaggle in the meantime.
### Experiments
There have been consisting of trying to learn a lengthscale, and comparing it to the ground truth ML-II estimates
- **Multitask learning:** Move context delimiter to different locations during traning - learn to predict both prior and values: 
	- Prior PFN doesn't learn normal PFN predictions 
	- ![[Pasted image 20260114104754.png]]
- **PFN prediction:** We are doing Fully Bayesian Inference for our hyperparameters, based on our "PFN prior" we training on 
	- PFN Prior: Bernoulli of $p =0.5$ on GP with lengthscale $l \in \{ 0.4, 0.6 \}$ 
	- Predictions: Generate a GP with a given GT lengtscale, try to predict from what value it originates at inference
- **ML-II**:
	- In the single dataset case, I thought this was very similar to ML-II
	- Reason: There is no global information of how the prior varies between other datasets - thus, the expectation of the PFN distribution should align with an ML-II estimate of the value. The distribution around that value is purely specified by the distribution we train our PFN on. 
- **Results of Prior-Learning PFN:**
- ![[Pasted image 20260116081333.png]]
![[Pasted image 20260114122120.png]]
**Above is what it used to be**
- For a long time, it didn't manage to distinguish between - it showed a big preference for 0.4 
	- The speculation was that the most likely value for a lengthscale, given a small data size on a small range 
		- Sampling from a GP with $l=x$ does not guarantee the sample lengthscale will be that 
		- Confirmed by optimizing marginal likelihood with L-BFGS-B - I think this could even be a baseline - 
			- Wouldn't technically such optimization work to learn prior parameters on an arbitrary prior - (even when no closed form exists )
		- To compare to this more, a Bayesian linear regression example where we have a single dataset and are trying to estimate $\mu$ from  $w \sim \mathcal N (\mu, \sigma)$ . A MAP-II estimate (which should align with the ML-II and OLS) will be the center of our distribution
	- Last experiments, I may have found a bug of not passing in the correct output scale and noise at evaluation (not training) - though I think it should be working either way
	- Made distinguishing slightly easier between 0.3 and 0.7
- To train it, I really need a measure of the optimal loss - otherwise I don't know when to stop training 
The final configuration seems to work at 10, 15, 25 points  - in the end it must have been noise 
![[Pasted image 20260116082756.png]]
**Sensitivities of training**
- Number of buckets - efficiency in training
- Sequence length - slowness of training (attention is quadratic, slower even with caching)
	- Expectation is that for a GP, the longer the sequence (both in samples and in range) should converge to the true value - not applicable for every prior though - Bayesian linear regression will not work
- X-range (Currently sample uniformly between -2, 2) - affects lengthscale especially 
- Number of datasets - around 500 000 seems to reach a good loss 

### Multiple datasets 
If we want to predict an actual prior, not just an ML-II over a single dataset, I believe we need multiple datasets.

Think Bayesian Linear Regression - the mean estimate of a single destimate is its OLS estimate, but with multiple datasets, it will be the mean of them - converging to the full prior distribution. It would also give an idea of the variance, and be able to output that as well. 

**Approach:** Embed $N$ dataset directly  as $N$ embeddings- predict prior scale as we do now 

### First Stage Review
When do you want the deadline - 30th Jan? 15th February? Can we schedule greenlight after that too?
- There is the semester break

**Report Structure:**
- Short intro and related work
- Methodology
- Research Questions 
- Early experiment results
**Present report during meeting**
Thesis Commitee: Can I pick whoever I want? Do we pick together

### Meetings
- Keep this time? Switch to biweekly? 

### Next Steps 
- Verify if issue with prior learning is fixed 
- DAIC setup 
- Learning multiple parameters; Learning hierarchical parameters
- First State Evaluation Report Writing
- (Optional) Embedding datasets directly 

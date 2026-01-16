
1. **Updates**
2. **Project Proposal Discussion**
3. **Literature Survey** (Discuss here or last?)
4. **Questions for PFN Prior Learning**

---

## 1. Updates

- **Dropped the idea of learning a general prior (for now).**
	- Neural Processes already do this.
	- The bottleneck seems to be data availability.
- **Focusing on initial experiments + understanding the problem better.**
	- The approach involves using the original PFN code and first learning the lengthscale of a Gaussian Process (GP).
	- This will highlight areas to explore and help formulate better research questions.
	- **Approach:** Induce a prior over the given Bayesian model we are currently using.
- **Goal:** Re-write the proposal by next week.
- **Literature Focus:** PFNs, Embeddings, Multi-Task Learning, and PFN statistical foundations.

---

## 2. Project Proposal

### Updates and Strategy
- **Sticking to learning the prior using PFNs.** I agree; the other direction has no prospects with a generative model. While Neural Processes attempt this, they seem to struggle due to a lack of data, as you mentioned.
- **Goal:** Train a PFN to predict different prior parameters under an assumed parametric model. Take advantage of in-context learning to predict the prior parameters of that model over real datasets.
- **Experimental Design:** I am trying to design the experiment to help me derive research questions regarding the data learning approach. Specifically, I want to focus on:
	- How much data I might need to synthesize.
	- What prior I might need to induce.
	- How to represent hyper-priors.

### Potential Research Questions
- **Q1: What prior should we induce over variables?**
	- *Options seem to be:*
		- Very large uniform distribution (likely very inefficient).
		- Very large Standard Normal Distribution around 0.
		- Uninformative Prior.
		- Cauchy Distribution (can still get arbitrarily high values).
- **Q2: How much data would we need for this?**
	- Essentially, we are looking over datasets from multiple distributions.
	- *Considerations:* Impact on training time and training efficiency.
- **Q3: How can I represent a hyper-prior?**
	- The only method I can think of is outputting a weight vector from the PFN. However, this assumes a linear combination of the kernels. Perhaps I could constrain or regularize it?
	- I am having a hard time seeing how to get around this. In general, does Bayesian Model Averaging not average around the different priors?
- **Q4: How should I modify the embeddings?**
	- *Option A:* **Try to predict one prior at a time.**
		- For example: The training set only has one prior; the test set is just the parameter with which it was generated.
		- *Result:* This creates internal cohesion that might build up to a given prior parameter (closer to supervised meta-learning per dataset).
	- *Option B:* **Train the model to predict over multiple datasets.**
		- Goal: Input multiple datasets as input (all with different parameters for training) and a new dataset with a different parameter for testing.
		- *Result:* No internal cohesion necessarily. We would need to embed parameters as well. This could allow reasoning over a single dataset *for* the dataset.

*Note: If I want to tackle Neural Processes again, I want to do it further down the line.*

### Questions on Feedback Received
- You asked me about a "setting" and gave examples of **Learning Curves, BO, GPs**. I do not necessarily understand this context. We are doing prior learning, correct? So, is our setting not just Bayesian models in general?
- You seemed to be asking about "choosing a prior" (e.g., choosing between a Matern and an RBF kernel). I am a bit stuck on how this could work.
- *Clarification needed:* "Training a PFN to induce a prior may be too costly. It may also be possible to avoid training a PFN altogether if you use a GAN-discriminator approach for this." **Do you see what I mean by this?** (I do not see what you mean by this).

---

## 3. Literature Survey

- Feedback Discussion.
- You mentioned you want to discuss **Fully Bayesian Model Selection** again.
- Do you have any questions for me?

---

## 4. Notes

*(Space reserved for meeting minutes)*
- Is a PFN a valid stochastic process? (Send on Mattermost)
- NanoPFN - https://github.com/automl/nanoTabPFN
- Start with 1 million datasets
- Jeffrey's prior 
- Scale-invariance
- BO Works great in high dimensions - https://arxiv.org/abs/2402.02229
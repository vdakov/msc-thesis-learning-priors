---
tags:
  - meetings
---
# Meeting Agenda

- **Updates**
- **Alignment Between Ideas**
  - Shared Notation
  - Method 1: Learning the PFN Prior with Generative Models
  - Method 2: Learning the Prior with PFNs
  - Planning
- **Literature Survey Questions**: Now or later?

---

# Updates

- Spent time on the proposal, relevant literature, and understanding PFN priors more deeply.
- TabPFN and LCPFN appear tclarify what PFN priors look like.
- Outlined ideas in writing and defined approaches for both directions — these will be included in the proposal.
- Plan: submit proposal by **Friday afternoon** to give time for review.
- I aim to include at least one outlined experiment; the structure is clear.

---

# Idea 1: Meta-Learning the PFN Prior

### TL;DR
If we define a valid stochastic process that generates datasets, we can use it to train a PFN.
If the stochastic process itself is *trained on a task*, then it becomes a **meta-learned prior**, enabling Bayesian inference with an empirically learned prior for any dataset within that task family.

### Inspiration from Generative Modelling
- Define a latent distribution \( p(z) \) to approximate the distribution over tasks \( p(D) \).
- This latent variable model becomes a way to express a meta-learned task prior.

### Value Proposition
- Enables **Bayesian inference without hand-crafted assumptions**.
- Produces an explicit **distribution over tasks**, allowing comparisons (e.g., KL divergence) with assumed priors.
- Compatible with analysis via other meta-learning prior approaches.

### Connection to Neural Processes - It has been done already?
- Neural Processes (NPs) define a stochastic process over datasets using a parameterized neural network.
- They resemble latent-variable generative models and support ICL-like behavior.
- They effectively contain a meta-learned prior.

### Open Questions
- NPs are not widely used, especially not for classification — unclear why.
- This idea has value but may not become the main focus.

---

# Idea 2: Using PFNs to Meta-Learn Other Priors

### Motivation
Work such as LCPFN and PFN4BO uses **hand-tuned parametric priors**, e.g.  

$$\mathcal{U}(-0.6, 0.6)$$ 

Can we **empirically learn** the numerical values governing these priors via PFNs?

### Core Idea
- The PFN prior depends both on the **parametric form** and the **parameter values**.
- We might use **ICL over synthetic datasets** (generated with varying parameter values) to infer those values empirically.

### Example: GP Lengthscale \( \ell \ )
1. Assume an uninformative base prior for GP hyperparameters.
2. Sample datasets:  
   $$
   D_t \sim p(D) \quad\text{from GPs with random lengthscales}
   $$ 
3. Train a PFN to predict the dataset from the prior.
4. During ICL:
   - Feed multiple GP datasets as context.
   - Ask PFN to infer the underlying lengthscale distribution.
5. Averaging over tasks may yield an approximate learned prior.

### Challenges
- The setup risks becoming *recursive*: attempting to use a PFN trained on a prior to infer that same prior.
- Need a principled framework for “uninformative” hyperpriors.
- Most importantly, I need to define how my labels work - one-shot, zero-shot
- I do not want to input priors into context
- Encoding whole datasets as a single point and their lengthscale as output
- I always end up at a definition that ends up inputting priors 

---

# Next Steps

1. **Finish Proposal**
   - Integrate both ideas and outline at least one experiment.

2. **Clarify Idea 1**
   - Why aren’t Neural Processes more widely used?
   - Are there technical limitations (expressivity, scaling, likelihood quality)?

3. **Plan Experiments**
   - Consider LCPFN or TabPFN as starting points.
   - Recommendations or guidance would be helpful.

4. **Resolve Confusion in Idea 2**
   - How to structure ICL to extract priors without falling into recursive definitions?
   - Need a more formal treatment of the “uninformative prior” assumption.

5. **Review Theory on Uninformative Priors**
   - Ensure the theoretical footing is clear.

---


**Notes:**
- generativve models incorporate prior knowledge as well 
- address the computation - justify it! 
- 
---
tags:
  - meetings
---
### Plan:
- Discuss brief updates 
- Questions from me - theoretical and logistics
- Questions for me

### **Updates:** 
- MARE should be all good 
- Started writing proposal 
- Will present Pre-Trained GP Paper - Maybe my Literature Review? 
- Setup Repository + Teams + HackMD - You should See them
- PFN Revision - Reimplemented assignments - making my way through literature - same setup as literature survey 
	- Zotero + Table
### **Questions:**
#### Logistics Questions: 
- I currently have included survey time within the Proposed Schedule?
- Do you want a presentation every time? No experiments, just questions currently  - if I have experiments and results - then do I think it is appropriate
- Struggling with later schedule of project
#### Theoretical Questions: 
**General:**
- Learning the noise in a PFN prior? I understand it may slow down convergence, but would noise always be symmetric for example?
- Would you say that the input distribution matters here $p(x)$ for the dataset distribution. Can I keep it the uniform hypercube for example? What is its impact? Couldn't really find it in literature
**Ideas**
Simple baseline solution - wouldn't meta-learning explicit processes be learning the PFN prior as well. For example meta-learning a GP, a BNN or a Linear Regression and then sampling from them. I saw it is possible to meta-learn all of them already. 
- **Minimum Viable Thesis** - Comparing Various Ways to Meta-Learn a PFN via meta-learning a GP?**
*Initial idea for a solution:* meta-learn a generative model
- PFN prior is a probability distribution over functions - any valid stochastic process should work
- Was thinking about nonparametric and maybe placing some Parzen window over them? 
- Need to survey generative modelling methods 
- Wonder if that is theoretically similar to ELBO methods same way pre-training GPs is similar to ML-II 
- You mentioned that if we don't specify a synthetic prior, and implicitly learn it from data, this is just normal in-context learning over multiple datasets. If we manage to find such a generative model based on the dataset, wouldn't this get the same result? As one is a probability distribution over the same thing? 

*When am I Bayesian?*  - Do you have literature on how to ensure my methods. How do I ensure I make coherent stochastic processes?

### Notes
- Planning of thesis can be adjusted
-  Start writing experiments during proposal writing 
- Research questions  - shift from "can" to "how" - focus on insight from questions 
	- PFN papers go in direction of speeding things up 
	- Do meta-learning questions 
- Propose multiple research questions - have them at least partially done by the first evaluation
	- at least one question done by then 
- In introduction - explain what the problem is in learning the prior 
- MotherNet paper 


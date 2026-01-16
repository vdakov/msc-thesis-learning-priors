## Agenda
- Status and Implementation Questions
- Research Questions Formulation
- Planning

---

## Updates
- **Implementation Status:** I have an implementation, but it is bugged—I can't get out the results for it yet.
    - I didn't use **NanoTabPFN**, as I wanted a more modular implementation. This probably backfired.
    - Re-implementing the PFN from scratch and to learn a Gaussian Process (GP) took the majority of the time; now trying to get the extension running.
    - Currently have some issues: the model is training but I have some GPU issues—can't get out a proper sample yet.
- **Training Challenges:** CUDA issues, hyperparameter sensitivities.
    - Learning rate, warmup epochs, and optimizer seem particularly important.
    - Experienced a collapse many times.
- **Modularity:** Implementation with the intention of being modular so that I can easily swap hyperpriors and transformer implementation.

---

## General Questions
- **Validation Sets:** What is the point of a validation set when training a Prior-Fitted Network (PFN)? We tune on new data anyway? Is it just to keep track of a single piece of data?
- **Underfitting:** Can we avoid it? Predictions do not pass through context perfectly.
- **Embeddings:** Custom embedding per priors seems to be working.
- **Training Improvements:** How to train better—incorporating context and extrapolation too?
    - Initially, I was thinking I want to accumulate loss only on prior parameters like $\ell$.
    - I am worried PFN proper PFN training might be lost due to not needing to extrapolate on training samples as much, and the loss not affecting the gradient there.

---

## Research Questions
- **How does the prediction of prior parameters affect training compared to the prediction of test samples for Prior-Fitted Networks?**
    - Training might be faster, or with fewer datasets.
    - Similar to supervised learning 
- **What training strategy needs to be selected for a Prior Learning PFN?**
    - As PFNs are sensitive in general, it would be worth exploring which one works here, and whether it is different.
    - Specifically, the **BarDistribution**  and the **number of datasets** seems to be important due to support and the like.
- **How can prior parameters be best embedded into a PFN?**
- **How can an efficient prior be induced over different types of parameters?**
- **How does misspecifying the parametric format, but fitting it with our PFN, affect performance?**
    - The approach we are proposing still requires us specifying the distribution we would be using as a prior.
    - How does that compare if misspecified? Would our approach still be viable?
        - *Example:* Think the true prior being a Gaussian $\mathcal{N}(0, 2)$, but we assume a uniform one that ends up being $\text{Uniform}(-3, 3)$, thus covering the majority of the real one.
        - We are not really limited as we can have unlimited support on our values.
- **How does PFN prior learning compare to ML-II methods for learning the prior?**
    - Pre-Trained GPs, Linear Regression—good comparison due to having explicit pre-training solutions.

---
## Planning
- Finish lengthscale ($\ell$) implementation.
- Re-submit proposal.
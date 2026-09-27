candidate_pairs.tsv is part of your final submission

Blocking has to scale. Amazon resolves business entities across billions of records, so comparing every record with every other one is not an option. Your blocking / candidate-generation step must cut the search space to a small candidate set per Source 1 entity.
Candidate generation counts toward the final ranking. We will review your candidate_pairs.tsv and the code that produces it when deciding final rankings, alongside your matching_results.tsv score. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard.
Please make sure to go through the problem statement carefully and review all the requirements, guidelines, and submission details before getting started.

Best of luck to all the teams. 

Best Regards,
Team Amazon

• **Understand the problem statement** and evaluation metric before writing code.\
 • **Build a reliable baseline first.** Move to fine-tuned or newer models only after understanding where the baseline fails.\
 • **Identify the required modality** and explore suitable pre-trained models on Hugging Face.\
 • **Divide experiments among teammates:** architectures, loss functions, preprocessing, post-processing, and validation strategies.\
 • **Reproduce every promising result** locally before submitting.\
 • **Avoid overfitting to the public leaderboard.** The private leaderboard is what ultimately matters.\
 • **Track every experiment**, fix random seeds, and keep the final pipeline reproducible.\
 • Leave enough time for **ensembling, inference optimization, documentation,** and final submission checks.

 GPU access can be a significant advantage. The challenge includes guidance around AWS Free Tier and compute credits. If additional compute is required, platforms such as **Jarvislabs.ai** provide A100 access at relatively affordable hourly rates.

 Most importantly, **do not chase complexity from the beginning.** A well-validated simple approach can outperform an advanced model backed by a weak experimental setup.
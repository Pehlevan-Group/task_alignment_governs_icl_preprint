# ICL-structured-data

Mary Letey 2025

Code reproducing figures and experiments for  

**_Pretrain–Test Task Alignment Governs Generalization in In-Context Learning_**

In-context learning (ICL) is a central capability of Transformer models, but the structures in data that enable its emergence and govern its robustness remain poorly understood. In this work, we study how the structure of pretraining tasks governs generalization in ICL. Using a solvable model for ICL of linear regression by linear attention, we derive an exact expression for ICL generalization error in high dimensions under arbitrary pretraining–testing task covariance mismatch. This leads to a new alignment measure that quantifies how much information about the pretraining task distribution is useful for inference at test time. We show that this measure directly predicts ICL performance not only in the solvable model but also in nonlinear Transformers. Our analysis further reveals a tradeoff between specialization and generalization in ICL: depending on task distribution alignment, increasing pretraining task diversity can either improve or harm test performance. Together, these results identify train-test task alignment as a key determinant of generalization in ICL.

_Mary I. Letey, Jacob A. Zavatone-Veth, Yue M. Lu, Cengiz Pehlevan_

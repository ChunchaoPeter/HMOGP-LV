# HMOGP-LV

This repository contains the implementation of Hierarchical Multi-output Gaussian Processes with Latent Information model. The entire code is written in Python and highly depends on GPflow package.

Multi-output Gaussian processes (MOGPs) have been introduced to deal with multiple tasks by exploiting the correlations between different outputs. Generally, MOGPs models assume a flat correlation structure between the outputs. However, such a formulation does not account for more elaborate relationships, for instance, if several replicates were observed for each output (which is a typical setting in biological experiments). This paper proposes an extension of MOGPs for hierarchical datasets (i.e. datasets for which the relationships between observations can be represented within a tree structure). Our model defines a tailored kernel function accounting for hierarchical structures in the data to capture different levels of correlations while leveraging the introduction of latent variables to express the underlying dependencies between outputs through a dedicated kernel. This latter feature is expected to significantly improve scalability as the number of tasks increases. An extensive experimental study involving both synthetic and real-world data from genomics and motion capture is proposed to support our claims.

The model is implemented based on GPflow 2.1.3 or GPflow 2.1.4.

One example is provided in Experiment_demo.

In the Experiment_demo folder, we run python main.py --cfg Synthetic_dataset.yaml

We can change the parameter in Synthetic_dataset.yaml, e.g., adding your own path in there.
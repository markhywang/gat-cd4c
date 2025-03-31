# IMPORTANT: YOU MUST UNZIP THE FILE AND RUN main.py FROM INSIDE gat-cd4c.


# Graph Attention Network for Drug-Target Interaction Prediction

Drug-Target Interaction (DTI) prediction is an essential process in the development of new drugs. However, the existing drug development process relies on experimental verification, which requires high costs and long development times. Recently, artificial intelligence (AI)-based molecular graph analysis has been playing an important role in the new drug screening process, and in particular, Graph Neural Networks (GNNs) have been attracting attention as a powerful tool for effectively learning the structural information of molecules.

The CandidateDrug4Cancer dataset contains 29 cancer-related target proteins, 54,869 drug molecules, and a total of 73,770 drug-protein interaction data. Molecules are expressed in the form of graphs, with atoms as nodes and chemical bonds as edges. The goal of this study is to analyze these molecular graphs and predict whether a specific drug is likely to have a significant biological interaction with a cancer target protein.

**In this study, our objective is to verify whether Graph Attention Networks (GATs) show improved DTI prediction performance compared to existing traditional techniques (e.g., Morgan Fingerprints + XGBoost).** We will implement a GAT with PyTorch. Furthermore, we will compare our results against existing molecular fingerprint-based techniques, and analyze whether the GAT model provides higher prediction accuracy. Lastly, we will graphically present the efficacy of a given drug (as evaluated by our GAT model).

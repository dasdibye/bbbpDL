# bbbpDL
DL model for Blood-Brain-Barrier Permeability (BBBP) Prediction.
There are several .py files corresponding to various training sets and test sets that we have used for the purpose of this work. These are the following: 
  a. bbbp.mic.lightbbb.py that uses y_test_indices.mod.csv of 7000+ compounds for train/test/validation ( for LightBBB results )
  b. bbbp.mic.moleculenet.py that uses BBBP.csv of 2000+ compounds for train/test/validation ( for Moleculenet results )
  c. bbbp.mic.mic.gupta.py that uses new_full_bbbp_trng.csv for train/validation and gupta_test.csv for test ( for Gupta et al.'s results )
  d. bbbp.mic.mic.ghosh_cns.py that uses new_full_bbbp_trng.csv for train/validation and ghosh_cns_test.csv for test ( for Ghosh et al.'s results )
  e. bbbp.mic.mic.ghosh_non_cns.py that uses new_full_bbbp_trng.csv for train/validation and ghosh_non_cns_test.csv for test ( for Ghosh et al.'s results)
  

To run you would also need the file model_300dim.pkl for the 300-dim Mol2vec encoding. The link for that file is here: https://github.com/samoturk/mol2vec_notebooks/blob/master/Notebooks/model_300dim.pkl

Relevant papers:
  1. LightBBB: Computational prediction model of blood-brain-barrier penetration based on LightGBM
     Bilal Shaker, Myeong-Sang Yu, Jin Sook Song, Sunjoo Ahn, Jae Yong Ryu, Kwang-Seok Oh, and Dokyun Na, Bioinformatics, 2020
  2. THE BLOOD-BRAIN BARRIER (BBB) SCORE
     Mayuri Gupta, Hyeok Jun Lee, Christopher J. Barden, and Donald F. Weaver, Journal of Medicinal Chemistry, 2019
  3. Technically Extended MultiParameter Optimization (TEMPO): An Advanced Robust Scoring Scheme To Calculate Central Nervous System Druggability and Monitor Lead Optimization
     Arup K. Ghose, Gregory R. Ott, and Robert L. Hudkins, ACS Chemical Neuroscience, 2017

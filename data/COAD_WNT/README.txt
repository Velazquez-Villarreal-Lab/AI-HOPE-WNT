In our study of Colorectal Adenocarcinoma with WNT pathway information (COAD
WNT)  patient samples, we assessed several key attributes, including SUBTYPE, AGE, SEX, ETHNICITY, RACE, and TUMOR_STAGE.

TMB_status is the tumor mutational burden (TMB) status analyzed specifically for nonsynonymous mutations per million bases (Mb), referred to as TMB_NONSYNONYMOUS.
However, there is currently no consensus in the definition of TMB cutoffs for patient stratification. Here, we follow that the recent tissue-agnostic US FDA approval of pembrolizumab defined elevated TMB as being ≥10 mutations/Mb for TMB_status.

MSI_status indicates the presence of microsatellite instability (MSI). To detect MSI in cancer, we use the MANTIS score (MSI_SCORE_MANTIS), with a recommended cutoff threshold of 0.4.

Genetic mutations (SNP/INDEL) were examined across a range of critical oncogenes (>5% in the cohort), including APC, TP53, KRAS, PIK3CA, FAT4, LRP1B, FBXW7, PCLO, ATM, SMAD4, AMER1, SOX9, KMT2D, BRAF, KMT2B, ARID1A, TCF7L2, PTPRT, FAT1, KMT2C, RNF213, RELN.
Each selected gene is associated with four summarized data attributes:

*_mutation_status: A binary indicator showing whether the gene is mutated.(1:mutated; 0:otherwise)
*_mutation_effect: The functional effect of the mutations.
*_mutation_Amino_Acid_Change: The amino acid change resulting from the mutations.
*_mutation_PolyPhen: A prediction of whether the mutation is deleterious.(benign, possibly_damaging or probably_damaging)

In these attributes, "*" represents the gene name (e.g., FAT1_mutation_status would indicate whether the FAT1 gene is mutated).

Copy Number Variation (CNV) statuses were assessed across various chromosomal arms, covering both the p and q arms of chromosomes 1 through 22, such as Chr1p_CNV_status, Chr1q_CNV_status, Chr2p_CNV_status, Chr2q_CNV_status, ... Chr22q_CNV_status. 

The treatment information includes TREATMENT_TYPE, specifying the method of treatment (e.g., chemotherapy or immunotherapy), and AGENT, detailing the specific drug used (e.g., pembrolizumab, cisplatin). It also includes MEASURE_OF_RESPONSE, indicating the observed clinical outcome, such as complete response or disease progression. Together, these elements provide insights into treatment efficacy and patient response.
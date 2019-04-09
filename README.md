# Quora-Question-Duplicates
## Setup instruction
* get the dataset and place to data folder.
* pip install -r requirement.txt (Requirment should be updated.)



## Dataset
* You can download data from the https://drive.google.com/open?id=1KtUYCZICmOsCSeFcphui6e7E0SHDtMDn
. Use tamu drive for logging.


## How to use
* Do the inital processing by running. This do the spell check and crude tokenization and save to data folder. Or you can download processed file from the drive link above. 
    * cd models
    * python data_processing.py
   
 ## Word Embedding
 * Check the Embedding_Helper class in the embedding.py
 * The glove model can be downloaded from the drive. 
 

 ## Features
 ### TF-IDF scores with muliple weighting schemes.
 * import tf_idf_scores
 * tf_idf_scores.compute_all_similarities_train_and_test(train_df,test_df,data_dir='../data',file_suffix='idf_appended')
 
 See the test function inside tf_idf_scores.py for the example usage.




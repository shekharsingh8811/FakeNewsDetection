# FakeNewsDetection
Project is to create a Fake news classifier by providing it URLs for which: a) there is at least one tweet, and b) where the title and content of the URL is available.

We have built on certain aspects of this paper: Castelo, S., Almeida, T., Elghafari, A., Santos, A., Pham, K., Nakamura, E., & Freire, J. (2019). A Topic-Agnostic Approach for Identifying Fake News Pages. San Francisco, CA, USA: International World Wide Web Conference Committee.

We did not crawl websites to extract HTML. Instead we used this Github repo (github.com/several27/FakeNewsCorpus). It already had the Title and Content for various URLs and these URLs were tagged as being Fake or Reliable (among other tags). There were around 2.0 million "Reliable" and 0.9 million "Fake" URLs. We selected approx. 5,500 data points from these two categories to create a dataset of approx. 11,000 URLs for which feature extraction was done.

Features belong to 4 groups: Morphological, Psychological, Twitter and Readability.

The accuracy values for all models in the report are only for the non-normalized features values being used for model training and then the 20% test data being applied.

################################################################################################ DATA files uploaded in the data folder:

Files with only regular features and NO normalized values. 1a) All_Features_No_Norm.csv -- contains all features 1b) M_Features_No_Norm.csv -- contains only the Morphological features 1c) L_Features_No_Norm.csv -- contains only the Psychological features 1d) T_Features_No_Norm.csv -- contains only the Twitter features 1e) R_Features_No_Norm.csv -- contains only the Readability features 1f) MLR_Features_No_Norm.csv -- contains all features EXCEPT Twitter features

Files with ONLY normalized values. 2a) All_Features_Norm.csv -- contains all features 2b) M_Features_Norm.csv -- contains only the Morphological features 2c) L_Features_Norm.csv -- contains only the Psychological features 2d) T_Features_Norm.csv -- contains only the Twitter features 2e) R_Features_Norm.csv -- contains only the Readability features 2f) MLR_Features_Norm.csv -- contains all features EXCEPT Twitter features

Note on the normalized values: Used MinMax scalar. For Pyschological, Morpohological and Readability: all feature values were set to -1 before the computed values were inserted. If the value was not computed then -1 is reatained. Similarly, for Twitter: some features values (e.g. span, average time between tweets, etc.) are only meaningful if there is more than one tweet. So these values are set as -1 when that particular URL had only one tweet.

Note that each of the data files have the Domain, URL and UrlType at the beginning followed by the actual feature data. The number of features for each group are:

Feature Group -- Number of Features (non-Normalised)
Morphological -- 50
Psychological -- 93
Twitter -- 22
Readability -- 18
Total -- 183
################################################################################################

CODE files uploaded:

scriptExtractTweets1.py Takes an input file for the URLs and extracts all tweets. Uses command line parameters.

scriptCreateReadabilityFeatures1.py Uses a file with URL + Title + Content and processes and creates the Readability features.

demoRunFakeNews1.ipynb Logic: 3a) Starts with the input file containing the URL, Title and Content on which Feature Extraction is to be performed. Broadly, the logic is to extract tweets. Then using this newly created tweets file and the input file, processing is done to create the four features files (one for each group). Then combines the individual feature files into one file containing all the features. Note that this combined file has the regular set of features and normalised (using MinMax scaler) values. 3b) Using the All_features file as input now, further subsets of the data are created. These are for excluding and including the normalized values, and for the feature group to be kept in the data subset. 3c) Thus there will be 12 files: All_Features_WITH_norm.csv , All_Features_NO_norm.csv , M_Features_WITH_norm.csv , M_Features_NO_norm.csv , L_Features_WITH_norm.csv , L_Features_NO_norm.csv , T_Features_WITH_norm.csv , T_Features_NO_norm.csv , R_Features_WITH_norm.csv , R_Features_NO_norm.csv , MLR_Features_WITH_norm.csv , MLR_Features_NO_norm.csv. 3d) For Psychological features: we used a paid tool called Linguistic Inquiry and Word Count (LIWC 3 , Version 1.3.1 2015) for this. We uploaded the Title+Content as a csv file and it returned the feature values. These we combined manually into the temp_L_features.csv.

modelTrainingAndCheckAccuracy_without_FeatureSelection.ipynb Logic: 4a) Train:Test subsets created 80:20 ratio. Then all the features in the input file are used to train models for Decision Tree (Gini and Entropy), KNN, Logistic Regression, Random Forest, SVM and XG-Boost.

modelTrainingAndCheckAccuracy_with_FeatureSelection.ipynb 5a) Feature Selection is done using scikit-learn's FeatureSelection method with estimator as Random Forest. The default settings are used. All the input data is used for this step. 5b) Now, only the important features are retained and split into Train:Test subsets with 80:20 ratio. 5c) These split subsets used to train each model as before.

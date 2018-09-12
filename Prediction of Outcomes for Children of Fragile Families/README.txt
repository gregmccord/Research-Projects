-------------------------------------------------
COS 424: Fragile Families Challenge (Spring 2018)
Author: Gregory McCord
-------------------------------------------------

-classificationDiscrete.py, called as follows
    python classificationDiscrete.py [eviction/layoff/jobTraining]
        [lr/mnb/gnb/svm/rf] [all/both/question/constructed]'
This program creates and tunes a model using the specified parameters (response variable or goal,
dataset, and classifier type or model) and outputs the model for later use as:
'model_%s_%s_%s.pkl' % (model,goal,dataset)

-predict.py, called as follows
    python predict.py [lr/mnb/gnb/svm/rf] [all/both/question/constructed]
This program predicts outcomes for all training and test cases using the specified model and outputs
them in prediction.csv

-topTerms.py, called as follows
    python topTerms.py [all/both/question/constructed]
This program outputs the most influential features to terminal for each of the discrete response
variables based on the Random Forest model with 500 trees trained on the specified dataset.
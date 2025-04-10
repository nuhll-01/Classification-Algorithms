# sources:
#   - Assignment 1 - Slides
#   - Assignment 2 - Slides
#   - https://www.w3schools.com/python/pandas/pandas_dataframes.asp
#   - https://www.w3schools.com/python/numpy/numpy_creating_arrays.asp
#   - https://www.datacamp.com/tutorial/decision-tree-classification-python
#   - https://www.kaggle.com/code/abdokamr/cross-validation-hyperparameters-tuning
#   - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html

from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
import pandas as pd


def main():
    # fetch and load the dataset
    dataset = datasets.fetch_openml(data_id=4534)

    data = dataset.data  # extract the data from the dataset
    data_frame = pd.DataFrame(data)  # create the DataFrame
    categorical_cols = data_frame.select_dtypes(include=['object', 'category']).columns  # select categorical columns
    column_transformer = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False),
                                             categorical_cols)], remainder="passthrough")
    processed_data = column_transformer.fit_transform(dataset.data)
    # create a DataFrame object by using the processed data (which is a numpy array)
    vir_new_data = pd.DataFrame(processed_data,
                                columns=column_transformer.get_feature_names_out(), index=dataset.data.index)

    # at this point in the program, the data should now have been processed into numerical values

    # classification algorithms:
    #       - Decision Tree Classifier
    #       - K-Nearest Neighbor Classifier
    #       - Multi-nominal Naive-Bayes Classifier
    #       - Logistic Regression Classifier
    #       - Dummy Classifier

    print('\n- regular evaluations -')

    # Multi-nominal Naive-Bayes Classifier
    naive_bayes_model = MultinomialNB()  # Naive-Bayes Object
    nb_test_scores = cross_validate(naive_bayes_model, vir_new_data, dataset.target, cv=10, scoring='accuracy')
    print('\nMultinomial Naive Bayes Model Accuracy: ')
    print('Test Scores: ', nb_test_scores['test_score'].mean())

    # Decision Tree Classifier
    decision_tree_model = DecisionTreeClassifier(min_samples_leaf=10)  # Decision Tree Object
    tree_test_scores = cross_validate(decision_tree_model, vir_new_data, dataset.target, cv=10, scoring='accuracy')
    print('\nDecision Tree Model Accuracy: ')
    print('Test Scores: ', tree_test_scores['test_score'].mean())

    # K-Nearest Neighbor Classifier
    knn_model = KNeighborsClassifier(n_neighbors=8)  # K-Nearest Neighbor Object
    knn_test_scores = cross_validate(knn_model, vir_new_data, dataset.target, cv=10, scoring='accuracy')
    print('\nK-Nearest Neighbor Model Accuracy: ')
    print('Test Scores: ', knn_test_scores['test_score'].mean())

    # Logistic Regression
    logistic_regression_model = LogisticRegression()  # Logistic Regression Object
    logistic_test_scores = cross_validate(logistic_regression_model, vir_new_data,
                                          dataset.target, cv=10, scoring="accuracy")
    print('\nLogistic Regression Model Accuracy: ')
    print('Test Scores: ', logistic_test_scores['test_score'].mean())

    # Dummy Classifier
    dummy_model = DummyClassifier()
    dummy_test_scores = cross_validate(dummy_model, vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nDummy Classifier Model Accuracy: ')
    print('Test Score: ', dummy_test_scores['test_score'].mean())

    # at this point in the program, we'll now evaluate each of their bagged versions

    print('\n- bagged evaluations -')

    # Multi-nominal Naive-Bayes Classifier (bagged)
    bagged_naive_bayes_model = BaggingClassifier(estimator=naive_bayes_model)
    bagged_naive_bayes_test_scores = cross_validate(bagged_naive_bayes_model,
                                                    vir_new_data, dataset.target, cv=10, scoring='accuracy')
    print('\nMultinomial Naive Bayes Model Accuracy (bagged): ')
    print('Test Scores: ', bagged_naive_bayes_test_scores['test_score'].mean())

    # Decision Tree Classifier (bagged)
    bagged_decision_tree_model = BaggingClassifier(estimator=decision_tree_model)
    bagged_decision_tree_test_scores = cross_validate(bagged_decision_tree_model,
                                                      vir_new_data, dataset.target, cv=10, scoring='accuracy')
    print('\nDecision Tree Classifier Model Accuracy (bagged): ')
    print('Test Scores: ', bagged_decision_tree_test_scores['test_score'].mean())

    # K-Nearest Neighbor Classifier (bagged)
    bagged_knn_model = BaggingClassifier(estimator=knn_model)
    bagged_knn_test_scores = cross_validate(bagged_knn_model,
                                            vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nKNN Classifier Model Accuracy (bagged):')
    print('Test Scores: ', bagged_knn_test_scores['test_score'].mean())

    # Logistic Regression (bagged)
    bagged_logistic_regression_model = BaggingClassifier(estimator=logistic_regression_model)
    bagged_logistic_test_scores = cross_validate(bagged_logistic_regression_model,
                                                 vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nLogistic Regression Model Accuracy (bagged):')
    print('Test Scores: ', bagged_logistic_test_scores['test_score'].mean())

    # Dummy Classifier (bagged)
    bagged_dummy_model = BaggingClassifier(estimator=dummy_model)
    bagged_dummy_test_scores = cross_validate(bagged_dummy_model,
                                              vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nDummy Classifier Model Accuracy (bagged):')
    print('Test Score: ', bagged_dummy_test_scores['test_score'].mean())

    # at this point in the program, we'll now evaluate each of their boosted versions - (except two of them)

    print('\n- boosted evaluations -')

    # Multi-nominal Naive-Bayes Classifier (boosted)
    boosted_naive_bayes_model = AdaBoostClassifier(estimator=naive_bayes_model)
    boosted_naive_bayes_test_scores = cross_validate(boosted_naive_bayes_model,
                                                     vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nMultinomial Naive Bayes Model Accuracy (boosted): ')
    print('Test Scores: ', boosted_naive_bayes_test_scores['test_score'].mean())

    # Logistic Regression (boosted)
    boosted_logistic_regression_model = AdaBoostClassifier(estimator=logistic_regression_model)
    boosted_logistic_test_scores = cross_validate(boosted_logistic_regression_model,
                                                  vir_new_data, dataset.target, cv=10, scoring='accuracy')
    print('\nLogistic Regression Model Accuracy (boosted):')
    print('Test Scores: ', boosted_logistic_test_scores['test_score'].mean())

    # Dummy Classifier (boosted)
    boosted_dummy_model = AdaBoostClassifier(estimator=dummy_model)
    boosted_dummy_test_scores = cross_validate(boosted_dummy_model,
                                               vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nDummy Classifier Model Accuracy (boosted):')
    print('Test Score: ', boosted_dummy_test_scores['test_score'].mean())

    # Random Forest Classifier
    print('\n- Random Forest Evaluation -')
    random_forest_model = RandomForestClassifier()
    random_forest_test_scores = cross_validate(random_forest_model,
                                               vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nRandom Forest Classifier Model Accuracy: ')
    print('Test Score: ', random_forest_test_scores['test_score'].mean())

    # Voting Ensemble
    voting_model = VotingClassifier([('naive-bayes', naive_bayes_model),
                                     ('decision-tree', decision_tree_model),
                                     ('k-nearest neighbors', knn_model),
                                     ('logistic regression', logistic_regression_model),
                                     ('dummy classifier', dummy_model),
                                     ('random forest', random_forest_model)])
    voting_model_scores = cross_validate(voting_model, vir_new_data, dataset.target, cv=10, scoring="accuracy")
    print('\nVoting Ensemble Accuracy: ')
    print('Test Score: ', voting_model_scores['test_score'].mean())


if '__main__' == __name__:
    main()

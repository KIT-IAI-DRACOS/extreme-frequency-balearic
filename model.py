from sklearn.utils.class_weight import compute_sample_weight  
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve  
import xgboost as xgb  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import joblib  
import shap
  
def train_and_evaluate(X_train, y_train, X_validate, y_validate, X_test, y_test):  
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)  
      
    # Hyperparameter tuning using GridSearch  
    param_grid = {  
        'max_depth': [9, 10, 11, 12],  
        'gamma': [0.2, 0.15, 0.25],  
        'subsample': [0.7, 1.0, 0.5],  
        'colsample_bytree': [0.7, 0.6, 0.8],  
        'learning_rate': [0.06, 0.1, 0.08, 0.12, 0.14],  
        'n_estimators': [250, 200, 300, 350],  
        'reg_alpha': [0],  
        'reg_lambda': [0.1],  
    }  
      
    xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=3)  
    grid_search = GridSearchCV(xgb_clf, param_grid, scoring=['f1_macro', "precision_macro", "recall_macro", "neg_log_loss", "roc_auc_ovr"], refit='neg_log_loss', cv=3, verbose=100)  
    # Uncomment the following lines to perform hyperparameter tuning  
    # grid_search.fit(X_train, y_train, sample_weight=sample_weights)  
    # print("Best parameters:", grid_search.best_params_)  
    # print("Best score:", grid_search.best_score_)  
  
    # Train classifier with best parameters  
    classifier = xgb.XGBClassifier(  
        base_score=0.5, booster='gbtree', early_stopping_rounds=50,  
        objective='multi:softprob', num_class=3, subsample=0.5, scale_pos_weight=2,  
        reg_lambda=0.1, reg_alpha=0, n_estimators=250, min_child_weight=5,  
        max_depth=11, learning_rate=0.04, gamma=0.2, colsample_bytree=0.7,  
        eval_metric=["mlogloss", "auc"]  
    )  
    classifier.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=100, sample_weight=sample_weights)  
  
    # Train baseline model  
    baseline_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)  
    baseline_model.fit(X_train, y_train, sample_weight=sample_weights)  
  
    # Evaluate models  
    evaluate_model(classifier, X_test, y_test, "XGBoost Classifier")  
    evaluate_model(baseline_model, X_test, y_test, "Baseline Logistic Regression")  
  
    # Save models and predictions  
    save_models_and_predictions(classifier, baseline_model, X_train, y_train, X_validate, y_validate, X_test, y_test)  
  
    return classifier, baseline_model  
  
def evaluate_model(model, X_test, y_test, model_name):  
    y_pred = model.predict(X_test)  
    cm = confusion_matrix(y_test, y_pred)  
    report = classification_report(y_test, y_pred)  
    print(f"Confusion Matrix for {model_name}:\n", cm)  
    print(f"Classification Report for {model_name}:\n", report)  
      
    # ROC and Precision-Recall Curves  
    probabilities = model.predict_proba(X_test)  
    plot_roc_curve(probabilities, y_test, model_name)  
    plot_precision_recall_curve(probabilities, y_test, model_name)  
      
    return y_pred  
  
def plot_roc_curve(probabilities, y_test, model_name):  
    n_classes = probabilities.shape[1]  
    fpr = dict()  
    tpr = dict()  
    roc_auc = dict()  
    for i in range(n_classes):  
        fpr[i], tpr[i], _ = roc_curve(y_test == i, probabilities[:, i])  
        roc_auc[i] = roc_auc_score(y_test == i, probabilities[:, i])  
      
    plt.figure()  
    for i in range(n_classes):  
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')  
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title(f'Receiver Operating Characteristic - {model_name}')  
    plt.legend(loc="lower right")  
    plt.show()  
  
def plot_precision_recall_curve(probabilities, y_test, model_name):  
    n_classes = probabilities.shape[1]  
    precision = dict()  
    recall = dict()  
    average_precision = dict()  
    for i in range(n_classes):  
        precision[i], recall[i], _ = precision_recall_curve(y_test == i, probabilities[:, i])  
        average_precision[i] = average_precision_score(y_test == i, probabilities[:, i])  
  
    plt.figure()  
    for i in range(n_classes):  
        plt.plot(recall[i], precision[i], label=f'Class {i} (area = {average_precision[i]:.2f})')  
    plt.xlabel('Recall')  
    plt.ylabel('Precision')  
    plt.title(f'Precision-Recall curve - {model_name}')  
    plt.legend(loc="lower left")  
    plt.show()  
  
def save_models_and_predictions(classifier, baseline_model, X_train, y_train, X_validate, y_validate, X_test, y_test):  
    joblib.dump(classifier, "models/classifier.pkl")  
    joblib.dump(baseline_model, "models/baseline_predictor.pkl")  
  
    # Save predictions and probabilities  
    save_predictions_and_probabilities(classifier, "classifier", X_train, y_train, X_validate, y_validate, X_test, y_test)  
    save_predictions_and_probabilities(baseline_model, "baseline", X_train, y_train, X_validate, y_validate, X_test, y_test)  
  
    # Save datasets  
    joblib.dump(X_test, "data/test_data.lib")  
    joblib.dump(y_test, "data/test_data_label.lib")  
    joblib.dump(X_train, "data/train_data.lib")  
    joblib.dump(y_train, "data/train_data_label.lib")  
    joblib.dump(X_validate, "data/validate_data.lib")  
    joblib.dump(y_validate, "data/validate_data_label.lib")  
  
def save_predictions_and_probabilities(model, model_name, X_train, y_train, X_validate, y_validate, X_test, y_test):  
    # Save predictions and probabilities for test, train, and validation sets  
    datasets = [("train", X_train, y_train), ("validate", X_validate, y_validate), ("test", X_test, y_test)]
    
    for set_name, X, y in datasets:  
        y_pred = model.predict(X)  
        probabilities = model.predict_proba(X)  
        joblib.dump(y_pred, f"data/{set_name}_data_pred_{model_name}.lib")  
        joblib.dump(probabilities, f"data/{set_name}_data_prob_{model_name}.lib")  
  
def compute_shap_values(classifier, X_test, y_test):
    # Create the samples for the shap values

    y_test_df = y_test.to_frame(name='label')
    X_test_df = pd.DataFrame(X_test)

    # Combine X_test and y_test for easier filtering
    test_combined = pd.concat([X_test_df, y_test_df], axis=1)

    # Filter samples for each class
    sample_class_0 = test_combined[test_combined['label'] == 0].sample(75)
    sample_class_1 = test_combined[test_combined['label'] == 1].sample(100)
    sample_class_2 = test_combined[test_combined['label'] == 2].sample(75)

    sample_test_combined = pd.concat([sample_class_0, sample_class_1, sample_class_2])

    sample_X_test = sample_test_combined.drop('label', axis=1)

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(
        classifier, sample_X_test, feature_perturbation='interventional', check_additivity=False)


    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)

    return shap_values



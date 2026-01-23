from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, mean_absolute_error, 
    mean_squared_error, r2_score, root_mean_squared_error
)
from sklearn.utils.multiclass import type_of_target

def calculate_metrics(y_true, y_pred):
    target_type = type_of_target(y_true)
    
    metrics = {
        'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0,
        'confusion_matrix': None, 'classification_report': "",
        "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0
    }

    if target_type == 'continuous':
        
        metrics["MSE"] = mean_squared_error(y_true, y_pred)
        metrics["RMSE"] = root_mean_squared_error(y_true, y_pred)
        metrics["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics["R2"] = r2_score(y_true, y_pred)
    else:
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
       
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0)
    
    return metrics
from Model import(
    KmeansSklearn,
    KmeansScratch,
    BinaryClassificationScratch,
    BinaryClassificationSklearn,  
    DecisionTreeScratch,
    DecisionTreeClassifierSklearn,
    AnomalyDetection,
    LinearRegressionScratch,
    LinearRegressionSklearn,
    MulticlassScratch,
    MulticlassShallowNN,
    RandomForestClassifierSklearn,
    XGBoostClassifierSklearn

)

MODEL_REGISTRY={
    "Clustering":{
    "Kmeans":{
        "sklearn":KmeansSklearn,
        "scratch":KmeansScratch
    }
    },
    "Classification":{
    "Binary_Classification":{
        "sklearn":BinaryClassificationSklearn,
        "scratch":BinaryClassificationScratch
    },
    "Decision_Tree":{
        "sklearn":DecisionTreeClassifierSklearn,
        "scratch":DecisionTreeScratch
    },
    
    "Random_Forest":{
        "sklearn":RandomForestClassifierSklearn
    },
    "XGBoost":{
        "sklearn":XGBoostClassifierSklearn
    },},

    "Anomaly_Detection":{
    "Gaussian Anomaly":{
        "scratch":AnomalyDetection
    },
},
    "Regression":{
    "Linear_Regression":{
        "sklearn":    LinearRegressionSklearn,
        "scratch":LinearRegressionScratch
    }},
    "Multiclass_Classification":{
    "softmax_NN":{
        "shallow_NN":MulticlassShallowNN,
        "scratch":MulticlassScratch
    },}

}
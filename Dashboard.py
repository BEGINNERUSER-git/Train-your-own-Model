from sklearn.model_selection import train_test_split
from Model_Registry import MODEL_REGISTRY
from Accuracy import calculate_metrics
import streamlit as st
import pandas as pd
import numpy as np
from Visualization import Visualization
from Data_Cleaning import DataCleaner
from Model_Training import split_data
from Standardization import Standardization
from model_io import save_model,load_model,list_saved_model
from sklearn.metrics import silhouette_score


page1,page2= st.tabs(["Model Training Dashboard","About"])

with page1:

    st.write("# Model Training Dashboard")
    st.markdown("""
    Welcome to the Model Training Dashboard! This application allows you to upload your dataset, select a machine learning model from our registry, and train it directly within the dashboard. You can also visualize your data and evaluate model performance using various metrics.
    """)
    
    sidebar=st.sidebar
    sidebar.title("Model Dashboard")
    mode = sidebar.radio(
        "Select Mode",
        ["Train New Model", "Use Existing Model"]
    )
    
    if mode == "Use Existing Model":
        sidebar.subheader("Select Existing Model")

        saved_models = list_saved_model()

        if not saved_models:
            sidebar.warning("No saved models available")
            st.stop()

        selected_model = sidebar.selectbox(
            "Available Models",
            saved_models
        )
    if mode == "Use Existing Model":
        if sidebar.button("Activate Model"):
            bundle = load_model(selected_model)

            st.session_state['active_model'] = bundle['model']
            st.session_state['active_preprocessor'] = bundle['preprocessor']

            sidebar.success("Model activated and ready")
    # if mode == "Use Existing Model":
    #     pred_file = sidebar.file_uploader(
    #         "Upload data for prediction (CSV)",
    #         type=["csv"]
    #     )

    #     if pred_file:
    #         st.session_state['prediction_data'] = pd.read_csv(pred_file)

    sidebar.title("Model Training Dashboard")
    sidebar.write("Upload your dataset, select a model, and train it right here!")
    
    sidebar.divider()
 
    
    if mode == "Train New Model":
        if 'data' not in st.session_state:
            uploaded_file = sidebar.file_uploader(
                "Upload your dataset (CSV format)",
                type=["csv"]
            )
            if uploaded_file:
                st.session_state['data'] = pd.read_csv(uploaded_file)
                st.rerun()
            else:
                st.warning("Upload dataset to continue")
                st.stop()


        if sidebar.button("Reset and Upload New File"):
            del st.session_state['data']
            st.rerun()
    

    if 'data' in st.session_state:
        data=st.session_state['data']
    else:
        data=None


    tab1, tab2, tab3,tab4 ,tab5,tab6,tab7= st.tabs(["Data Preview","Cleaning", "Visualization","Standarization" ,"Training","Test Model","Prediction"])
   
    
    with tab1:
        
        st.subheader("Data Preview")

        if mode == "Train New Model":
            if 'data' not in st.session_state:
                st.info("Upload a dataset to preview")
                st.stop()
            head=st.button("Show Data head")
            if head:
                st.dataframe(data.head())
            info=st.button("Show Data Summary")
            if info:
                st.dataframe(data.describe())
            
        # else:  
        #     if 'prediction_data' not in st.session_state:
        #         st.info("Upload data for prediction to preview")
        #         st.stop()
        #     st.dataframe(st.session_state['prediction_data'].head())


    with tab2:
        st.subheader("Data Cleaning Options")
        
        if mode != "Train New Model":
            st.info("Data cleaning disabled for existing models")
            

            
        else:    
            if mode == "Train New Model":
                if 'cleaner' not in st.session_state:
                    st.session_state['cleaner'] = DataCleaner(data)

                cleaner = st.session_state['cleaner']

            if st.button("Remove Duplicates"):
                data=cleaner.remove_duplicates()
                st.session_state['data']=data
                st.dataframe(data.head())
                st.success("Duplicates removed.")
                st.rerun()   
                
            st.divider()

            num_col=st.multiselect("Select Numerical Columns to fill Missing Values:", options=data.select_dtypes(include=[np.number]).columns.tolist())
            strategy=st.selectbox("Select Strategy for Missing Values:", options=['mean', 'median', 'mode'])
            if st.button("Fill Missing Values"):
                if num_col:
                    data=cleaner.fill_missing_values(num_col,strategy=strategy)
                    st.session_state['data']=data
                    st.dataframe(data.head())
                    st.success("Missing values filled.")
            st.divider()
            cat_col=st.multiselect("Select the categorical columns to encode:", options=data.select_dtypes(include=['object']).columns.tolist())
            if st.button("Encode Categorical Variables"):
                if cat_col:
                    data=cleaner.encoding_coategorical(cat_col)
                    st.session_state['data']=data
                    st.dataframe(data.head())
                    st.success(f"Categorical column '{cat_col}' encoded.")
            
            st.divider()
            columns=st.multiselect("Select columns to drop:", options=data.columns.tolist())
            if st.button("Drop Columns"):
                if columns:
                    data=cleaner.drop_columns(columns)
                    st.session_state['data']=data
                    st.dataframe(data.head())
                    st.success(f"ID columns '{columns}' dropped.")
            st.divider()       
            if st.button("Trim String Columns"):
                data=cleaner.data_clipped_str()
                st.session_state['data']=data
                st.dataframe(data.head())
                st.success("String columns trimmed.")
            if st.button("Show Cleaned Data"):
                st.dataframe(data.head())
            st.divider()


            st.subheader("Standard Feature Transformations")
            col=st.selectbox("Select Column for Transformation:", options=data.select_dtypes(include=[np.number]).columns.tolist())
            func=st.selectbox("Select Transformation Function:", options=['Log Transformation','Square Root Transformation ','Square Transformation','Absolute Transformation'])
            if st.button("Apply Transformation"):
                if col and func:
                    data=cleaner.custom_column(col,func)
                    st.session_state['data']=data
                    st.dataframe(data.head())
                    st.success(f"Applied {func} on column '{col}'.")
            


    if mode=='Train New Model':
        shape=sidebar.button("Check shape of your data")
        if shape:
            st.sidebar.info(f"Data Shape: {data.shape}")


    if mode == "Train New Model":
        X = sidebar.multiselect(
            "Select Feature Columns (X)",
            options=data.columns.tolist()
        )
        if Algorithm_Type!="Clustering":
            y = sidebar.selectbox(
                "Select Target Column (y)",
                options=data.columns.tolist()
            )
        else:
            y=None

        st.session_state['X'] = X
        st.session_state['y'] = y


    if mode == "Train New Model":
        X = st.session_state.get('X')
        y = st.session_state.get('y')

    if mode == "Train New Model":
        spilt=sidebar.checkbox("Do you want to spilt the Data?")
        if spilt:
            test_size=sidebar.slider("Select Test Size (as a fraction):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            random_state=sidebar.number_input("Enter Random State (integer):", min_value=0, value=42, step=1)
            if not X or not y:
                st.warning("Please select both feature (X) and target (y) columns from the sidebar to split the data.")
              if Algorithm_Type=='Clustering':
                  X_train=data[X]
                  st.session_state['X_train']=X_train
             else:
                X_train, X_test_value, y_train, y_test_value=split_data(data,feature_columns=X, target_column=y, test_size=test_size, random_state=random_state)
                st.session_state['X_train']=X_train
                st.session_state['y_train']=y_train
            sidebar.success(f"Data split into training and testing sets with test size {test_size}.")
            cv=sidebar.checkbox("Do you want to spilt into Cross Validation set?")
            st.session_state['use_cv'] = cv
            if cv:
                cv_size=sidebar.slider("Select CV set size:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                random_state_cv=sidebar.number_input("Enter CV Random State (integer):", min_value=0, value=42, step=1)
                X_test, X_cv, y_test, y_cv=train_test_split(
                    X_test_value, y_test_value, test_size=cv_size, random_state=random_state_cv
                )
                st.session_state['X_cv']=X_cv
                st.session_state['y_cv']=y_cv
                st.session_state['X_test']=X_test
                st.session_state['y_test']=y_test
            else:
                st.session_state['X_test']=X_test_value
                st.session_state['y_test']=y_test_value
    
    with tab3:
        
        st.subheader("Exploratory Data Analysis")
        if mode == "Train New Model":
            if X and y:
                viz=Visualization(data,X,y)
                if st.button("Show Scatter Plot"):
                    viz.scatter_plot()
                if st.button("Show Histogram"):
                    viz.histogram()
                if st.button("Show Box Plot"):
                    viz.box_plot()
                if st.button("Show Heatmap"):
                    viz.heatmap()
            else:
                st.warning("Please select both feature (X) and target (y) columns from the sidebar to visualize the data.")
        
    with tab4:
        
        st.subheader("Data Standardization")
        

        if mode == "Train New Model":
            if 'scaler_object' not in st.session_state:
                st.session_state['scaler_object'] = Standardization()

            standard=st.session_state['scaler_object']
        else:
           st.info("Standardization is disabled for existing models")
        is_train=mode=='Train New Model'
        applies_to_X_train= st.checkbox("Fit_transform to X_train",disabled=not is_train)
        if applies_to_X_train:
            if 'scaler_object' not in st.session_state:
                st.error("Scaler not initialized. Please reload the page.")
                

            standard = st.session_state['scaler_object']
            if 'X_train' in st.session_state:
                X_train=st.session_state['X_train']
                X_train_scaled=standard.fit_transform(X_train)
                st.session_state['X_train']=X_train_scaled
                st.dataframe(X_train_scaled.head())
                st.success("Standardization applied to X_train.")
            if 'X_cv' in st.session_state:
                X_cv=st.session_state['X_cv']
                X_cv_scaled=standard.transform(X_cv)
                st.session_state['X_cv']=X_cv_scaled
                st.dataframe(X_cv.head())
                st.success("Standardization applied to X_cv.")
            if 'X_test' in st.session_state:
                X_test=st.session_state['X_test']
                X_test_scaled=standard.transform(X_test)
                st.session_state['X_test']=X_test_scaled
                st.dataframe(X_test_scaled.head())
                st.success("Standardization applied to X_test.")
            else:
                st.error("Please split the data and have X_train available before applying standardization.")
            st.session_state['scaled']=True
            # st.rerun()
        
    
 

    
    with tab5:
        if mode != "Train New Model":
            st.info("Switch to Train New Model mode to train models.")

        disabled = mode=='Use Existing Model'

        
       
        st.subheader("Model Registry & Training")
        
        compare_model=st.checkbox("Compare Model",disabled=disabled)
        if compare_model:
            MODEL_SLOT=['Model A','Model B']
        else:
            MODEL_SLOT=['Model A']
        if 'trained_model' not in st.session_state:
            st.session_state['trained_model']={}
        if 'metrics_model' not in st.session_state:
            st.session_state['metrics_model']={}
        if not disabled:
            for slot in MODEL_SLOT:
                st.markdown(f"### {slot}")
                Algorithm_Type=st.selectbox(f"Select Algorithm Type:{slot}", options=list(MODEL_REGISTRY.keys()),key=f"{slot}_algo",disabled=disabled)
                if Algorithm_Type:
                    Model_Type=st.selectbox(f"Select Model Type: {slot}", options=list(MODEL_REGISTRY[Algorithm_Type].keys()),key=f"{slot}_model",disabled=disabled)
                    st.session_state[f"{slot}_Algorithm_Type"]=Algorithm_Type
                    if Model_Type:
                        Implementation_Type=st.selectbox(f"Select Implementation Type: {slot}", options=list(MODEL_REGISTRY[Algorithm_Type][Model_Type].keys()),key=f"{slot}_impl",disabled=disabled)
                        if Implementation_Type:
                            model=MODEL_REGISTRY[Algorithm_Type][Model_Type][Implementation_Type]
                            st.info(f"You have selected: {Algorithm_Type} > {Model_Type} > {Implementation_Type}")
                            st.write("Set Parameters:")
                            params={}
                            if Model_Type=='Kmeans':
                                params['k']=st.number_input("Enter No of clusters: ",min_value=2,max_value=10,value=3,step=1,key=f"{slot}_k")
                                params['max_iters']=st.number_input("Enter Maximum no of iterations: ",min_value=100,max_value=10000,value=500,step=100,key=f"{slot}_kmeans_iters")
                            if (Model_Type=='Binary_Classification')and Implementation_Type=='scratch':
                                params['lr']=st.slider("Slide to set learning rate: ",min_value=0.0001,max_value=0.1,value=0.01,step=0.0001,format="%.4f",key=f"{slot}_lr")
                                params['num_iters']=st.number_input("Enter Maximum no of iterations: ",min_value=100,max_value=10000,value=500,step=100,key=f"{slot}_num_iters")
                                params['threshold']=st.slider("Set classification threshold: ",min_value=0.0,max_value=1.0,value=0.5,step=0.01,format="%.2f",key=f"{slot}_threshold")
                            if Model_Type=='Decision_Tree':
                                params['max_depth']=st.number_input("Enter Maximum Depth of Tree: ",min_value=1,max_value=20,value=5,step=1,key=f"{slot}_dt_max_depth")
                                params['min_samples_split']=st.number_input("Enter Minimum Samples to Split: ",min_value=2,max_value=10,value=2,step=1,key=f"{slot}_dt_min_samples_split")
                                params['min_samples_leaf']=st.number_input("Enter Minimum Samples at Leaf Node: ",min_value=1,max_value=10,value=1,step=1,key=f"{slot}_dt_min_samples_leaf")
                            if Model_Type=='Random_Forest':
                                params['n_estimators']=st.number_input("Enter Number of Trees: ",min_value=10,max_value=500,value=100,step=10,key=f"{slot}_rf_n_estimators")
                                params['max_depth']=st.number_input("Enter Maximum Depth of Trees: ",min_value=1,max_value=20,value=5,step=1,key=f"{slot}_rf_max_depth")
                                params['min_samples_split']=st.number_input("Enter Minimum Samples to Split: ",min_value=2,max_value=10,value=2,step=1,key=f"{slot}_rf_min_samples_split")
                                params['min_samples_leaf']=st.number_input("Enter Minimum Samples at Leaf Node: ",min_value=1,max_value=10,value=1,step=1,key=f"{slot}_rf_min_samples_leaf")
                            if Model_Type=='XGBoost':
                                params['n_estimators']=st.number_input("Enter Number of Trees: ",min_value=10,max_value=500,value=100,step=10,key=f"{slot}_xgb_n_estimators")
                                params['learning_rate']=st.slider("Slide to set learning rate: ",min_value=0.0001,max_value=0.1,value=0.01,step=0.0001,format="%.4f",key=f"{slot}_xgb_lr")
                                params['max_depth']=st.number_input("Enter Maximum Depth of Trees: ",min_value=1,max_value=20,value=5,step=1,key=f"{slot}_xgb_max_depth")
                            if Model_Type=="Multiclass_Classification":
                                params['max_iters']=st.number_input("Enter Maximum no of iterations: ",min_value=100,max_value=10000,value=500,step=100,key=f"{slot}_multi_max_iters")
                                params['lr']=st.slider("Slide to set learning rate: ",min_value=0.0001,max_value=0.1,value=0.01,step=0.0001,format="%.4f",key=f"{slot}_multi_lr")
                            if Model_Type=="Linear_Regression"and Implementation_Type=='scratch':
                                params['lr']=st.slider("Slide to set learning rate: ",min_value=0.0001,max_value=0.1,value=0.01,step=0.0001,format="%.4f",key=f"{slot}_lin_lr")
                                params['max_iters']=st.number_input("Enter Maximum no of iterations: ",min_value=100,max_value=10000,value=500,step=100,key=f"{slot}_linreg_max_iters")
                            train=st.button(f"Train {slot}",key=f"train_{slot}",disabled=disabled)
                            if train:
                                if Algorithm_Type=="Clustering" and Model_Type=="Kmeans":
                                     X_train=st.session_state['X_train']
                                     model_instance=model(**params)
                                     labels=model_instance.fit(X_train)
                                     st.session_state['trained_model'][f"{slot}_model"]=model_instance
                                     st.session_state[f"{slot}_labels"]=labels
                                     score=silhouette_score(X_train,labels)
                                     st.metric(label="Silhouette Score",
                                              value=round(score,4))
                                     st.stop()
                                    
                                X_train=st.session_state['X_train']
                                y_train=st.session_state['y_train']
                                with st.spinner("Training the model..."):  

                                    if params:
                                        model_instance=model(**params)
                                    else:
                                        model_instance=model()
                                    if Model_Type=='Gaussian Anomaly' and 'X_cv' in st.session_state and 'y_cv' in st.session_state:
                                        X_cv=st.session_state['X_cv']
                                        y_cv=st.session_state['y_cv']
                                        model_instance.fit(X_train, y_train,X_cv,y_cv)
                                    else:
                                        model_instance.fit(X_train, y_train)
                                    st.success("Model trained successfully!")
                                    st.session_state['trained_model'][f"{slot}_model"]=model_instance
                                    st.session_state['preprocessor'] = {
                                        'cleaner':st.session_state['cleaner'] ,
                                        'scaler': st.session_state['scaler_object'],
                                        'features': st.session_state['X_train'].columns.tolist()
                                    }
                                    model_name=f"{slot}_{Algorithm_Type}_{Model_Type}_{Implementation_Type}"
                                    save_path=save_model({
                                        "model":model_instance,
                                        "preprocessor":{
                                            'cleaner':st.session_state['cleaner'] ,
                                        'scaler': st.session_state['scaler_object'],
                                        'features': st.session_state['X_train'].columns.tolist()
                                        },
                                        "algorithm_type":Algorithm_Type
                                    },model_name)
                                    st.success(f"Model saved as {model_name}")
                                    use_cv = st.session_state.get('use_cv', False)

                                    if use_cv and 'X_cv' in st.session_state and 'y_cv' in st.session_state:
                                        X_eval = st.session_state['X_cv']
                                        y_eval = st.session_state['y_cv']
                                        eval_label = "Cross-Validation Set"
                                    else:
                                        X_eval = X_train
                                        y_eval = y_train
                                        eval_label = "Training Set"

                                    
                                    model_instance=st.session_state['trained_model'][f"{slot}_model"]
                                    y_eval_pred=model_instance.predict(X_eval)
                                    st.session_state[f'y_eval_pred for {slot}']=y_eval_pred
                                    if Algorithm_Type!="Clustering":
                                        metrics_cv=calculate_metrics(y_eval, y_eval_pred)
                                        st.session_state['metrics_model'][f"{slot}_metrics"]=metrics_cv
                                with st.expander(f"{slot} Model Performance"):
                                    if f'{slot}_metrics' in st.session_state["metrics_model"]:
                                        metrics_cv = st.session_state['metrics_model'][f"{slot}_metrics"]

                                        tab_summary, tab_details = st.tabs(["Summary", "Detailed Metrics"])

                                        with tab_summary:
                                            if Algorithm_Type=='Regression':
                                                st.subheader(f"{eval_label} Performance: {slot}")

                                                
                                                st.write(f"MAE: {metrics_cv['MAE']:.4f}")
                                                st.write(f"MSE: {metrics_cv['MSE']:.4f}")
                                                st.write(f"RMSE: {metrics_cv['RMSE']:.4f}")
                                                st.write(f"R2: {metrics_cv['R2']:.4f}")
                                            else:
                                                st.subheader(f"{eval_label} Performance: {slot}")

                                                st.write(f"Accuracy: {metrics_cv['accuracy']:.4f}")
                                                st.write(f"Precision: {metrics_cv['precision']:.4f}")
                                                st.write(f"Recall: {metrics_cv['recall']:.4f}")
                                                st.write(f"F1 Score: {metrics_cv['f1_score']:.4f}")
                                        

                                        with tab_details:
                                            if Algorithm_Type!='Regression':
                                                st.write("Confusion Matrix:")
                                                st.write(metrics_cv['confusion_matrix'])
                                                st.write("Classification Report:")
                                                st.text(metrics_cv['classification_report'])
                                            else:
                                                st.info("Not Valid for Continuous Values.")

            if Algorithm_Type!='Regression':
                with st.expander("Model Comparison"):            
                    if 'Model A_metrics' in st.session_state["metrics_model"]:
                        st.divider()
                        st.subheader("Model Comparison")
                        col1,col2=st.columns(2)
                        with col1:
                            st.markdown("### Model A Performance")
                            metrics_a=st.session_state['metrics_model']['Model A_metrics']
                            st.write(f"Accuracy: {metrics_a['accuracy']:.4f}")
                            st.write(f"Precision: {metrics_a['precision']:.4f}")
                            st.write(f"Recall: {metrics_a['recall']:.4f}")
                            st.write(f"F1 Score: {metrics_a['f1_score']:.4f}")
                        if 'Model B_metrics' in st.session_state["metrics_model"]:
                            with col2:
                                st.markdown("### Model B Performance")
                                metrics_b=st.session_state['metrics_model']['Model B_metrics']
                                st.write(f"Accuracy: {metrics_b['accuracy']:.4f}")
                                st.write(f"Precision: {metrics_b['precision']:.4f}")
                                st.write(f"Recall: {metrics_b['recall']:.4f}")
                                st.write(f"F1 Score: {metrics_b['f1_score']:.4f}")
                            


    
                            

                    
                                

    with tab6:
        st.subheader("Test Set Evaluation")
        Test_Model=st.button("Evaluate on Test Set",disabled=disabled)
        if Test_Model:
            for slot in MODEL_SLOT:
                if f"{slot}_model" in st.session_state['trained_model']:
                    Algorithm_Type=st.session_state[f"{slot}_Algorithm_Type"]
                    st.markdown(f"### {slot} Test Results")
                    model=st.session_state['trained_model'][f"{slot}_model"]
                    X_test=st.session_state['X_test']
                    y_test=st.session_state['y_test']

                    y_test_pred=model.predict(X_test)
                    if Algorithm_Type=='Clustering':
                        st.write("Clustering models do not have traditional accuracy metrics.")
                        st.subheader("Test Set Performance")
                        labels=model.predict(X_test)
                        st.write("Cluster labels: ")
                        st.write(labels)
                        if hasattr(model,"centroids"):
                            st.write(f"Cluster Centers:")
                            st.dataframe(model.centroids)
                    else:
                        st.subheader("Test Set Performance")
                        metrics_test=calculate_metrics(y_test, y_test_pred)
                        if Algorithm_Type=='Regression':
                            st.write(f"MAE: {metrics_test['MAE']:.4f}")
                            st.write(f"MSE: {metrics_test['MSE']:.4f}")
                            st.write(f"RMSE: {metrics_test['RMSE']:.4f}")
                            st.write(f"R2: {metrics_test['R2']:.4f}")

                        else:    
                            st.write(f"Accuracy: {metrics_test['accuracy']:.4f}")
                            st.write(f"Precision: {metrics_test['precision']:.4f}")
                            st.write(f"Recall: {metrics_test['recall']:.4f}")
                            st.write(f"F1 Score: {metrics_test['f1_score']:.4f}")
                            st.write("Confusion Matrix:")
                            st.write(metrics_test['confusion_matrix'])
                            st.write("Classification Report:")
                            st.text(metrics_test['classification_report'])
                    
                else:
                    st.warning("Please train a model first in the 'Training' tab before evaluating on the test set.")

    

    with tab7:
        st.subheader("Make predictions and Download")
        
        
        pred_file=st.file_uploader("Upload CSV file for prediction",type=['csv'])
        if pred_file:
            pred_data=pd.read_csv(pred_file)
            st.write("Prediction Data Preview")
            st.dataframe(pred_data.head())
        if st.button("Run Prediction"):
            if mode == "Use Existing Model":
                if 'active_model' not in st.session_state:
                    st.warning("Activate a model first")
                    st.stop()
                model = st.session_state['active_model']
                pre = st.session_state['active_preprocessor']
                
            else:
                trained_models = st.session_state.get('trained_model', {})

                if not trained_models:
                    st.error("No trained model available. Train a model first.")
                    st.stop()

                model = next(iter(trained_models.values()))
                pre = st.session_state['preprocessor']


            df = pre['cleaner'].transform(pred_data)
            df = df.reindex(columns=pre['features'], fill_value=0)
            scaler = pre.get("scaler")

            if scaler is not None and getattr(scaler, "scaled_data", None) is not None:
                df_scaled = scaler.transform(df)
            else:
                df_scaled = df 

            preds = model.predict(df_scaled)

            result = pred_data.copy()
            if Algorithm_Type=="Clustering":
                result["Clusters"] = preds
            else:    
                result["Prediction"] = preds
        
            st.session_state['prediction'] = result
            st.success("Prediction Complete")

            st.dataframe(result.head())

            if 'prediction' in st.session_state:
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Prediction",
                    csv,
                    "prediction.csv",
                    "text/csv"
                )

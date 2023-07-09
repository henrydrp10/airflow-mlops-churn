from airflow.models import DAG
from airflow.decorators import dag, task
from datetime import datetime, timedelta

# List of Default Arguments
default_arguments = {
    "depends_on_past": False,
    "email": ["hdrub.02@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes = 5),
    "start_date": datetime(2023, 2, 4)
}

# Pipeline DAG
with DAG(

    "sdg_pipeline",
    description = "SDG Technical Test Machine Learning pipeline",
    default_args = default_arguments

) as dag:

    import numpy as np
    import pandas as pd

    @task
    def data_preprocessing():

        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import OrdinalEncoder

        # Load csv into a Pandas df
        dataset_df = pd.read_csv("/opt/airflow/data/dataset.csv", sep = ";", index_col = "Customer_ID", decimal = ',')

        # Divide columns by dtypes
        float_columns = list(dataset_df.select_dtypes(include='float64'))
        int_columns = list(dataset_df.select_dtypes(include='int64'))
        object_columns = list(dataset_df.select_dtypes(include='object'))

        # Modify dtypes of some columns
        cols_to_int = ['phones', 'models', 'lor', 'adults', 'income', 'numbcars']
        cols_to_bool = ['truck', 'rv', 'forgntvl']

        for col in cols_to_int:
            float_columns.remove(col)
            int_columns.append(col)

        for col in cols_to_bool:
            float_columns.remove(col)
            object_columns.append(col)

        fig, axs = plt.subplots(6, 4)
        flat_axs = axs.flatten()
        for i in range(len(object_columns)):
            sns.countplot(ax = flat_axs[i], x = object_columns[i], hue = "churn", data = dataset_df)
            flat_axs[i].tick_params(axis='x', rotation=90) if object_columns[i] == 'area' else flat_axs[i].tick_params(axis='x')

        plt.setp(axs[-1, :])
        plt.setp(axs[:, 0])
        fig.set_size_inches(40, 70)
        plt.savefig('/opt/airflow/data/categorical_vars.png')

        # Drop all the columns which may not have a considerable effect
        dataset_df.drop(columns = ['new_cell',
                            'prizm_social_one',
                            'area',
                            'refurb_new',  
                            'ownrent',
                            'dwlltype',
                            'marital', 
                            'HHstatin', 
                            'dwllsize',
                            'rv',
                            'truck', 
                            'forgntvl',
                            'ethnic', 
                            'kid0_2', 
                            'kid3_5', 
                            'kid6_10', 
                            'kid11_15', 
                            'kid16_17',
                            'creditcd'], inplace = True)

        # Add NaN values of 'hnd_webcap' to the variable UNKNOWN
        dataset_df['hnd_webcap'].fillna('UNKW', inplace = True)

        # Drop row containing the only NaN value of various columns
        dataset_df.drop([1077200], inplace = True)

        # Fill NaN values with the mean - numerical variables
        dataset_df['rev_Mean'].fillna(dataset_df['rev_Mean'].mean(), inplace = True)
        dataset_df['mou_Mean'].fillna(dataset_df['mou_Mean'].mean(), inplace = True)
        dataset_df['totmrc_Mean'].fillna(dataset_df['totmrc_Mean'].mean(), inplace = True)
        dataset_df['da_Mean'].fillna(dataset_df['da_Mean'].mean(), inplace = True)
        dataset_df['ovrmou_Mean'].fillna(dataset_df['ovrmou_Mean'].mean(), inplace = True)
        dataset_df['ovrrev_Mean'].fillna(dataset_df['ovrrev_Mean'].mean(), inplace = True)
        dataset_df['vceovr_Mean'].fillna(dataset_df['vceovr_Mean'].mean(), inplace = True)
        dataset_df['datovr_Mean'].fillna(dataset_df['datovr_Mean'].mean(), inplace = True)
        dataset_df['roam_Mean'].fillna(dataset_df['roam_Mean'].mean(), inplace = True)

        dataset_df['change_mou'].fillna(dataset_df['change_mou'].mean(), inplace = True)
        dataset_df['change_rev'].fillna(dataset_df['change_rev'].mean(), inplace = True)

        dataset_df['avg6mou'].fillna(dataset_df['avg6mou'].mean(), inplace = True)
        dataset_df['avg6qty'].fillna(dataset_df['avg6qty'].mean(), inplace = True)
        dataset_df['avg6rev'].fillna(dataset_df['avg6rev'].mean(), inplace = True)

        dataset_df['hnd_price'].fillna(dataset_df['hnd_price'].mean(), inplace = True)
        
        dataset_df.drop(columns = ['lor', 'adults', 'income', 'numbcars'], inplace = True)
        dataset_df['infobase'].fillna('U', inplace = True)

        # Dropping highly correlated features
        corr = dataset_df.corr()

        upper_tri = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
        dataset_df.drop(columns = to_drop, inplace = True)

        # Using encoding for the rest of the categorical variables
        dataset_df['crclscod'] = OrdinalEncoder().fit_transform(dataset_df['crclscod'].to_numpy().reshape(-1, 1))

        dataset_df['asl_flag'].replace('N', 0, inplace = True)
        dataset_df['asl_flag'].replace('Y', 1, inplace = True)

        dataset_df['hnd_webcap'].replace('WC', -1, inplace = True)
        dataset_df['hnd_webcap'].replace('UNKW', 0, inplace = True)
        dataset_df['hnd_webcap'].replace('WCMB', 1, inplace = True)

        dataset_df['infobase'].replace('M', -1, inplace = True)
        dataset_df['infobase'].replace('U', 0, inplace = True)
        dataset_df['infobase'].replace('N', 1, inplace = True)

        dualband_ohe_df = pd.get_dummies(dataset_df['dualband'])
        dataset_df = pd.concat([dataset_df, dualband_ohe_df], axis = 1)
        dataset_df.drop(columns = 'dualband', inplace = True)

        # Target variable: 'churn' - Moving it to the end of df
        dataset_df.insert(len(dataset_df.columns) - 1, 'churn', dataset_df.pop('churn'))

        # Saving the processed dataset
        dataset_df.to_csv('/opt/airflow/data/dataset_after_t1.csv', sep = ';', index = True, decimal = ',')

    @task
    def model_training():

        import matplotlib.pyplot as plt
        from pprint import pprint

        import keras
        from tensorflow import keras
        from keras import Sequential
        from keras.layers import Dense, Dropout
        import keras_tuner as kt

        import pickle

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        import mlflow
        from mlflow.tracking import MlflowClient


        # Load csv into a Pandas df
        dataset_df = pd.read_csv("/opt/airflow/data/dataset_after_t1.csv", sep = ";", decimal = ',', index_col = 'Customer_ID')

        # Loading numpy arrays - X & y
        X = dataset_df.to_numpy()[:,:-1]
        y = dataset_df.to_numpy()[:,-1]

        # Splitting the data - val: model_selection; - test: model_evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        # Saving the created test batch
        np.savetxt("/opt/airflow/data/X_test.csv", X_test, delimiter=",")
        np.savetxt("/opt/airflow/data/y_test.csv", np.round(y_test), delimiter=",")

        ### MLFLOW METHODS AND TOOLS FOR TRACKING 
        mlflow.set_experiment('Churn Prediction Experiment')
        mlflow.sklearn.autolog()

        def yield_artifacts(run_id, path=None):
            """Yield all artifacts in the specified run"""
            client = MlflowClient()
            for item in client.list_artifacts(run_id, path):
                if item.is_dir:
                    yield from yield_artifacts(run_id, item.path)
                else:
                    yield item.path


        def fetch_logged_data(run_id):
            """Fetch params, metrics, tags, and artifacts in the specified run"""
            client = MlflowClient()
            data = client.get_run(run_id).data
            # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
            tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
            artifacts = list(yield_artifacts(run_id))
            return {
                "params": data.params,
                "metrics": data.metrics,
                "tags": tags,
                "artifacts": artifacts,
            }

        # RANDOM FOREST CLASSIFIER
        rf_clf = RandomForestClassifier()
        rf_param_grid = {
            'n_estimators': [x * 10 for x in range(1, 10)],
            'max_depth': [x * 5 for x in range(5, 51, 5)], 
            'min_samples_leaf': [x for x in range(5, 11)],
            'max_features': ['sqrt', 'log2']
        }

        rand_search_rf = RandomizedSearchCV(
            estimator = rf_clf, 
            param_distributions = rf_param_grid, 
            cv = 3, 
            n_jobs = 1, 
            verbose = 3, 
            return_train_score=True
        )

        rand_search_rf.fit(X_train, y_train)
        run_id = mlflow.last_active_run().info.run_id
        mlflow.tracking.MlflowClient().set_tag(run_id, "mlflow.runName", "RandomForestClassifier")

        # show data logged in the parent run
        print("========== parent run ==========")
        for key, data in fetch_logged_data(run_id).items():
            print("\n---------- logged {} ----------".format(key))
            pprint(data)

        # show data logged in the child runs
        filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run_id)
        runs = mlflow.search_runs(filter_string=filter_child_runs)
        param_cols = ["params.{}".format(p) for p in rf_param_grid.keys()]
        metric_cols = ["metrics.mean_test_score"]

        print("\n========== child runs ==========\n")
        pd.set_option("display.max_columns", None)  # prevent truncating columns
        print(runs[["run_id", *param_cols, *metric_cols]])

        best_rf_model = rand_search_rf.best_estimator_

        importances = best_rf_model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in best_rf_model.estimators_], axis=0)

        best_predictions_rf = best_rf_model.predict(X_val).round()
        rf_acc = accuracy_score(y_val, best_predictions_rf)
        rf_precision = precision_score(y_val, best_predictions_rf)
        rf_recall = recall_score(y_val, best_predictions_rf)
        rf_f1 = f1_score(y_val, best_predictions_rf)

        # Feature importance
        importances = best_rf_model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in best_rf_model.estimators_], axis=0)
        forest_importances = pd.Series(importances, index = dataset_df.columns[:-1])

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.savefig('/opt/airflow/data/feature_importance_graph.png')

        # GRADIENT BOOSTING CLASSIFIER
        gb_clf = GradientBoostingClassifier()
        gb_param_grid = {
            'learning_rate': [0.001, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [7, 8, 9, 10], 
            'min_samples_leaf': [35, 40, 45, 50],
            'n_estimators': [10, 15, 20, 30],
            }

        rand_search_gb = RandomizedSearchCV(
            estimator = gb_clf, 
            param_distributions = gb_param_grid, 
            cv = 3, 
            n_jobs = 1, 
            verbose = 3, 
            return_train_score=True
        )

        rand_search_gb.fit(X_train, y_train)
        run_id = mlflow.last_active_run().info.run_id
        mlflow.tracking.MlflowClient().set_tag(run_id, "mlflow.runName", "GradientBoostingClassifier")

        # show data logged in the parent run
        print("========== parent run ==========")
        for key, data in fetch_logged_data(run_id).items():
            print("\n---------- logged {} ----------".format(key))
            pprint(data)

        # show data logged in the child runs
        filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run_id)
        runs = mlflow.search_runs(filter_string=filter_child_runs)
        param_cols = ["params.{}".format(p) for p in rf_param_grid.keys()]
        metric_cols = ["metrics.mean_test_score"]

        print("\n========== child runs ==========\n")
        pd.set_option("display.max_columns", None)  # prevent truncating columns
        print(runs[["run_id", *param_cols, *metric_cols]])

        best_gb_model = rand_search_gb.best_estimator_

        predictions_gb = rand_search_gb.predict(X_val).round()
        gb_acc = accuracy_score(y_val, predictions_gb)
        gb_precision = precision_score(y_val, predictions_gb)
        gb_recall = recall_score(y_val, predictions_gb)
        gb_f1 = f1_score(y_val, predictions_gb)

        # NEURAL NETWORK - SIMPLE MODEL
        # https://towardsdev.com/using-mlflow-with-keras-tuner-f6df5dd634bc       
        class TrackedHyperModel(kt.HyperModel):
            def build(self, hp):

                hp_units1 = hp.Int('units1', min_value=32, max_value=128, step=32)
                hp_units2 = hp.Int('units2', min_value=32, max_value=128, step=32)
                hp_units3 = hp.Int('units3', min_value=32, max_value=128, step=32)

                hp_dropout = hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3, 0.4])
                hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.005, 0.01, 0.05])

                model = Sequential()
                model.add(Dense(units = hp_units1, activation = 'relu', input_dim = X_train.shape[1]))
                model.add(Dropout(hp_dropout))
                model.add(Dense(units = hp_units2, activation = 'relu'))
                model.add(Dropout(hp_dropout))
                model.add(Dense(units = hp_units3 , activation = 'relu'))
                model.add(Dropout(hp_dropout))
                model.add(Dense(1, activation = 'sigmoid'))

                model.compile(optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate),
                            loss=keras.losses.BinaryCrossentropy(), metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.FalseNegatives()])

                return model

            def fit(self, hp, model, *args, **kwargs):
                with mlflow.start_run():
                    mlflow.log_params(hp.values)
                    mlflow.tensorflow.autolog()
                    return model.fit(*args, **kwargs)

        tuner = kt.RandomSearch(TrackedHyperModel(),
                                objective = 'val_binary_accuracy',
                                max_trials = 6,
                                overwrite = True)

        tuner.search(X_train, y_train, epochs=50, validation_data = (X_val, y_val), batch_size = 512)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_mlp_model = tuner.hypermodel.build(best_hps)
        best_model_history = best_mlp_model.fit(X_train, y_train, epochs=50, validation_data = (X_val, y_val), batch_size = 512)

        run_id = mlflow.last_active_run().info.run_id
        mlflow.tracking.MlflowClient().set_tag(run_id, "mlflow.runName", "MultiLayerPerceptronClassifier")

        # Plotting the training history of the model
        def plot_loss(fit_history):
            plt.figure(figsize=(13,5))
            plt.plot(range(1, len(fit_history.history['binary_accuracy'])+1), fit_history.history['binary_accuracy'], label='train')
            plt.plot(range(1, len(fit_history.history['val_binary_accuracy'])+1), fit_history.history['val_binary_accuracy'], label='validate')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('/opt/airflow/data/accuracy_history.png')

        plot_loss(best_model_history)

        predictions_model = best_mlp_model.predict(X_val).round()
        model_acc = accuracy_score(y_val, predictions_model)
        model_precision = precision_score(y_val, predictions_model)
        model_recall = recall_score(y_val, predictions_model)
        model_f1 = f1_score(y_val, predictions_model)

        print("RANDOM FOREST CLASSIFIER:")
        print("Accuracy: " + str(rf_acc))
        print("Precision: " + str(rf_precision))
        print("Recall: " + str(rf_recall))
        print("F1 Score: " + str(rf_f1))
        print("------------------------------")

        print("GRADIENT BOOSTING CLASSIFIER:")
        print("Accuracy: " + str(gb_acc))
        print("Precision: " + str(gb_precision))
        print("Recall: " + str(gb_recall))
        print("F1 Score: " + str(gb_f1))
        print("------------------------------")

        print("NEURAL NETWORK CLASSIFIER:")
        print("Accuracy: " + str(model_acc))
        print("Precision: " + str(model_precision))
        print("Recall: " + str(model_recall))
        print("F1 Score: " + str(model_f1))
        print("------------------------------")

        if rf_f1 > gb_f1 and rf_f1 > model_f1:
            pickle.dump(best_rf_model, open('/opt/airflow/models/best_model.pkl', 'wb'))
        elif gb_f1 > rf_f1 and gb_f1 > model_f1:
            pickle.dump(best_gb_model, open('/opt/airflow/models/best_model.pkl', 'wb'))
        else:
            pickle.dump(best_mlp_model, open('/opt/airflow/models/best_model.pkl', 'wb'))


    @task
    def model_evaluation():

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import pickle

        # Load previously selected model
        best_model = pickle.load(open('/opt/airflow/models/best_model.pkl', 'rb'))

        # Load variables for testing
        X_test = np.loadtxt('/opt/airflow/data/X_test.csv', delimiter = ',')
        y_test = np.loadtxt('/opt/airflow/data/y_test.csv', delimiter = ',')

        predictions = best_model.predict(X_test).round()
        best_acc = accuracy_score(y_test, predictions)
        best_precision = precision_score(y_test, predictions)
        best_recall = recall_score(y_test, predictions)
        best_f1 = f1_score(y_test, predictions)

        print("BEST MODEL:")
        print("Accuracy: " + str(best_acc))
        print("Precision: " + str(best_precision))
        print("Recall: " + str(best_recall))
        print("F1 Score: " + str(best_f1))
        print("------------------------------")

    data_preprocessing() >> model_training() >> model_evaluation()





from sklearn.ensemble import BaggingRegressor

#Bagging Regressor

#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
# https://scikit-learn.org/stable/modules/ensemble.html#bagging

bagging = BaggingRegressor(n_estimators=30,verbose=1,random_state=random_seed )
bagging_name = "Bagging Regressor"

baggings = train_model(x_train, y_train,bagging)
bagging_predictions = bagging.predict(x_val)

# then we can get some performance metrics
print_error_metrics(calculate_error_metrics(y_val,bagging_predictions, bagging_name), bagging_name, 'validation')

# and store the results on the test dataset for later model comparison, after we are done optimizing the parameters
#bagging_predictions = bagging.predict(x_test)
#print_error_metrics(calculate_error_metrics(y_test,bagging_predictions, bagging_name), bagging_name, 'test')
#metrics_collection = store_error_metrics(y_test,bagging_predictions, bagging_name, metrics_collection)
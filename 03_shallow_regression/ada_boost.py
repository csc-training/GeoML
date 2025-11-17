from sklearn.ensemble import AdaBoostRegressor
# AdaBoost Regressor

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
# https://scikit-learn.org/stable/modules/ensemble.html#adaboost

ada_boost = AdaBoostRegressor(n_estimators=30, learning_rate=1.0, loss='linear', random_state=random_seed)
ada_boost_name = "AdaBoost Regressor"

ada_boost = train_model(x_train, y_train,ada_boost)
ada_boost_predictions = ada_boost.predict(x_val)

# then we can get some performance metrics
print_error_metrics(calculate_error_metrics(y_val,ada_boost_predictions, ada_boost_name), ada_boost_name, 'vaidation')

# and store the results on the test dataset for later model comparison, after we are done optimizing the parameters
#ada_boost_predictions = ada_boost.predict(x_test)
#print_error_metrics(calculate_error_metrics(y_test,ada_boost_predictions, ada_boost_name), ada_boost_name, 'test')
#metrics_collection = store_error_metrics(y_test,ada_boost_predictions, ada_boost_name, metrics_collection)
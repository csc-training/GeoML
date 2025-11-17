from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting Regressor
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
#https://scikit-learn.org/stable/modules/ensemble.html#regression

grad_boost = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1,verbose=1)
grad_boost_name = "Gradient Boosting Regressor"
grad_boost = train_model(x_train, y_train,grad_boost)
grad_boost_predictions = grad_boost.predict(x_val)

# then we can get some performance metrics
print_error_metrics(calculate_error_metrics(y_val,grad_boost_predictions, grad_boost_name), grad_boost_name, 'validation')

# and store the results on the test dataset for later model comparison, after we are done optimizing the parameters
#grad_boost_predictions = grad_boost.predict(x_test)
#print_error_metrics(calculate_error_metrics(y_test,grad_boost_predictions, grad_boost_name), grad_boost_name, 'test')
#metrics_collection = store_error_metrics(y_test,grad_boost_predictions, grad_boost_name, metrics_collection)
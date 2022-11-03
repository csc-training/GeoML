from sklearn.ensemble import ExtraTreesRegressor
#Extra Trees Regressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

extra_trees = ExtraTreesRegressor(n_estimators=30,verbose=1, random_state=random_seed)
extra_trees_name = "Extra Trees Regressor"

extra_trees = train_model(x_train, y_train,extra_trees)
extra_trees_predictions = extra_trees.predict(x_val)

# then we can get some performance metrics
print_error_metrics(calculate_error_metrics(y_val,extra_trees_predictions, extra_trees_name), extra_trees_name, 'validation')

# and store the results on the test dataset for later model comparison, after we are done optimizing the parameters
#extra_trees_predictions = extra_trees.predict(x_test)
#print_error_metrics(calculate_error_metrics(y_test,extra_trees_predictions, extra_trees_name), extra_trees_name,'test')
#metrics_collection = store_error_metrics(y_test,extra_trees_predictions, extra_trees_name, metrics_collection)
param_dic = {'learning_rate': [0.05],  # weighting factor for the corrections by new trees when added to the model
             # number of trees added to the model
             'n_estimators': [500],
             'max_depth': [2],  # maximum depth of the tree
             # sets the minimum number of samples to split
             'min_samples_split': [100],
             # the minimum number of samples to form a leaf
             'min_samples_leaf': [7],
             # square root of features is usually a good starting point
             'max_features': [6],
             'subsample': [0.75]}  # the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.

Best Model parameters: {'subsample': 0.75, 'n_estimators': 500, 'min_samples_split': 100, 'min_samples_leaf': 7, 'max_features': 6, 'max_depth': 2, 'learning_rate': 0.05}
Best Model mean accuracy: 0.8644766997708174

Best Model parameters: {'subsample': 0.75, 'n_estimators': 750, 'min_samples_split': 2, 'min_samples_leaf': 7, 'max_features': 7, 'max_depth': 4, 'learning_rate': 0.005}
Best Model mean accuracy: 0.8571279621123995

Best Model parameters: {'subsample': 0.75, 'n_estimators': 750, 'min_samples_split': 4, 'min_samples_leaf': 7, 'max_features': 3, 'max_depth': 2, 'learning_rate': 0.01}
Best Model mean accuracy: 0.8661117478428133

param_dic = {'learning_rate': [0.005],  # weighting factor for the corrections by new trees when added to the model
             # number of trees added to the model
             'n_estimators': [750],
             'max_depth': [4],  # maximum depth of the tree
             # sets the minimum number of samples to split
             'min_samples_split': [2],
             # the minimum number of samples to form a leaf
             'min_samples_leaf': [7],
             # square root of features is usually a good starting point
             'max_features': [6],
             'subsample': [0.75]}  # the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.

n = 5
param_dic = {'learning_rate':[0.01],  
            # number of trees added to the model
            'n_estimators': [750],
            'max_depth':[2],    #maximum depth of the tree
            # sets the minimum number of samples to split
            'min_samples_split':[4],    #sets the minimum 
            # the minimum number of samples to form a leaf
            'min_samples_leaf': [7],
            # square root of features is usually a good starting point
            'max_features':[3],     #square root of features is usually 
            'subsample': [0.75]}  # the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.

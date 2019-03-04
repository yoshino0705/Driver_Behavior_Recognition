from driver_behavior import DriverBehavior

if __name__ == '__main__':
    path = 'data.csv'
    db = DriverBehavior(path)
    
    for m in db.models.keys():
        db.train(m)    
    db.print_accuracies(force_update=True)
    
    # The above loop would yield outputs similiar to below:
    ##################################
    # Train Accuracy:
    # {'adaboost': 0.4126015398742671,
    #  'decision_tree': 1.0,
    #  'gradient_boosting': 0.9590873772691955,
    #  'knn': 0.8430317157589885,
    #  'linear_svc': 0.5568552659461751,
    #  'logistic': 0.5602316875044148,
    #  'mlp': 0.9540721904358268,
    #  'naive_bayes': 0.2903439994349085,
    #  'random_forest': 0.999420781238963}
    # Test Accuracy:
    # {'adaboost': 0.41508794236066965,
    #  'decision_tree': 0.9738503920322102,
    #  'gradient_boosting': 0.9548633184996821,
    #  'knn': 0.7443949989404535,
    #  'linear_svc': 0.5582962492053402,
    #  'logistic': 0.5618987073532528,
    #  'mlp': 0.9334180970544607,
    #  'naive_bayes': 0.2879423606696334,
    # 'random_forest': 0.9684255138800594}
    ####################################

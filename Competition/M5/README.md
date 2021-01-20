# Competition Summary

Below is a summary of our model:

**Models** 
10 * 28 LGBM ([10 store_id] * [28 day-by-day]) 

**Features**

- https://www.kaggle.com/kyakovlev/m5-simple-fe
- https://www.kaggle.com/kyakovlev/m5-lags-features
    But we didn't use the recursive features.

- Target encoding of 8 groups by ordered time series: 
    ```
    ['id'],['item_id'],['dept_id'],['cat_id'],['id','tm_dw'],['item_id','tm_dw'],['dept_id','tm_dw'],['cat_id','tm_dw']
    ```
    We used the average of sales for 28 or 60 days as a feature for the next 28 days.
   source code:  https://www.kaggle.com/suuuuuu/m5-target-encoding
    

- Stacking - Binary prediction (sales = 0 or else) features by above features.
    source code: https://www.kaggle.com/suuuuuu/binary-challenge-evaluation-ca-1-2
    
- Deviation from the day of the event.
    source code: https://www.kaggle.com/suuuuuu/m5-event-lag

- sales lag from 1 to 27 (day-by-day)
    
**Objectives**

We used custom objectives. These were determined based on our metric below.

|store_id|objective|
|---|---|
|CA_1| tweedie (power = 1.1)
|CA_2| custom\_asymmetric\_train
|CA_3| tweedie (power = 1.2)
|CA_4| asymmetric_tweedie (p = 1.1, a = 1.2)
|TX_1| tweedie (power = 1.1)
|TX_2| tweedie (power = 1.1)
|TX_3| tweedie (power = 1.1)
|WI_1| asymmetric_tweedie (p = 1.08, a = 1.16) 
|WI_2| asymmetric_tweedie (p = 1.1, a = 1.15)
|WI_3| asymmetric_tweedie (p = 1.06, a = 1.2)

```
def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    a = 1.15
    b = 1
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual &lt; 0, -2 * residual * b, -2 * residual * a)
    hess = np.where(residual &lt; 0, 2 * b, 2 * a)
    return grad, hess
```

```
def asymmetric_tweedie(y_pred, y_true):
    p = 1.1
    a = 1.2
    b = 1
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = -y_true*np.exp(y_pred*(1-p))+np.exp(y_pred*(2-p))
    grad = np.where(residual &lt; 0,grad*b,grad*a)
    hess = -(1-p)*y_true*np.exp(y_pred*(1-p))+(2-p)*np.exp(y_pred*(2-p))
    hess = np.where(residual &lt; 0,hess*b,hess*a)
    return grad, hess

```

**Metrics and Validation**
We defined WRMSSE for each store. The LGBM hyperparameters were tuned using this evaluation metric.
- Validation: [1884-1913] and [1914-1941]
- grid search by optuna
    source code: https://www.kaggle.com/suuuuuu/m5-ca1-grid-final


**Final training**
- train data: from 1 to 1941
- no early-stopping
- LGBM round : 300
- ensemble of num_iteration = 100, 200, and 300
- source code: https://www.kaggle.com/suuuuuu/m5-ca1-model-final


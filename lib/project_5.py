
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, SGDRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif


def connect_to_postgres(url_string):
    url = url_string
    engine = create_engine(url)
    return engine


def load_data_from_database(url_string):
    connection  = connect_to_postgres(url_string)
    sql_query = """
    select * from madelon
    """
    madelon_df = pd.read_sql(sql_query, con=connection, index_col='index')
    return madelon_df


def make_data_dict(x_data, y_data, test_size=0.33, random_state=42):
    """X_data, y_data, test_size, random_state
       Create data_dict
    """
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    data_dict = {
    'X'  : x_data,
    'y'  : y_data,
    'X_train' : X_train,
    'y_train' : y_train,
    'X_test'  : X_test,
    'y_test'  : y_test,
    'test_size': test_size,
    'random_state': random_state
    }

    return data_dict


def general_transformer(transformer, data_dict):
    if 'processes' in data_dict.keys():
        data_dict['processes'].append(transformer)
    else:
        data_dict['processes'] = [transformer]

    transformer = StandardScaler()

    transformer.fit(data_dict['X_train'])

    # transformer.fit(data_dict['X_train'],data_dict['y_train'])
    # transformer.fit(data_dict['X_test'])

    #transformer.fit(data_dict['X_train'], data_dict['y_train'])

    data_dict['X_train'] = transformer.transform(data_dict['X_train'])
    data_dict['X_test'] = transformer.transform(data_dict['X_test'])

    return data_dict


#this_dd = general_transformer(StandardScaler(), data_dict)


def general_model(model, data_dict, random_state=None):
    """Build your general model after you've created data dictionary"""

    this_model = model
    this_model.fit(data_dict['X_train'], data_dict['y_train'])
    data_dict['train_score'] = this_model.score(data_dict['X_train'], data_dict['y_train'])
    data_dict['test_score'] = this_model.score(data_dict['X_test'], data_dict['y_test'])
    data_dict['model'] = this_model
    data_dict['coef'] = this_model.coef_

    return data_dict

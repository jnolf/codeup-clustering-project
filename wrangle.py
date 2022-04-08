import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from env import username, password, host


############################# Acquire ###############################

def acquire_df(use_cache=True):
    ''' 
    This function acquires all necessary housing data from zillow 
    needed to better understand future pricing
    '''
    
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')

    database_url_base = f'mysql+pymysql://{username}:{password}@{host}/zillow'
    query = '''
            SELECT prop.*, 
               pred.logerror, 
               pred.transactiondate, 
               air.airconditioningdesc, 
               arch.architecturalstyledesc, 
               build.buildingclassdesc, 
               heat.heatingorsystemdesc, 
               landuse.propertylandusedesc, 
               story.storydesc, 
               construct.typeconstructiondesc 
               FROM   properties_2017 prop 
               
               INNER JOIN (SELECT parcelid,
                                  logerror,
                                  Max(transactiondate) transactiondate 
                           FROM   predictions_2017 
                           GROUP  BY parcelid, logerror) pred
                       USING (parcelid)
                       
               LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
               LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
               LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
               LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
               LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
               LEFT JOIN storytype story USING (storytypeid) 
               LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
               WHERE  prop.latitude IS NOT NULL 
               AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31' 
'''
    
    df = pd.read_sql(query, database_url_base)
    df.to_csv('zillow.csv', index=False)
    
    return df

################## Dealing With Missing Values #####################

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    ''' Takes in a DataFrame and is defaulted to have at least 60% of values for 
    columns and 75% for rows'''
    threshold = int(round(prop_required_column * len(df.index),0))
    df.dropna(axis=1, thresh = threshold, inplace = True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return df

def missing_values(df):
    missing_values =pd.concat([
                    df.isna().sum().rename('count'),
                    df.isna().mean().rename('percent')
                    ], axis=1)
    return missing_values


def missing_counts_and_percents(df):
    missing_counts_and_percents = pd.concat([
                                  df.isna().sum(axis=1).rename('num_cols_missing'),
                                  df.isna().mean(axis=1).rename('percent_cols_missing'),
                                  ], axis=1).value_counts().sort_index()
    return pd.DataFrame(missing_counts_and_percents).reset_index()

 ############################ Outliers #############################

def remove_outliers(df, k, cols):
    ''' Take in a dataframe, k value, and specified columns within a dataframe 
    and then return the dataframe with outliers removed
    '''
    for col in cols:
        # Get quartiles
        q1, q3 = df[col].quantile([.25, .75]) 
        # Calculate interquartile range
        iqr = q3 - q1 
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

############################# Clean ################################

def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['los_angeles', 'orange', 'ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies

def clean_df(df):
    '''
    This function takes in the zillow data, cleans it, and returns a dataframe
    '''
    # Identify the use codes that are single family from SequelAce
    single_fam_use = [261, 262, 263, 264, 265, 266, 268, 273, 275, 276, 279]
    # Make sure the DataFarme only includes the above
    df = df[df.propertylandusetypeid.isin(single_fam_use)]
    
#     df = get_counties(df) 
    
    # Remove further outliers for sqft to ensure data is usable
    df = df[(df['calculatedfinishedsquarefeet'] > 500) 
            & (df['calculatedfinishedsquarefeet'] <3_000)]
    
    # Remove further outliers for taxvalue to ensure data is usable
    df = df[(df.taxvaluedollarcnt < 3_000_000)]
    
    # Restrict df to only those properties with at least 1 bath & bed 
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
            
    # Deal with remaining nulls
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .7)
    
    # Create a column that is the age of the property
    df['age'] = 2022 - df.yearbuilt
    
    # Create a column for county name based on FIPS
    df['county'] = df.fips.apply(lambda x: 'orange' if x == 6059.0 else 'los_angeles'
                                 if x == 6037.0 else 'ventura')  
            
    # Determine unnecessary columns
    cols_to_remove = ['id','calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt',
              'heatingorsystemtypeid', 'heatingorsystemdesc', 'propertycountylandusecode',
              'propertylandusetypeid','propertyzoningdesc', 'regionidcity', 'regionidzip', 
              'propertylandusedesc', 'unitcnt', 'censustractandblock', 'buildingqualitytypeid']    
     # Create a new dataframe that dropps those columns       
    df = df.drop(columns = cols_to_remove)
    
        #Drop rows with null values since it is only a small portion of the dataframe 
    df = df.dropna()
    
    return df

############################# Split #################################

def split_data(df):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=1349)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=1349)

#     #Make copies of train, validate, and test
#     train = train.copy()
#     validate = validate.copy()
#     test = test.copy()
    
#     # create X_train by dropping the target variable 
#     X_train = train.drop(columns=[target_var])
#     # create y_train by keeping only the target variable.
#     y_train = train[[target_var]]

#     # create X_validate by dropping the target variable 
#     X_validate = validate.drop(columns=[target_var])
#     # create y_validate by keeping only the target variable.
#     y_validate = validate[[target_var]]

#     # create X_test by dropping the target variable 
#     X_test = test.drop(columns=[target_var])
#     # create y_test by keeping only the target variable.
#     y_test = test[[target_var]]

    return train, validate, test, 
############################## Scale #################################

def min_max_df(df, features):
    '''
    Scales the df. using the MinMaxScaler()
    takes in the df and returns the df in a scaled fashion.
    '''
    # Make a copy of the original df
    df = df.copy()

    # Create the scaler
    scaler = MinMaxScaler()

    # Fit the scaler 
    scaler.fit(df)

    # Transform and rename columns for the df
    df_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns.tolist())
    return df_scaled

def min_max_split(train, validate, test):
    '''
    Scales the 3 data splits. using the MinMaxScaler()
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    If return_scaler is true, the scaler object will be returned as well.
    '''
    # Make the scaler
    scaler = MinMaxScaler()
    # List columns that need to be scaled
    cols = train[['bathroomcnt', 'bedroomcnt',
       'calculatedfinishedsquarefeet',
       'lotsizesquarefeet', 'roomcnt', 'yearbuilt',
       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
       'landtaxvaluedollarcnt', 'taxamount', 'taxrate', 'age']].columns.tolist()
    # Make a copy of original train, validate, and test
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # Use/fit the Scaler
    train_scaled[cols] = scaler.fit_transform(train[cols])
    validate_scaled[cols] = scaler.fit_transform(validate[cols])
    test_scaled[cols] = scaler.fit_transform(test[cols])

    return train_scaled, validate_scaled, test_scaled

############################# Wrangle ################################

def wrangle_zillow():
    ''' 
    This function combines both functions above and outputs three 
    cleaned and prepped datasets
    '''
    # Acquire the df
    acquire = acquire_df()

    # Get a clean df
    cleaned = clean_df(acquire)

    # Create more features to use
    df = create_features(cleaned)
    
    # Split that clean df to ensure minimal data leakage
    train, validate, test, 
    X_train, X_validate, X_test, 
    y_train, y_validate, y_test = split_data(df, 'logerror')

    return train, validate, test, X_train, X_validate, X_test, y_train, y_validate, y_test

############################# Modeling ################################   
##################### Show Clusters/Centroids #########################

def cluster(df, feature1, feature2, k):
    X = df[[feature1, feature2]]

    kmeans = KMeans(n_clusters=k).fit(X)
    
    df['cluster'] = kmeans.labels_
    df.cluster = df.cluster.astype('category')
    
    df['cluster'] = kmeans.predict(X)

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    df.groupby('cluster')[feature1, feature2].mean()
    
    plt.figure(figsize=(9, 7))
    
    for cluster, subset in df.groupby('cluster'):
        plt.scatter(subset[feature1], subset[feature2],  label='cluster ' + str(cluster), 
                    alpha=.6)
    
    centroids.plot.scatter(x=feature1, y=feature2, c='black', marker='x', s=100, ax=plt.gca(),
                           label='centroid')
    
    plt.legend()
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Visualizing Cluster Centers')

    return

################ Find The Best K Value For Clustering ##################

def inertia(df, feature1, feature2, r1, r2):
    cols = [feature1, feature2]
    X = df[cols]
    
    inertias = {}
    
    for k in range(r1, r2):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertias[k] = kmeans.inertia_
    
    pd.Series(inertias).plot(xlabel='k', ylabel='Inertia', figsize=(9, 7)).plot(marker='x')
    plt.grid()
    return

####################### Plotting Variable Pairs #########################

def plot_variable_pairs(df):
    # plot the columns in a pairplot
    sns.pairplot(df, kind = 'reg', corner = True, plot_kws={'line_kws':{'color':'red'}})
    plt.show()
#     plt.tight_layout()


def create_features(df):
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 
                                   80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, 
                                     .533, .60, .666, .733, .8, .866, .933])

    # Create taxrate column
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # Create acres column
    df['acres'] = df.lotsizesquarefeet/43560

    # Bin the acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 
                                               1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, 
                                 .5, .6, .7, .8, .9])
    # Bin counties
    df['county_code_bin'] = pd.cut(df.fips, bins=[0, 6037.0, 6059.0, 6111.0], 
                             labels = ['Los Angeles County', 'Orange County',
                             'Ventura County'])
    
    # Make an absolute value of logerror column
    df['abs_logerror'] = abs(df.logerror)
    
    # Bin abs_logerror
    df['abs_logerror_bin'] = pd.cut(df.abs_logerror, [0, .05, .1, .15, .2, .25, 
                                                      .3, .35, .4, .45, 5])
    
    # Bin logerror
    df['logerror_bin'] = pd.cut(df.logerror, [-5, -.2, -.05, .05, .2, 5])
    
    # Bin sqft
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 
                                    2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, 
                                      .5, .6, .7, .8, .9]
                       )

    # Dollar/Sqft for structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet

    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 
                                                     200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, 
                                                       .5, .6, .7, .8, .9]
                                            )

    # Dollar/Sqft for land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 
                                                                        250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, 
                                                 .5, .6, .7, .8, .9]
                                      )

    # Make bins floats
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})

    # Ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

#     # 12447 is the ID for city of LA. 
#     # I confirmed through sampling and plotting, as well as looking up a few addresses.
#     df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    return df


def get_zillow_dummies(train, validate, test, cat_columns = ['age_bin', 'county', 'county_code_bin', 
                                                             'abs_logerror_bin', 'logerror_bin']):
    '''
    This function takes in train, validate, test and a list of categorical columns for dummies (cat_columns)
    default col_list is for zillow 
    '''
    # create dummies 
    train = pd.get_dummies(data = train, columns = cat_columns, drop_first=False)
    validate = pd.get_dummies(data = validate, columns = cat_columns, drop_first=False)
    test = pd.get_dummies(data = test, columns = cat_columns, drop_first=False)
    
    return train, validate, test

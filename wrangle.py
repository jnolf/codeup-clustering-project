import numpy as np
import pandas as pd
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

def handle_missing_values(df, prop_required_column = .6, prop_required_row = .75):
    ''' Takes in a DataFrame and is defaulted to have at least 60% of values for 
    columns and 75% for rows'''
    threshold = int(round(prop_required_column * len(df.index),0))
    df.dropna(axis=1, thresh = threshold, inplace = True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return 

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

def clean_df(df):
    '''
    This function takes in the zillow data, cleans it, and returns a dataframe
    '''
    # Identify the use codes that are single family from SequelAce
    single_fam_use = [261, 262, 263, 264, 265, 266, 268, 273, 275, 276, 279]
    # Make sure the DataFarme only includes the above
    df = df[df.propertylandusetypeid.isin(single_fam_use)]
     
    # Remove further outliers for sqft to ensure data is usable
#     df = df[(df.calculatedfinishedsquarefeet > 500 & df.calculatedfinishedsquarefeet < 3_000)]
    
    # Remove further outliers for taxvalue to ensure data is usable
    df = df[(df.taxvaluedollarcnt < 3_000_000)]
    
    # Restrict df to only those properties with at least 1 bath & bed 
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
            
    # Deal with remaining nulls
    df = handle_missing_values(df, prop_required_column = .6, prop_required_row = .75)
    
    #Drop rows with null values since it is only a small portion of the dataframe 
    df = df.dropna()
    
    # Create a column that is the age of the property
    df['age'] = 2022 - df.yearbuilt
            
    # Rename FIPS codes to their respective counties
    df.fips = df.fips.replace({6037:'Los Angeles',
                           6059:'Orange',          
                           6111:'Ventura'})
    # Rename 'fips' to 'county
    df.rename(columns={'fips':'county'}, inplace = True)
            
    # Determine unnecessary columns
    cols_to_remove = ['id','calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt',
              'heatingorsystemtypeid','propertycountylandusecode',
              'propertylandusetypeid','propertyzoningdesc', 
              'propertylandusedesc', 'unitcnt', 'censustractandblock']    
     # Create a new dataframe that dropps those columns       
    df = df.drop(columns = cols_to_remove)
            
    #Drop rows with null values since it is only a small portion of the dataframe 
    df = df.dropna()
    
    return df

############################# Split #################################

def split_data(df):
    ''' 
    This function will take in the data and split it into train, 
    validate, and test datasets for modeling, evaluating, and testing
    '''
    train_val, test = train_test_split(df, train_size = .8, random_state = 123)

    train, validate = train_test_split(train_val, train_size = .7, random_state = 123)

    return train, validate, test

############################## Scale #################################

def min_max_df(df):
    '''
    Scales the df. using the MinMaxScaler()
    takes in the df and returns the df in a scaled fashion.
    '''
    # Make a copy of the original df
    df = df.copy()

    # Create the scaler
    scaler = sklearn.preprocessing.MinMaxScaler()

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
    # Make copies of train, validate, and test data splits
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    # Create the scaler
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit scaler on train dataset
    scaler.fit(train)

    # Transform and rename columns for all three datasets
    train_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns.tolist())
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns = train.columns.tolist())
    test_scaled = pd.DataFrame(scaler.transform(test), columns = train.columns.tolist())

    return train_scaled, validate_scaled, test_scaled

############################# Wrangle ################################

def wrangle_zillow():
    ''' 
    This function combines both functions above and outputs three 
    cleaned and prepped datasets
    '''
    # Acquire the df
    df = acquire_df()

    # Get a clean df
    cleaned = clean_df(df)

    # Split that clean df to ensure minimal data leakage
    train, validate, test = split_data(cleaned)

    return train, validate, test

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
        plt.scatter(subset[feature2], subset[feature1], label='cluster ' + str(cluster), alpha=.6)
    
    centroids.plot.scatter(y=feature1, x=feature2, c='black', marker='x', s=100, ax=plt.gca(), label='centroid')
    
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
    
    pd.Series(inertias).plot(xlabel='k', ylabel='Inertia', figsize=(9, 7))
    plt.grid()
    return


---- 
## Zillow Clustering Project
#### Reported by Jerry Nolf  -  April 8, 2022
---- 
### 1. Overview Project Goals
- Create a model that minimizes absolute log error
- Continue working with the Zillow Regression Project and incorporate clustering methodologies
- Construct a machine learning regression model
- Find key drivers of absolute log error of the zestimate

---- 
### 2. Project Description
- Use clustering methodologies to further your findings and refine teh current model

---- 
### 3. Initial Questions/Hypothesis
- Which county has the highest absolute log error?
- Is there a linear relationship between logerror and age for each county?
- Is there a linear relationship between acres and logerror?
- What counties have the largest log errors?

---- 
### 4. Data Dictionary 
| Column Name               | Description                              | Data Type              |
|---------------------------|------------------------------------------|------------------------|
| fips                      | FIPS code of property                    | float64                |
| regionidzip               | zip code (not accurate)                  | float64                |
| county                    | which county property is in              | object                 |
| lot_dollar_sqft_bin       | bin size for lotsize                     | float64                |
| land_dollar_per_sqft      | landtaxvaluedollarcnt/sqft               | float64                |
| structure_dollar_sqft_bin | bin size for dollar per square foot      | float64                |
| structure_dollar_per_sqft | dollar per square foot                   | float64                |
| sqft_bin                  | bin size for square footage              | float64                |
| acres_bin                 | bin size for acres                       | float64                |
| acres                     | number of acres (lotsize/43560)          | float64                |
| taxrate                   | tax rate of property                     | float64                |
| age_bin                   | bin size for age                         | float64                |
| age                       | age of a property (2017 - yearbuilt)     | float64                |
| ventura                   | if county is in Ventura =1, else 0       | unit8                  |
| orange                    | if property is in Orange =1, else 0      | unit8                  |
| los_angeles               | if property is in Los Angeles =1, else 0 | unit8                  |
| property_id               | property id                              | int64                  |
| propertylandusedesc       | property land use type                   | pbject                 |
| transactiondate           | transaction date                         | object                 |
| logerror                  | log error                                | float64                |
| id                        | ID for the property                      | int64                  |
| censustractandblock       | census information                       | float64                |
| tax_amount                | tax amount of property                   | float64                |
| assessmentyear            | year of tax assessment                   | float64                |
| landtaxvaluedollarcnt     | tax value in dollars of the land         | float64                |
| tax_value                 | tax value of property                    | float64                |
| structuretaxvaluedollarcnt| tax value in dollars of the structure    | float64                |
| year_built                | year property was built                  | float64                |
| roomcnt                   | number of rooms                          | float64                |
| regionidzip               | id for region by zip code                | float64                |
| regionidcity              | id for region by city                    | float64                |
| rawcensustractandblock    | census data                              | float64                |   
| propertycountylandusecode | property county land use                 | object                 |
| lotsizesquarefeet         | lot size square footage                  | float64                |
| latitude                  | latitude coordinate                      | float64                |
| longitude                 | longitude coordinate                     | float64                |
| fullbathcnt               | number of full bathrooms                 | float64                |
| finishedsquarefeet12      | square footage                           | float64                |
| calculatesfinishedsquarefeet  | square footage of property               | float64                |
| calculatedbathnbr         | calculated number of bathrooms           | float64                |
| bedroomcnt                | number of bedrooms                       | float64                |
| bathroom cnt              | number of bathrooms                      | float64                |
| id                        | ID for the property                      | int64                  |
| propertylandusetypeid     | property ID                              | float64                |
| parcelid                  | property ID                              | int64                  |

---- 
## PROCESS:
The following outlines the process taken through the Data Science Pipeline to complete this project.  

Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver

### 1. PLAN
- Define the project goal
- Determine proper format for the audience
- Asked questions that would lead to final goal
- Define an MVP


### 2. ACQUIRE
- Create a function to pull appropriate information from the zillow database
- Create and save a wrangle.py file in order to use the function to acquire


### 3. PREPARE
- Ensure all data types are usable
- Create a function that  will:
        - remove duplicates
        - handle missing values
        - convert data types
        - handle outliers
        - encode categorical columns
        - renames columns
        - created a columns for house 'age'
        - scale data for exploration
- Add a function that splits the acquired data into Train, Validate, and Test sets
- 20% is originally pulled out in order to test in the end
- From the remaining 80%, 30% is pullout out to validate training
- The remaining data is used as testing data
- In the end, there should be a 56% Train, 24% Validate, and 20% Test split 
- Create a prepare.py file with functions that will quickly process the above actions


### 4. EXPLORE
- Create an exploratory workbook
- Create initial questions to help explore the data further
- Make visualizations to help identify and understand key driver
- Create clusters in order to dive deeper and refine features
- Use stats testing on established hypotheses


### 5. MODEL & EVALUATE
- Use clusters to evaluate drivers of assessed tax value
- Create a baseline
- Make predictions of models and what they say about the data
- Compare all models to evaluate the best for use
- Use the best performing model on the test (unseen data) sample
- Compare the modeled test versus the baseline


### 6. DELIVERY
- Present a final Jupyter Notebook
- Make modules used and project files available on Github

---- 
## REPRODUCIBILITY: 
	
### Steps to Reproduce
1. Have your env file with proper credentials saved to the working directory

2. Ensure that a .gitignore is properly made in order to keep privileged information private

3. Clone repo from github to ensure availability of the acquire and prepare imports

4. Ensure pandas, numpy, matplotlib, scipy, sklearn, and seaborn are available

5. Follow steps outline in this README.md to run Final_Zillow_Report.ipynb


---- 
## KEY TAKEAWAYS:

### Conclusion:
#### The goals of this project were to identify key drivers of tax value for single family residential homes purchased during 2017. These key drivers were found to be the following created clusters:

- age_acres_cluster
- location_cluster
- house-to_lot_cluster

Using these drivers, the model will decrease log error by 0.015191

### Recommendation(s):
While our model does improve slightly on absolute log error, higher quality data is needed in order for larger gains.

### Next Steps:
With more time, I would like to:

- Work on more feature engineering and explore relationships of categories to log error further.
- Gather more adequate and complete data that will allow for a clearer picture and the possibility for a more refined and detailed model. 

---- 
# Data-Refinement-Project
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv("Bengaluru_House_Data.csv")
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
# print(df2.isnull().sum())
df3 = df2.dropna()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df3.bhk.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

NEW_DF3 = df3[~df3['total_sqft'].apply(is_float)].head(10)
# print(DF3)
# print(df3.total_sqft.unique())


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
new_df4 = df4['total_sqft'].apply(convert_sqft_to_num) 
# print(new_df4.loc[30])

#  DATA CLEANING COMPLETED HERE



#  NOW WE WILL DO SOME FEATURE ENGINEERING 
def round_off_int(x):
    for x in df5['total_sqft']:
        if type(x) == str:
            x = int(x)
        return x 

df5 = df4.copy()
df5['total_sqft'] = df5['total_sqft'].apply(round_off_int)
df5["price_per_sqft"] = (df5['price']*100000) / df5['total_sqft'] 
# print(df5.head())
df5.location = df5.location.apply(lambda x: x.strip())
location_stat = df5.groupby("location")['location'].agg("count").sort_values(ascending=False)
# print(location_stat)

location_stat_lessthan10 = location_stat[location_stat<=10]

df5.location = df5.location.apply(lambda x: 'other' if x in location_stat_lessthan10 else x)
# print((df5.location.unique()))

# feature engineerning is done here 


#  now we will do outlier detection 
 

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
# print(df6)
# print(df6.price_per_sqft.describe())

def remove_pps_outlier(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[((subdf.price_per_sqft>(m-st)) & ((subdf.price_per_sqft <= (m+st) )))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return reduced_df 

df7 = remove_pps_outlier(df6)
# print(df7)


def remove_BHK_outliers(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stat = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stat['bhk'] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stat.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.appende(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis = 'index')

df8 = remove_BHK_outliers(df7)
# print(remove_BHK_outliers(df7).head())
# import matplotlib
# matplotlib.rcParams['figure.figsize'] = (20,10)
# plt.hist(df8.price_per_sqft,rwidth=0.8)
# plt.xlabel('Price per squarefeet')
# plt.ylabel('Count')

df9 = df8[df8.bath<df8.bhk + 2]
# print(df9.shape)

df10 = df9.drop(['size','price_per_sqft'],axis = 'columns')
# print(df10.head(50))



# NOW ADDING ML HERE
dummies = (pd.get_dummies(df10.location))

df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
# print(df11.head())

df12 = df11.drop('location',axis='columns')

x = df12.drop('price',axis = 'columns')
# print(x.head())
y = df12.price
# print(y)

import matplotlib 
plt.scatter(df9.total_sqft, df9.price)
plt.xlabel("Total Square Feet")
plt.ylabel("Price (in Lakhs)")
plt.title("House Price vs Area in Bangalore")
plt.show()

plt.scatter(df9.bhk, df9.price)
plt.xlabel("Number of Bedrooms (BHK)")
plt.ylabel("Price (in Lakhs)")
plt.title("House Price vs Number of Bedrooms")
plt.show()

plt.scatter(df5.total_sqft, df5.price_per_sqft)
plt.xlabel("Total Square Feet")
plt.ylabel("Price per Sqft")
plt.title("Before Outlier Removal")
plt.show()

plt.scatter(df8.total_sqft, df8.price_per_sqft)
plt.xlabel("Total Square Feet")
plt.ylabel("Price per Sqft")
plt.title("After Outlier Removal")
plt.show()

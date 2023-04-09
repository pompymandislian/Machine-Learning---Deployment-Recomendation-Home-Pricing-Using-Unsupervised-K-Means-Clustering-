import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as scs
import joblib
import matplotlib.pyplot as plt

# call data
data_analisa = joblib.load("D:/BOOTCAMP/project/(Block 4) ML Process/Data_project.csv")

data_analisa.head(2)

data_analisa.describe().astype('int64')

# Check duplicate
duplicateRows = data_analisa[data_analisa.duplicated()]
duplicateRows.head(5)

# it's not duplicated values

# EDA

# check data skew (distribution)
data_analisa.skew(numeric_only = True)

# check with visualization
fig, axs = plt.subplots(3, 3, figsize=(20, 10))

sns.kdeplot(data=data_analisa, x="Price",color="blue", ax=axs[0, 0])
sns.kdeplot(data=data_analisa, x="Bedroom",color="blue", ax=axs[0, 1])
sns.kdeplot(data=data_analisa, x="Bathroom",color="blue", ax=axs[0, 2])
sns.kdeplot(data=data_analisa, x="Land",color="blue", ax=axs[1, 0])
sns.kdeplot(data=data_analisa, x="Building",color="blue", ax=axs[1, 1])
sns.kdeplot(data=data_analisa, x="Garage",color="blue", ax=axs[1, 2])
sns.kdeplot(data=data_analisa, x="Installment_BCA",color="blue", ax=axs[2, 0])
sns.kdeplot(data=data_analisa, x="Installment_BNI",color="blue", ax=axs[2, 1])
sns.kdeplot(data=data_analisa, x="Installment_Mandiri",color="blue", ax=axs[2, 2])

# Check boxplot for outlier data numeric

sns.set(style="darkgrid")

fig, axs = plt.subplots(3, 3, figsize=(20, 15))

sns.boxplot(data=data_analisa, x="Price", color="skyblue", ax=axs[0, 0])
sns.boxplot(data=data_analisa, x="Bedroom", color="olive", ax=axs[0, 1])
sns.boxplot(data=data_analisa, x="Bathroom", color="red", ax=axs[0, 2])
sns.boxplot(data=data_analisa, x="Land", color="green", ax=axs[1, 0])
sns.boxplot(data=data_analisa, x="Building", color="green", ax=axs[1, 1])
sns.boxplot(data=data_analisa, x="Garage", color="red", ax=axs[1, 2])
sns.boxplot(data=data_analisa, x="Installment_BCA", color="green", ax=axs[2, 0])
sns.boxplot(data=data_analisa, x="Installment_BNI", color="green", ax=axs[2, 1])
sns.boxplot(data=data_analisa, x="Installment_Mandiri", color="green", ax=axs[2, 2])

plt.show()

# group data Floor
sns.histplot(data = data_analisa, x = "Floor", hue = "Floor")

# see pattern data with analysis
floor_1 = data_analisa[(data_analisa['Floor'] == 'Lantai 1')]
floor_2 = data_analisa[(data_analisa['Floor'] == 'Lantai 2')]
floor_3 = data_analisa[(data_analisa['Floor'] == 'Lantai 3')]

median_1 = floor_1.groupby(["Floor"])["Price"].median().astype('int64')
median_2 = floor_2.groupby(["Floor"])["Price"].median().astype('int64')
median_3 = floor_3.groupby(["Floor"])["Price"].median().astype('int64')

median = {
          'Floor' : ['1','2','3'],
          'median' : ['1500000000','2150000000','1800000000']
         }

pd.DataFrame.from_dict(median)

# make binning data for buildings and lands

data_analisa['bin_land'] = pd.cut(
                            data_analisa['Land'],
                            [0,100,200,300,1000,99506], 
                            right=False, 
                            labels=list('ABCDE'))


data_analisa['bin_build'] = pd.cut(
                            data_analisa['Building'],
                            [0,100,200,300,1000,2020], 
                            right=False, 
                            labels=list('ABCDE'))                       

# change bin name

data_analisa['bin_build'] = data_analisa['bin_build'].apply(lambda x : x.replace('A','Very Small'))
data_analisa['bin_build'] = data_analisa['bin_build'].apply(lambda x : x.replace('B','Small'))
data_analisa['bin_build'] = data_analisa['bin_build'].apply(lambda x : x.replace('C','Medium'))
data_analisa['bin_build'] = data_analisa['bin_build'].apply(lambda x : x.replace('D','Large'))
data_analisa['bin_build'] = data_analisa['bin_build'].apply(lambda x : x.replace('E','Very Large'))

data_analisa['bin_land'] = data_analisa['bin_land'].apply(lambda x : x.replace('A','Very Small'))
data_analisa['bin_land'] = data_analisa['bin_land'].apply(lambda x : x.replace('B','Small'))
data_analisa['bin_land'] = data_analisa['bin_land'].apply(lambda x : x.replace('C','Medium'))
data_analisa['bin_land'] = data_analisa['bin_land'].apply(lambda x : x.replace('D','Large'))
data_analisa['bin_land'] = data_analisa['bin_land'].apply(lambda x : x.replace('E','Very Large'))

# see visualization data bin land
bin_land = data_analisa.groupby(["bin_land"])["Price"].count().sort_values(ascending=False)
bin_build = data_analisa.groupby(["bin_build"])["Price"].count().sort_values(ascending=False)

# visualization bin_land
plt.subplot(2, 2, 1)
bin_land.plot(x="Price", y="bin_land", kind="bar", figsize=(15, 15), rot= 0, color='blue')
plt.title ("Price Base On Bin_Land", size = 15)
plt.ylabel('Price', size = 10)
plt.xlabel('Bin_land',  size = 10)
plt.minorticks_on()

# visualization bin_build
plt.subplot(2, 2, 2)
bin_build.plot(x="Price", y="bin_build", kind="bar", figsize=(15, 15), rot= 0, color='blue')
plt.title ("Price Base On Bin_Build", size = 15)
plt.ylabel('Price', size = 10)
plt.xlabel('Bin_land',  size = 10)
plt.minorticks_on()

# check median base on bin_lands

land_verysmall = data_analisa[(data_analisa['bin_land'] == 'Very Small')]
land_small = data_analisa[(data_analisa['bin_land'] == 'Small')]
land_medium = data_analisa[(data_analisa['bin_land'] == 'Medium')]
land_large = data_analisa[(data_analisa['bin_land'] == 'Large')]
land_verylarge = data_analisa[(data_analisa['bin_land'] == 'Very Large')]

median_land1 = land_verysmall.groupby(["bin_land"])["Price"].median()
median_land2 = land_small.groupby(["bin_land"])["Price"].median()
median_land3 = land_medium.groupby(["bin_land"])["Price"].median()
median_land4 = land_large.groupby(["bin_land"])["Price"].median()
median_land5 = land_verylarge.groupby(["bin_land"])["Price"].median()

# check median base on bin_build

build_verysmall = data_analisa[(data_analisa['bin_build'] == 'Very Small')]
build_small = data_analisa[(data_analisa['bin_build'] == 'Small')]
build_medium = data_analisa[(data_analisa['bin_build'] == 'Medium')]
build_large = data_analisa[(data_analisa['bin_build'] == 'Large')]
build_verylarge = data_analisa[(data_analisa['bin_build'] == 'Very Large')]

median_11 = build_verysmall.groupby(["bin_build"])["Price"].median()
median_22 = build_small.groupby(["bin_build"])["Price"].median()
median_33 = build_medium.groupby(["bin_build"])["Price"].median()
median_44 = build_large.groupby(["bin_build"])["Price"].median()
median_55 = build_verylarge.groupby(["bin_build"])["Price"].median()

# Check median lands
median_land = {
          'land' : ['very_small','small','medium','large','very_large'],
          'median' : ['690000000','1600000000','2500000000','4000000000','4500000000']
         }

pd.DataFrame.from_dict(median_land)

# Check median builds
median_build = {
          'land' : ['very_small','small','medium','large','very_large'],
          'median' : ['685000000','1700000000','2800000000','3900000000','10000000000']
         }

pd.DataFrame.from_dict(median_build)

# drop columns bin because not use again
data_analisa = data_analisa.drop(columns=['bin_land','bin_build'])

# we want check with Important places column with bin
price_location = data_analisa.groupby(["Location"])["Price"].median().astype('int64').sort_values(ascending=False).head(10)

# see with visualization
price_location.plot(x="Price", y="Location", kind="bar", figsize=(5, 5), rot= 0, color='green')
plt.title ("Price Base On Location", size = 15)
plt.ylabel('Price', size = 10)
plt.xlabel('Location',  size = 10)
plt.minorticks_on()

# group data important places
sns.histplot(data = data_analisa, x = "Important places", hue = "Important places")

# Check important places with price
tempat_penting = data_analisa[(data_analisa['Important places'] == 'Tempat Penting')]
Tidak_penting = data_analisa[(data_analisa['Important places'] == 'Tidak Penting')]

# grouping with price
median_place = tempat_penting.groupby(["Important places"])["Price"].median().astype('int64')
median_unknown = Tidak_penting.groupby(["Important places"])["Price"].median().astype('int64')

# Chekc median importent places
median_place = {
          'Places' : ['Tempat Penting','Tidak Penting'],
          'median' : ['1900000000','1400000000']
         }

pd.DataFrame.from_dict(median_place)

# Correlation

Correlation Category VS Category make chisquare

Now we want make method Bonferroni-adjusted p-value for checking correlation with data dummy

# we make chi-square

import researchpy as rp

results_1 = rp.crosstab(data_analisa['Floor'], 
                                 data_analisa['Location'], 
                                 prop= 'col', test= 'chi-square')

results_2 = rp.crosstab(data_analisa['Floor'], 
                                 data_analisa['Location'], 
                                 prop= 'col', test= 'chi-square')


results_3 = rp.crosstab(data_analisa['Important places'], 
                                 data_analisa['Location'], 
                                 prop= 'col', test= 'chi-square')

# look p-value
results_1, results_2, results_3

Check correlation numeric vs category with spearman

from scipy.stats import spearmanr # method for correlation numeric VS categoric

def correlation_spearman (x_column, y_column):
        # input data and Choose column integer
        x = data_analisa[x_column] # input data (change column data input for check)
        y = data_analisa[y_column] # data ouput

        # print data
        corr, _ = spearmanr(x, y)
        print('Spearmans correlation: %.3f' % corr)

# important places
correlation_spearman('Price', 'Important places')
correlation_spearman('Bedroom', 'Important places')
correlation_spearman('Bathroom', 'Important places')
correlation_spearman('Land', 'Important places')
correlation_spearman('Building', 'Important places')
correlation_spearman('Garage', 'Important places')
correlation_spearman('Installment_BCA', 'Important places')
correlation_spearman('Installment_BNI', 'Important places')
correlation_spearman('Installment_Mandiri', 'Important places')

# Correlation Location
correlation_spearman('Price', 'Location')
correlation_spearman('Bedroom', 'Location')
correlation_spearman('Bathroom', 'Location')
correlation_spearman('Land', 'Location')
correlation_spearman('Building', 'Location')
correlation_spearman('Garage', 'Location')
correlation_spearman('Installment_BCA', 'Location')
correlation_spearman('Installment_BNI', 'Location')
correlation_spearman('Installment_Mandiri', 'Location')

# Correlation Floor
correlation_spearman('Price', 'Floor')
correlation_spearman('Bedroom', 'Floor')
correlation_spearman('Bathroom', 'Floor')
correlation_spearman('Land', 'Floor')
correlation_spearman('Building', 'Floor')
correlation_spearman('Garage', 'Floor')
correlation_spearman('Installment_BCA', 'Floor')
correlation_spearman('Installment_BNI', 'Floor')
correlation_spearman('Installment_Mandiri', 'Floor')

#We want check correlation with ANOVA (multiple category vs numerik) we use columns Floor and Location because there are have > two variable

import statsmodels.api as sm
from statsmodels.formula.api import ols

#perform two-way ANOVA
def Anova (column):
            model = ols("""column ~ C(Floor) + C(Location) + 
                        C(Floor):C(Location)""", data=data_analisa).fit()

            output = sm.stats.anova_lm(model, typ=2)
            print(output)
            print('<<<<<<<>>>>>>')

# check Anova correlation
Anova(data_analisa['Price'])
Anova(data_analisa['Bedroom'])
Anova(data_analisa['Bathroom'])
Anova(data_analisa['Land'])
Anova(data_analisa['Building'])
Anova(data_analisa['Garage'])
Anova(data_analisa['Installment_BCA'])
Anova(data_analisa['Installment_BNI'])
Anova(data_analisa['Installment_Mandiri'])

#Check Linear Regression

import pandas as pd
import statsmodels.api as sm

# Specify the dependent variable and the independent variables
dependent_variable = 'Price'
independent_variables = ['Bedroom', 'Bathroom', 'Building', 'Garage', 'Land']

# Add a constant column to the independent variables
data_analisa['intercept'] = 1

# Fit the five-way linear regression model
model = sm.OLS(data_analisa[dependent_variable], data_analisa[independent_variables + ['intercept']]).fit()

# Print the model summary
print(model.summary())

####  summary : 
#### price = 3.016e+09 +  1.086e+08  (Bedroom) +  5.084e+07 (Bathroom)  + 1.009e+07 (Building (m²)) - 2.811e+08 (Garage) - 3.211e+04 Land (m²)

# delete column not use again
data_analisa = data_analisa.drop(columns=['intercept'])

### Hipotesis Uji (t-test)

#First we want split data base on visualization before, How big is the difference unqiue columns.. We make class 0 and class 1 !

# split class by category column Important places (2 variable) 
dataset_places = data_analisa[data_analisa['Important places'] == "Tempat Penting"].copy() 
dataset_not_places = data_analisa[data_analisa['Important places'] != "Tidak Penting"].copy() 

# split class by category column Floor (2 variable)
dataset_floor1 = data_analisa[data_analisa['Floor'] == "Lantai 1"].copy()
dataset_not_floor = data_analisa[data_analisa['Floor'] != "Lantai 1"].copy()

Uji Hipotesa !

h0    = mean price untuk kelas 0 = mean price untuk kelas 1

h1    = mean price untuk kelas 0 != mean price untuk kelas 1

NOTE : with value t-value < t-critical and value p-value < 0.05 then reject h0 

# t critical values for alpha 0.05
scs.t.ppf(0.05, df = (len(data_analisa) - 2))

# perform t-test 2 variable Important places column
scs.ttest_ind(
    dataset_places[dataset_places.Price.isnull() != True].Price,
    dataset_not_places[dataset_not_places.Price.isnull() != True].Price
)

Result: t-value < t-critical and p-value > 0.05 

mean between class 0 and class 1 for price statistically significant

# perform t-test 2 variable Floor column
scs.ttest_ind(
    dataset_floor1[dataset_floor1.Price.isnull() != True].Price,
    dataset_not_floor[dataset_not_floor.Price.isnull() != True].Price
)

Result: t-value < t-critical and p-value < 0.05 

mean between class 0 and class 1 for price then reject h0

# correlation linear
sns.pairplot(data_analisa, hue ='Price',palette="husl")

# to show
plt.show()

# check correlation between the two numeric with kernell method because we look visualization pairplot have not all linier
corr = data_analisa.corr(method='kendall')

# Membuat heatmap korelasi kernel dengan library Seaborn
sns.heatmap(corr, cmap='coolwarm', annot=True, vmin=-1, vmax=1)

### Sumarry

"""EDA : 

1. The highest floor column is floor 1
2. Median from column variabel is : 
    FLoor_1 : 1500000000
    Floor_2 : 2100000000
    Floor_3 : 1400000000
3. The highest data from bin_land and bin_build is : Small
4. Median From bin_land and bin build column is :
    Very Small : 690000000 / 685000000
    Small : 1600000000 / 1700000000
    Medium : 2500000000 / 2800000000
    Large : 4000000000 / 3900000000
    Very Large : 4500000000 / 10000000000
median bin_build greater than bin_land
5. The highest Location columns is Kelurahan
6. The highest Important Places is Tidak Penting and imbalance data ( median Tempat Penting : 1900000000, Tidak Penting : 1400000000

Correlation :
1. Correlation Bivariant from between column category vs category have < p-value, mean is columns between columns category have correlation
2. Correlation Bivariant from between column numeric vs category is low correlation in range (0..)
3. Correlation Anova 2 category (n) vs 1 numeric columns (floor & Important places) that have < p-value is Bedroom (0.002) and Bathroom (0.0008)
4. Correlation Multivariant from all numeric columns the highest is Bedroom vs Bathroom as big as 0.71, Building VS Bedroom as big as 0.61, Building vs Bathroom as big as 0.59, and all other columns have > 0.2 correlation.
5. Relationship Linier regression we obtain is :  
price = 3.016e+09 + 1.086e+08 (Bedroom) + 5.084e+07 (Bathroom) + 1.009e+07 (Building (m²)) - 2.811e+08 (Garage) - 3.211e+04 Land (m²)

Uji Hipotesis :

t-test :
1. Important places - mean between class 0 and class 1 for price statistically significant
2. Floor - mean between class 0 and class 1 for price then reject h0

Correlation Numeric vs Numeric (multivariant):
range correlation is between (0.3 - 0.7) low - strong correlation (+)"""
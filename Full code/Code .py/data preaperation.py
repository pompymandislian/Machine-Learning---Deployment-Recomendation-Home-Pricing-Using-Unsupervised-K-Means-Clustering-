import pandas as pd # for dataframe
import joblib # for save model
from sklearn.model_selection import train_test_split #for splot data

# upload data
df = pd.read_csv('rumah.csv')
df.head(2)

# drop columns
data_cleaning = df.drop(columns=['card-featured__media-section__favorite href','ui-molecules-carousel__item src','ui-molecules-carousel__item src',
                                 'ui-molecules-carousel__item src 2','ui-molecules-carousel__item src 3','ui-molecules-carousel__item src 4',
                                 'ui-molecules-carousel__item src 5', 'ui-molecules-carousel__item src 6', 'ui-molecules-carousel__item src 7',
                                 'card-featured__media-section__caption', 'card-featured__middle-section__header-badge',
                                 'card-featured__middle-section__header','card-featured__middle-section href','ui-atomic-link__icon href 2', 
                                 'card-featured__middle-section','ui-atomic-link__icon href','ui-atomic-link__icon href 2','attribute-info 3',
                                 'attribute-info 3','ui-molecules-carousel__item src 8','ui-molecules-carousel__item src 9','ui-molecules-carousel__item src 10',
                                 'ui-molecules-carousel__item src 11','ui-molecules-carousel__item src 10','attribute-info','ui-atomic-image src','ui-atomic-image src'
                                 ,'ui-atomic-button--children','card-featured__middle-section src','ui-atomic-image src 2','ui-organisms-card-r123-basic__bottom-section__agent 2'
                                 ,'ui-atomic-button__icon href','ui-atomic-link__icon href 3','card-featured__middle-section__price 2'])

data_cleaning.rename(columns = {'card-featured__middle-section__price':'Price', 'card-featured__middle-section 2':'Location',
                              'attribute-text':'Bedroom','attribute-text 2'
                               :'Bathroom', 'attribute-info 2':'Land','attribute-info 4':'Building','card-featured__middle-section 3'
                               :'Description','ui-organisms-card-r123-basic__bottom-section__agent':'Contact Name','truncate'
                               :'Phone Number','attribute-text 3':'Garage'}, inplace = True)

data_cleaning.head(2)

# Handling Missing Nulls

# check null values
data_cleaning.isnull().sum()

# drop all raws is null with use refer to price columns
data_cleaning.dropna(subset=['Price'], inplace=True)

# check missing nulls
data_cleaning["Price"].isnull().sum()

# Sanichek
data_cleaning.isnull().sum()

# replace data null
data_cleaning.fillna("0", inplace = True)

# Sanichek
data_cleaning.isnull().sum()

# Data Replacing

data_cleaning["Floor"] = None
data_cleaning["Important places"] = None
data_cleaning["Floor"][data_cleaning["Description"].str.contains("Lantai 2")] = "Lantai 2"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("lantai 2")] = "Lantai 2"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("lantai 3")] = "Lantai 3"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("Lantai 3")] = "Lantai 3"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("tingkat 2")] = "Lantai 2"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("tingkat 3")] = "Lantai 3"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("2 lantai")] = "Lantai 2"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("2 Lantai")] = "Lantai 2"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("3 Lantai")] = "Lantai 3"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("3 lantai")] = "Lantai 3"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("2 tingkat")] = "Lantai 2"
data_cleaning["Floor"][data_cleaning["Description"].str.contains("3 tingkat")] = "Lantai 3"

data_cleaning["Important places"][data_cleaning["Description"].str.contains("kampus")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("Kampus")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("univ")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("universitas")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("KAMPUS")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("elit")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("Elite")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("mewah")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("Modern")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("MEWAH")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("Mewah")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("Pasar")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("perbelanjaan")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("pasar","belanja")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("belanja")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("supermarket")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("Supermarket")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("wisata")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("Wisata")] = "Tempat Penting"
data_cleaning["Important places"][data_cleaning["Description"].str.contains("pegunungan")] = "Tempat Penting"

# check misisng nulls floor column
data_cleaning["Floor"].isnull().sum()

data_cleaning.isnull().sum()

data_cleaning.fillna("0", inplace = True)

# change values 
data_cleaning['Floor'] = data_cleaning['Floor'].apply(lambda x : x.replace('0','Lantai_1'))
data_cleaning['Important places'] = data_cleaning['Important places'].apply(lambda x : x.replace('0','Tidak Penting'))

data_cleaning.head()

# Clean for land and building columns
data_cleaning['Land'] = data_cleaning['Land'].apply(lambda x: x.strip('m²'))
data_cleaning['Building'] = data_cleaning['Building'].apply(lambda x: x.strip('m²'))

# Clean for location column
data_cleaning['Location'] = data_cleaning['Location'].apply(lambda x : x.replace(',',''))
data_cleaning['Location'] = data_cleaning['Location'].apply(lambda x: x.strip('Malang'))
data_cleaning['Location'] = data_cleaning['Location'].apply(lambda x : x.replace(' ',''))

# Clean for price column
data_cleaning['Price'] = data_cleaning['Price'].apply(lambda x : x.replace('Rp',''))
data_cleaning['Price'] = data_cleaning['Price'].apply(lambda x : x.replace(',',''))
data_cleaning['Price'] = data_cleaning['Price'].apply(lambda x : x.replace(' ',''))
data_cleaning['Price'] = data_cleaning['Price'].apply(lambda x : x.replace('Miliar','00000000'))
data_cleaning['Price'] = data_cleaning['Price'].apply(lambda x : x.replace('Juta','000000'))

data_cleaning['Location'].unique()

# simplify data location
data_cleaning["Location"][data_cleaning["Location"].str.contains("Blimbing")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Dau")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Lowokwaru")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Sukun")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Lawang")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Pakis")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Cemorokandang","Rampal Celaket")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Sawojajar")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Bareng")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Singosari")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Karangploso")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Oro-OroDowo")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tlogomas")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Wagir")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tunggulwulung")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Kebonsari")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Bumiayu")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Purwantoro")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Karangploso")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Balearjosari")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Arjosari")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("erjosari")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Pandanwangi")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Klojen")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Kedungkandang")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Kota")] = "Kota"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Sumbersari")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Pakisaji")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Gadang")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Kasin")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Gunung-gunung")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tasikmadu")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tumpang")] = "Kecamatan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Poncokusumo")] = "Kecamatan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Arjowinangun")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Buring")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Polowijen")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("ulyorejo")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tulusrejo")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("ojolangu")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Kepanjen")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Gondanglegi")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tanjungrejo")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Dinoyo")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Kotalama")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tlogowaru")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Turen")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Sulfat")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("SoekarnoHatta")] = "Kecamatan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Bandulan")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tunjungsekar")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Lesanpuro")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("RampalCelaket")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("KarangBesuki")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Araya")] = "Kelurahan"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Batu")] = "Kota"
data_cleaning["Location"][data_cleaning["Location"].str.contains("PermataJingga")] = "Kota"
data_cleaning["Location"][data_cleaning["Location"].str.contains("Sengkaling")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("GriyaShanta")] = "Kecamatan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("PisangCandi")] = "Kelurahan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("VillaPuncakTidar")] = "Kecamatan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("DiengTidar")] = "Kecamatan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Dieng")] = "Kecamatan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Tidar")] = "Kecamatan" 
data_cleaning["Location"][data_cleaning["Location"].str.contains("Buah-buahan")] = "Kecamatan" 

# check data unique
data_cleaning['Location'].unique()

data_cleaning.head(2)

# check data type
data_cleaning.dtypes

# change data type
data_cleaning['Price'] = data_cleaning['Price'].astype('int64')
data_cleaning['Bedroom'] = data_cleaning['Bedroom'].astype('int64')
data_cleaning['Bathroom'] = data_cleaning['Bathroom'].astype('int64')
data_cleaning['Land'] = data_cleaning['Land'].astype('int64')
data_cleaning['Building'] = data_cleaning['Building'].astype('int64')
data_cleaning['Garage'] = data_cleaning['Garage'].astype('int64')
data_cleaning['Phone Number'] = data_cleaning['Phone Number'].astype('str')

data_cleaning.info()

# replace 62 to 0 in phone number column
data_cleaning['Phone Number'] = data_cleaning['Phone Number'].apply(lambda x : x.replace('62','0'))
data_cleaning['Phone Number'] = data_cleaning['Phone Number'].apply(lambda x : x.replace('.0',''))

data_cleaning.head(2)

data_cleaning.info()

# make calcultaion for 20% dp
data_cleaning['calculation_bank'] = (data_cleaning['Price'])-(data_cleaning['Price']*0.2)

# make calculation for bunga fixed
data_cleaning['Installment_BCA'] = data_cleaning['calculation_bank']*0.07*20/240
data_cleaning['Installment_BNI'] = data_cleaning['calculation_bank']*0.0276*20/360
data_cleaning['Installment_Mandiri'] = data_cleaning['calculation_bank']*0.045*20/240

# drop columns calculation_bank
data_cleaning = data_cleaning.drop(columns='calculation_bank')

data_cleaning.info()

# change data type installment to int64
data_cleaning['Installment_BCA'] = data_cleaning['Installment_BCA'].astype('int64')
data_cleaning['Installment_BNI'] = data_cleaning['Installment_BNI'].astype('int64')
data_cleaning['Installment_Mandiri'] = data_cleaning['Installment_Mandiri'].astype('int64')

data_cleaning.info()

# reset index
data_cleaning.reset_index(inplace = True, drop = True)

data_analisa = data_cleaning[['Price','Bedroom','Bathroom','Land','Building','Garage','Important places','Location',
                              'Installment_BCA','Installment_BNI','Installment_Mandiri','Floor']]

data_analisa.describe()

# save data
joblib.dump(data_analisa, "D:/BOOTCAMP/project/(Block 4) ML Process/Data_project.csv")

# Data Spliting

data_train, data_test = train_test_split(data_analisa,
                                         test_size = 0.2,
                                         random_state = 42
                                        )
data_valid, data_test = train_test_split(data_test,
                                         test_size = 0.5, 
                                         random_state = 42
                                        )

data_train

data_train.shape
data_valid.shape
data_test.shape

# Save data 

joblib.dump(data_train, "D:/BOOTCAMP/project/(Block 4) ML Process/data_train.csv")
joblib.dump(data_valid, "D:/BOOTCAMP/project/(Block 4) ML Process/data_valid.csv")
joblib.dump(data_test, "D:/BOOTCAMP/project/(Block 4) ML Process/data_test.csv")
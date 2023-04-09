import pytest
import joblib
import numpy as np

# load data 
data_test = joblib.load("D:/BOOTCAMP/project/(Block 4) ML Process/data feature/data_pytest.csv")

def test_data_columns():
    # Check the data columns
    assert data_test.shape == (1960, 18) 
    assert data_test.columns.all() == 'Cluster' # check final columns
    assert data_test.columns.any() == 'Installment_BCA' # check start columns

def test_dtypes():
    # Check the data types of the columns
    assert data_test['Cluster'].dtypes == 'int32'
    assert data_test['Installment_BCA'].dtypes == 'float64'
    assert data_test['Bathroom'].dtypes == 'float64'
    assert data_test['Installment_Mandiri'].dtypes == 'float64'
    assert data_test['Building'].dtypes == 'float64'
    assert data_test['Price'].dtypes == 'float64'
    assert data_test['Installment_Mandiri'].dtypes == 'float64'
    assert data_test['Garage'].dtypes == 'float64'
    assert data_test['Bedroom'].dtypes == 'float64'
    assert data_test['Installment_BNI'].dtypes == 'float64'
    assert data_test['Land'].dtypes == 'float64'
    assert data_test['Important_places'].dtypes == 'float64'
    assert data_test['No_important_places].dtypes == 'float64'
    assert data_test['Location_Kecamatan'].dtypes == 'float64'
    assert data_test['Location_Kelurahan'].dtypes == 'float64'
    assert data_test['Location_Kota'].dtypes == 'float64'
    assert data_test['Floor_Lantai_1'].dtypes == 'float64'
    assert data_test['Floor_Lantai_2'].dtypes == 'float64'
    assert data_test['Floor_Lantai_3'].dtypes == 'float64'

def test_mean():
    assert data_test['Price'].mean() == 0.9999838591417305
    assert data_test['Cluster'].mean() == 0.6423469387755102
    assert data_test['Installment_BCA'].mean() == 0.004666590864539799
    assert data_test['Bathroom'].mean() == 2.5756128133858784e-09
    assert data_test['Installment_Mandiri'].mean() == 0.0029999515774251917
    assert data_test['Building'].mean() == 1.6463663276915264e-07
    assert data_test['Installment_Mandiri'].mean() == 0.0029999515774251917
    assert data_test['Garage'].mean() == 7.053581499095706e-10
    assert data_test['Bedroom'].mean() == 4.014538995342758e-09
    assert data_test['Installment_BNI'].mean() == 0.0012266463890922461
    assert data_test['Land'].mean() == 2.8691986028512193e-07
    assert data_test['Important_places'].mean() == 2.423981996620613e-10
    assert data_test['No_important_places'].mean() == 9.600455120009895e-10
    assert data_test['Location_Kecamatan'].mean() == 9.260840574066566e-10
    assert data_test['Location_Kelurahan'].mean() == 1.4024788269056826e-10
    assert data_test['Location_Kota'].mean() == 1.3611177156582615e-10
    assert data_test['Floor_Lantai_1'].mean() == 1.083462353056451e-09
    assert data_test['Floor_Lantai_2'].mean() == 1.0231483557682107e-10
    assert data_test['Floor_Lantai_3'].mean() == 1.6666523029778762e-11



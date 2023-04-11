## Machine-Learning | Deployment Recomendation Home Pricing Using Unsupervised K-Means Clustering

Created by : Pompy Mandislian

link for use this model :

<h3> Documentation : </h3>

<h4> <b> Requariments : </b> </h4>

**for Analysis and Machine Learning** :
<p>
  <img align="center" src="Image/python.png" width="200" height="100" />
  <img align="center" src="Image/scikit.png" width="200" height="100" />
  <img align="center" src="Image/seaborn.png" width="200" height="100" />
  <img align="center" src="Image/matplotlip.png" width="200" height="100" />
</p>

**for Deployment** :
<p>
  <img align="center" src="Image/aws.png" width="150" height="100" />
  <img align="center" src="Image/docker.jpg" width="150" height="100" />
  <img align="center" src="Image/streamlit.png" width="150" height="100" />
  <img align="center" src="Image/fastapi.png" width="150" height="100" />
  <img align="center" src="Image/pytest.png" width="150" height="100" />
</p>

<h3> <b> Workflow General <b> </h3>
<img align="center" src="image_flowchart/worklow general.png" width="700" height="300" />
  
<h2> <b> Background Project <b> </h2>
  
<p> 
On a website, Rumahs123 provides home buying or rental services, the prices of houses sold are based on facilities, area, and location. At this time, users are still doing searches manually without providing the features they want to search for, such as how much land they want, facilities, and house prices or mortgage payments. Therefore, so that users can find their desired housing needs more easily, a recommendation system was created for these business needs. This system is based on machine learning with the concept of unsupervised clustering so that it can group the features needed by the user. The making of this system will be directed to the city of Malang by describing it in the Kelurahan, District, and City areas. This system will later be accessible to the public so that it can be used by anyone.
</p>

<h2> <b> Problem Statement <b> </h2>
How to make a recommendation system for users?
</li><li> How can a user find out which recommendations are suitable for their needs?    
 <p> How to user can access this system ? </p>

  <h2> Architecture </h2>
  <li> Block Diagram Data Preaperation </li>
  <img align="center" src="image_flowchart/Block Diagram Data Preposesing.png" width="700" height="250" />
  <li> Block Diagram Data Prepocessing and Feature Engineering </li>
  <img align="center" src="image_flowchart/Block Diagram Data Prep and Feature Engineering.png" width="700" height="250" />
  <li> Block Diagram Modeling Unsupervised </li>
  <img align="center" src="image_flowchart/Block Diagram Model Unsupervised.png" width="700" height="250" />
  <h3> Format Massage Prediction FastAPI </h3>
   <p> {"message":"This is the homepage of the API "} </p>
   <img align="left" src="Image/fastapi input.jpg" width="500" height="250" />
    <p> Output of Prediction is  </p>

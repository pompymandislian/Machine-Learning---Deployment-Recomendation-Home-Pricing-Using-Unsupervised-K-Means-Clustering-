## Machine-Learning | Deployment Recomendation Home Pricing Using Unsupervised K-Means Clustering

Created by : Pompy Mandislian

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
<li> Block Diagram Data Preaperation - Pipeline </li>
<img align="center" src="image_flowchart/Block Diagram Data Prep and Feature Engineering.png" width="700" height="250" />
<li> Block Diagram Data Prepocessing and Feature Engineering </li>
<img align="center" src="image_flowchart/Block Diagram Data Prep and Feature Engineering.png" width="700" height="250" />
<li> Block Diagram Modeling Unsupervised </li>
<img align="center" src="image_flowchart/Block Diagram Model Unsupervised.png" width="700" height="250" />

<h3> Format Massage Prediction FastAPI </h3>
<p> {"message":"This is the homepage of the API "} </p>
<p> <img src="Image/fastapi input.jpg" width="500" height="250" /> </p>
<p> Output of Prediction is {"prediction Cluster is": 1} </p>

<h3> How to Run Model in Localhost </h3>
<li> Import/built file Streamlit and FastAPI to Docker Image >> docker built -t (name image) . </li> 
<li> Run Container Streamlit and FastAPI >> docker run -p (name images in docker) </li>
<li> Stop Image and container streamlit and FastAPI in the docker </li>
<li> Compose file docker compose >> docker compose up </li>
<li> Run docker compose in new images (fill streamlit and fastapi) </li>
<li> Input data in the streamlit </li>

<h2> Image docker </h2>
<p> <img src="Image/docker images.jpg" width="700" height="250" /> </p>
<p> <img src="Image/docker container.jpg" width="700" height="350" /> </p>
<li> Result in streamlit </li>
<p> <img src="Image/streamlit output.jpg" width="750" height="100" /> </p>

<li> Datasheet using feature engineering MINMAX Scaller </li>

<h2> Conclution and Next Improvement </h2>  
<li> Using minmax scaller is the best Feauture Engineering for K-means Model in this datasheet </li>
<li> Next Project can improvement to server, so that access in public </li>
<li> Can improvement about feauture frontend like add identity user and send data inputan to database </li>

<h2> Referensi </h2>
<li> Pacmann Class </li>
<li> FastAPi, Docker and Streamlit Doc </li>
<li> Youtube Deployment FastAPI and Streamlit </li>

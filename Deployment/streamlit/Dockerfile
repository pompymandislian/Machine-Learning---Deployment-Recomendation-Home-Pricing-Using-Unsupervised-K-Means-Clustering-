# Base image
FROM python:3.9.12

# Set the working directory
WORKDIR /streamlit

# Copy the requirements file into the container at /app
COPY requirements.txt /streamlit/
COPY data_fix.pkl /streamlit/

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY . /streamlit/

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "streamlit.py"]
# Use an official Python runtime as a parent image
FROM python:3.9.12-slim-buster

# Set the working directory to /app
WORKDIR /main

# Copy the requirements file into the container at /app
COPY requirements.txt /main/
COPY data_fix.pkl /main/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /main/

# Expose the port that the app will run on
EXPOSE 8000

# Set the environment variable for the app
ENV APP_ENV=Development

# Run the command to start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
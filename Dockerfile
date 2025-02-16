# Use an official lightweight Python image.
FROM python:3.11-slim-buster

# Set the working directory in the container.
WORKDIR /app

# Copy the current directory contents into the container at /app.
COPY . /app

# Update the package list and install awscli.
RUN apt-get update -y && apt-get install awscli -y

# Install Python dependencies.
RUN pip install -r requirements.txt

# Run app.py when the container launches.
CMD ["python3", "app.py"]


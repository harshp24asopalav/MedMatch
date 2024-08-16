# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Create logs directory
RUN mkdir -p /app/logs

# Expose the ports for Flask (5000) and Streamlit (8501)
EXPOSE 5000
EXPOSE 8501

# Command to run both the Flask app and the Streamlit app
# Here we use '&&' to run both commands in parallel
CMD ["sh", "-c", "python app.py & streamlit run frontend/interface.py"]

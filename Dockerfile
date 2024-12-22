# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the Streamlit entry point
ENTRYPOINT ["streamlit", "run"]

# Run the Streamlit app
CMD ["app.py", "--server.port=8501", "--server.enableCORS=false", "--server.address=0.0.0.0"]

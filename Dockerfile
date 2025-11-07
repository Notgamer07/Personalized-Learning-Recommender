FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files to the container
COPY . /app

# Install dependencies
# Using pip with no cache to reduce image size
RUN pip install --no-cache-dir \
    streamlit \
    joblib \
    pandas \
    numpy \
    scikit-learn \
    huggingface_hub   # <-- Added this line

# /data is automatically mounted when Persistent Storage is enabled
RUN mkdir -p /data && chmod 777 /data

# Expose the port Streamlit runs on (required for Spaces)
EXPOSE 7860

# Environment variables to make Streamlit work inside Docker
ENV STREAMLIT_PORT=7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit app (update 'app.py' if your file has a different name)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
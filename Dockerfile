

# Use a lightweight Python image
FROM python:3.10-slim

# Set NLTK to use a writable folder
ENV NLTK_DATA=/tmp/nltk_data


# Set working directory
WORKDIR /code

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only relevant project files
COPY . /code

# Expose the port (Hugging Face expects port 7860)
EXPOSE 7860

# Command to run the FastAPI app
# (assuming your app is in main.py and the FastAPI instance is named "app")
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

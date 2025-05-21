# Sentiment Analysis API

A FastAPI application that provides sentiment analysis for movie reviews and retrieves similar reviews from a vector database.

Note: Models and datasets are not included in the repository to reduce its size.  
However, they will be automatically downloaded when running the code.  
You can find the complete implementation at the following link:
https://drive.google.com/file/d/1OFoj1enMY0XWfWZZR0MSCWsGPvZhKi8U/view?usp=sharing

## Setup

1. Ensure you have the following resource directories in place:
   - `resources/classification_model` - containing the DistilBERT sentiment analysis model
   - `resources/embedding_model` - containing the sentence transformer model
   - `resources/vectorial_database` - containing the ChromaDB vectorial database

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## API Endpoints

### Sentiment Analysis

**Endpoint:** `POST /sentiment/analyze`

**Request Body:**
```json
{
    "review_text": "This movie exceeded my expectations..."
}
```

**Response:**
```json
{
    "sentiment": "positive",
    "confidence": 0.92,
    "similar_reviews": [
        {
            "text": "Similar review text...",
            "label": "positive",
            "similarity": 0.85
        },
        {
            "text": "Another similar review...",
            "label": "positive", 
            "similarity": 0.75
        },
        {
            "text": "Yet another similar review...",
            "label": "negative",
            "similarity": 0.65
        }
    ]
}
```

## Local Development

### Running the API Locally

To start the API server, run the following command from the `api` directory:

```bash
python main.py
```

Or using uvicorn:

```bash
uvicorn main:app --reload
```

This will start the server at http://localhost:8000

### API Documentation

Once the server is running, you can access the auto-generated Swagger documentation at:
http://localhost:8000/docs

### Testing the API

You can test the API using curl:

```bash
curl -X POST "http://localhost:8000/sentiment/analyze" -H "Content-Type: application/json" -d '{"review_text": "This movie exceeded my expectations with its brilliant screenplay and outstanding performances."}'
```

Or using the provided test script:

```bash
python test_api.py
```

## Deployment with Docker

This section provides instructions for deploying the API using Docker and Docker Compose.

### 1. Prerequisites

- Install Docker and Docker Compose on your system
  - [Docker Installation Guide](https://docs.docker.com/get-docker/)
  - [Docker Compose Installation Guide](https://docs.docker.com/compose/install/)

### 2. Build and Run with Docker Compose

To deploy the application with Docker Compose, run the following command from the root directory:

```bash
# For first-time deployment or after code changes
docker-compose up --build -d

# For subsequent runs without code changes
docker-compose up -d
```

These commands will:
- Build the Docker image for the API using the provided Dockerfile
- Pull the ChromaDB image from the registry
- Create and start containers for both services
- Set up the required volumes and network
- Automatically populate ChromaDB with sample reviews if the database is empty

### 3. Data Persistence

The Docker configuration is set up to use your local `resources/vectorial_database` directory for ChromaDB storage. This means:

- Your existing vector database (if any) will be used by the Docker containers
- Any data added to the database through the Docker containers will persist locally
- You can use the same database in both local and Docker environments

### 4. Accessing the Services

- The Sentiment Analysis API will be available at: `http://localhost:8000`
- The API documentation (Swagger UI): `http://localhost:8000/docs`
- ChromaDB service will be running on port 8001

### 5. Monitoring Logs

To view logs from the running containers:

```bash
# View logs from all services
docker-compose logs

# View logs from a specific service
docker-compose logs api
```

### 6. Stopping the Services

To stop the running containers while preserving the data:

```bash
docker-compose down
```

To stop the containers and remove the volumes (will delete all data):

```bash
docker-compose down -v
```

### 7. Development with Docker

The Docker configuration includes volume mounts for the application code, allowing you to make changes to the code without rebuilding the image. After making changes to the code, the API will automatically reload thanks to the `--reload` option in Uvicorn.

### 8. GPU Support (Optional)

If your machine has a GPU and you want to use it for model inference:

1. Install the NVIDIA Container Toolkit ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
2. Uncomment the GPU-related lines in `docker-compose.yml`
3. Restart the containers with `docker-compose up -d`

## Deployment with IIS

This section provides instructions for deploying the API on a Windows server using IIS.

### 1. Install IIS

If you do not already have IIS, follow these steps:
- Go to "Control Panel" > "Programs" > "Turn Windows features on or off"
- Check the box for "Internet Information Services" and ensure "Web Management Tools" and "World Wide Web Services" are selected
- Click "OK" to install

### 2. Install Python 3.11.9

Download and install Python 3.11.9 from:
https://www.python.org/downloads/release/python-3119/

Ensure you select "Add Python to PATH" during installation so IIS can locate it.

### 3. Install Miniconda

Download and install Miniconda from:
https://docs.anaconda.com/miniconda/

Make sure to activate: "Add Miniconda3 to my PATH environment variable".

### 4. Locate the ML Application

Place all API files in the appropriate web directory, typically:
```
C:\inetpub\wwwroot\SentimentAnalysisAPI
```

### 5. Create a Conda Environment

```bash
conda create --name sentiment_api python=3.11.9
conda activate sentiment_api
cd C:\inetpub\wwwroot\SentimentAnalysisAPI
pip install -r requirements.txt
```

### 6. Install httpPlatformHandler

Download and install the HTTP Platform Handler from:
https://www.iis.net/downloads/microsoft/httpplatformhandler

### 7. Configure IIS Application Pool

- Open IIS Manager
- Create a new Application Pool for your ML application
- Set the .NET Framework version to "No Managed Code" and the pipeline mode to "Integrated"

### 8. Create a New IIS Website

- Right-click on "Sites" in IIS Manager and choose "Add Website"
- Set the site name (e.g., "SentimentAnalysisAPI")
- Select the Application Pool you created previously
- Set the physical path to where your API is located
- Configure the IP address and port number

### 9. Create a web.config File

Create a `web.config` file in your application's root directory with the following content (adjust paths as needed):

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified" />
    </handlers>
    <httpPlatform processPath="C:\Users\[YourUsername]\AppData\Local\miniconda3\envs\sentiment_api\python.exe"
                  arguments="-m uvicorn main:app --host 127.0.0.1 --port %HTTP_PLATFORM_PORT%"
                  stdoutLogEnabled="true"
                  stdoutLogFile="C:\inetpub\logs\sentiment_api_stdout.log"
                  startupTimeLimit="120">
      <environmentVariables>
        <environmentVariable name="PYTHONPATH" value="C:\inetpub\wwwroot\SentimentAnalysisAPI" />
      </environmentVariables>
    </httpPlatform>
  </system.webServer>
</configuration>
```

### 10. Set Permissions

Ensure the IIS_IUSRS user group has appropriate permissions:
- On your application folder: `C:\inetpub\wwwroot\SentimentAnalysisAPI`
- On your conda environment folder: `C:\Users\[YourUsername]\AppData\Local\miniconda3\envs\sentiment_api`

The user should have these permissions:
- Read & Execute
- List Folder Contents
- Read

### 11. Configure applicationHost.config

Modify the `applicationHost.config` file located at `C:\Windows\System32\inetsrv\config\applicationHost.config`:

Change the `overrideModeDefault` attribute for the `handlers`, `httpErrors`, and `httpPlatform` sections from "Deny" to "Allow":

```xml
<sectionGroup name="system.webServer">
    <section name="handlers" overrideModeDefault="Allow" />
    <section name="httpErrors" overrideModeDefault="Allow" />
    <section name="httpPlatform" overrideModeDefault="Allow" />
</sectionGroup>
```

### 12. Verify the Installation

- Open a web browser and navigate to `http://[YourServerAddress]:[Port]/docs` to access the FastAPI interactive documentation
- Check the IIS logs and the specified FastAPI logs for any errors if the application does not load

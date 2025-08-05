# Pizza Store Scooper Violation Detection

## Overview

This project is focused on building a Computer Vision system for a pizza store to monitor hygiene protocol compliance.Specifically, the system detects whether workers are using a scooper when picking up certain ingredients from designated areas (ROIs - Regions of Interest).Any action of picking up these ingredients without a scooper will be flagged as a violation.

## Features

- **Real-time Violation Detection**: Analyzes video streams to flag hygiene protocol violations as they happen.
- **Microservices Architecture**: The system is decoupled into independent services for scalability and maintainability.
- **Database Logging**: Every confirmed violation is saved to a persistent database with relevant metadata.
- **Web-Based User Interface**: A live dashboard displays the video feed, all AI detections, and a real-time count of violations with on-screen alerts.


## Setup Instructions

These instructions are for setting up and running the project locally on a Windows machine.

### 1. Prerequisites

- **Git**: To clone the repository.  
- **Python**: Python 3.9+ is required.

### 2. Clone the Repository

Open your terminal and run the following command:

```bash
git clone https://github.com/farahahmed09/Pizza-Store-Scooper-Violation-Detection.git
cd Pizza-Store-Scooper-Violation-Detection
```

### 3. Setup Python Virtual Environment

Using a virtual environment is highly recommended.

```bash
# Create the virtual environment
python -m venv myenv

# Activate the virtual environment (on Windows)
myenv\Scripts\activate
```

### 4. Install Python Dependencies

Install the `requirements.txt` file in the root of the project.
this will install all the required libraries.

```
pip install -r requirements.txt
```


### 5. Install Erlang and RabbitMQ

The system uses RabbitMQ as a message broker, which requires Erlang.

#### A. Install Erlang:

Go to the official Erlang downloads page:  
https://www.erlang.org/downloads

Download and run the latest Windows installer, accepting the default settings.

#### B. Install RabbitMQ:

Go to the RabbitMQ releases page:  
https://github.com/rabbitmq/rabbitmq-server/releases/

Download the `.exe` installer and run it, accepting the default settings.

## How to Run the Project

Follow these steps in order to launch the application.

### 1. Start the RabbitMQ Service

Start the RabbitMQ service from your Windows Start Menu.

Enable the management dashboard by running the following command in the "RabbitMQ Command Prompt (sbin)":

```bash
rabbitmq-plugins enable rabbitmq_management
```

Restart the RabbitMQ service.

Verify it's running by opening a web browser to:  
http://localhost:15672/  
Login with:  
**Username**: `guest`  
**Password**: `guest`

### 2. Setup the Database

Run the database setup script once to create a clean `violations.db` file.

```bash
python setup_database.py
```

### 3. Run the Microservices



#### 1. Open RabbitMQ Management Dashboard

Open your web browser and go to:

```
http://localhost:15672
```
You should see the RabbitMQ management dashboard.

---


#### 2. You will need to open three separate terminals. In each, navigate to the project root and activate your virtual environment.

##### Terminal 1: Start the Detection Service

```bash
python services/detection_service/main.py
```

##### Terminal 2: Start the Streaming Service

```bash
python services/streaming_service/main.py
```

##### Terminal 3: Start the Frame Reader

```bash
python services/frame_reader/main.py
```

### 4. View the Output

Open your web browser and go to:

```
http://localhost:8000
```


## Technology Stack

* **Backend**: Python
* **Computer Vision**: OpenCV, Ultralytics YOLO
* **Microservices and API**: FastAPI, WebSockets, RabbitMQ, Pika, Uvicorn
* **Database**: SQLite , Pillow
* **Frontend**: HTML, CSS

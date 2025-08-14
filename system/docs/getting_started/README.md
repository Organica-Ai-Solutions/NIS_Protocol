# Getting Started with the NIS Protocol

This guide will walk you through the process of setting up and running the NIS Protocol on your local machine using Docker.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Docker Compose](https://docs.docker.com/compose/install/)
*   [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## Installation

1.  **Clone the Repository**:
    Open your terminal and clone the repository to your local machine:
    ```sh
    git clone <repository-url>
    cd NIS_Protocol
    ```

2.  **Set Up Environment Variables**:
    The project uses a `.env.example` file to manage environment variables. Copy this file to `.env` and modify it as needed.
    ```sh
    cp .env.example .env
    ```

## Running the Application

The project includes a set of scripts to manage the application's lifecycle.

*   **Start the Application**:
    To build the Docker images and start the application, run the following command:
    ```sh
    ./start.sh
    ```
    This script will start all the services defined in the `docker-compose.yml` file.

*   **Stop the Application**:
    To stop the application, run the following command:
    ```sh
    ./stop.sh
    ```
    This will stop and remove the running containers.

*   **Reset the Application**:
    To stop the application and remove all associated data, including Docker volumes, run the following command:
    ```sh
    ./reset.sh
    ```
    **Warning**: This command will permanently delete all data stored in the Docker volumes.

## Verifying the Installation

Once the application is running, you can verify that it is working correctly by accessing the following endpoints:

*   **Health Check**:
    Open your browser and navigate to `http://localhost:8000/health`. You should see a JSON response with the status "healthy".

*   **API Root**:
    Navigate to `http://localhost:8000/`. You should see a JSON response with information about the NIS Protocol. 
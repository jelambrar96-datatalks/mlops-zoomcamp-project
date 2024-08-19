"""
simple Python script that you can use to check if a server is running on port 4200.
This script will attempt to make an HTTP request to http://localhost:4200 and 
print whether the server is up or not.
"""

import os
import requests

# Default URL for Prefect UI if the environment variable is not set
PREFECT_UI_URL = os.getenv("PREFECT_UI_URL", "http://localhost:4200")


def check_server_status(url: str) -> None:
    """
    Checks the server status by sending a GET request to the specified URL.

    Args:
        url (str): The URL of the server to check.

    Returns:
        None

    Raises:
        requests.exceptions.ConnectionError: If there is a connection error.
        requests.exceptions.RequestException: For any request-related errors.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"Server is up and running on {url}")
        else:
            print(f"Server responded with status code: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"Failed to connect to {url}. The server might not be running.")
        raise e
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        raise e


if __name__ == "__main__":
    # Retrieve the Prefect UI URL from the environment or use the default
    temp_url = PREFECT_UI_URL
    check_server_status(temp_url)

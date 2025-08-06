"""connect_clients.py
This module provides functions to connect to the CAVE and FlyWire clients.
It retrieves authentication tokens from environment variables or prompts the user to enter them.
"""

# libraries
from decouple import Config, RepositoryEnv
from caveclient import CAVEclient
from fafbseg import flywire
from getpass import getpass


def connect_cave_client(env_path=".env"):
    """Connects to the CAVE client using the provided environment variables."""
    config = Config(RepositoryEnv(env_path))

    # Get the CAVE_AUTH_TOKEN from the environment variables
    cave_token = config("CAVE_AUTH_TOKEN", default=None)

    if not cave_token:
        print("No CAVE token found in the .env file.")
        temp_token = getpass(
            "Token not found in the .env file. Please enter your CAVE token (if you don't have one leave if blank and instructions will appear): "
        )
        if temp_token:
            cave_token = temp_token
        else:
            print("No CAVE token entered. Follow the information below to get a token.")
            CAVEclient.auth.get_new_token()
            return  # Exit if no token is provided
    if cave_token:
        # Initialize the CAVE client
        client = CAVEclient("flywire_fafb_public")
        auth = client.auth
        tk = auth.token
        if not tk:
            print("Adding the token to your account. You only need to do this once.")
            auth.save_token(cave_token)
        else:
            print("CAVE token already exists in your account. No need to add it again.")
    return client


def connect_flywire_client(env_path=".env"):
    """Connects to the FlyWire client using the provided environment variables."""
    config = Config(RepositoryEnv(env_path))

    # Get the FLYWIRE_AUTH_TOKEN from the environment variables
    flywire_token = config("FLYWIRE_AUTH_TOKEN", default=None)
    tk = flywire.get_chunkedgraph_secret()

    if not tk:
        print("No FLYWIRE token saved. Setting it with the value from the .env file.")
        if not flywire_token:
            flywire_token = getpass(
                "Token not found in the .env file. Please enter your FLYWIRE token (if you don't have one leave if blank and instructions will appear): "
            )
            if not flywire_token:
                print(
                    "No FLYWIRE token entered. Follow the information below to get a token."
                )
                flywire.get_new_token()
                return  # Exit if no token is provided
            else:
                print("Setting the FLYWIRE token with the value from the .env file.")
                # Save the secret
                flywire.set_chunkedgraph_secret(flywire_token)
    else:
        print("FLYWIRE token already exists in your account. No need to add it again.")

    return flywire

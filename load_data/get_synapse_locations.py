
#libraries
import os
import pandas as pd
from decouple import config, Config, RepositoryEnv
from caveclient import CAVEclient
from getpass import getpass


# Constants
ENV_PATH = ".env"
from constants import FAFB_VOXEL_RESOLUTION_NM


def get_fafb_cave_client()-> CAVEclient:
    """
    Initialize the FAFB CAVE client with the CAVE authentication token.
    """
    
    # return client
    config = Config(RepositoryEnv(ENV_PATH))
    # Get the CAVE_AUTH_TOKEN
    cave_token = config("CAVE_AUTH_TOKEN", default=None)
    if not cave_token:
        print("No CAVE token found in the .env file.")
        temp_token = getpass("Token not found in the .env file. Please enter your CAVE token (if you don't have one leave if blank and instructions will appear when you press enter): ")
        if temp_token:
            cave_token = temp_token
        else:
            print("No CAVE token entered. Follow the information below to get a token.")
            CAVEclient.auth.get_new_token()

    if cave_token:
        # Initialize the CAVE client
        client = CAVEclient('flywire_fafb_public')
        auth = client.auth
        # save the token if it is not already saved
        if not auth.token:
            print("Using the CAVE token from the .env file.")
            auth.save_token(cave_token, overwrite=True)
        else:
            print(f"Using existing CAVE token: {auth.token[:5]}... (truncated for security)")
    else:
        print("No CAVE token provided. Please set the CAVE_AUTH_TOKEN in the .env file or provide it interactively.")
        # Initialize the CAVE client without a token
        client = CAVEclient('flywire_fafb_public', auth_required=False)
        print("CAVE client initialized without authentication. Some features may be limited.")
    
    return client


def get_presynaptic_targets(client: CAVEclient, root_id: int, vx_resolution: list = None) -> pd.DataFrame:
    """
    Get the presynaptic targets for a given root ID from the FAFB synapses table.
    
    Parameters:
    - client: CAVEclient instance connected to the FAFB database.
    - root_id: The root ID for which to fetch presynaptic targets.
    - vx_resolution: Optional voxel resolution, if not provided it will be fetched from the metadata.
    
    Returns:
    - DataFrame containing presynaptic target information.
    """
    
    # Fetch the presynaptic targets
    pre_df = client.materialize.query_table(
        table='synapses_nt_v1',
        filter_in_dict={'pre_pt_root_id': [root_id]},
  )
    
    if vx_resolution is None:
        if FAFB_VOXEL_RESOLUTION_NM:
            print("Using default voxel resolution from constants.")
            # Use the default voxel resolution defined in constants.py
            vx_resolution = FAFB_VOXEL_RESOLUTION_NM
        else:
            # Fetch the voxel resolution from the table metadata if not provided
            print("No voxel resolution provided, fetching from table metadata. THIS IS NOT RECOMMENDED!")
            vx_resolution = client.materialize.get_table_metadata('synapses_nt_v1')["voxel_resolution"]
    
    # Prepare the output DataFrame
    output_df = pre_df[["pre_pt_root_id", "post_pt_root_id", "pre_pt_position", "pre_pt_supervoxel_id"]].copy()
    output_df = output_df.rename(columns={
        "pre_pt_root_id": "root_id",
        "post_pt_root_id": "post_synaptic_root_id",
        "pre_pt_position": "xyz_locations_vx_units",
        "pre_pt_supervoxel_id": "presynaptic_supervoxel_id"
    })
    
    # Add voxel resolution
    output_df["vx_resolution"] = [vx_resolution] * output_df.shape[0]
    
    return output_df

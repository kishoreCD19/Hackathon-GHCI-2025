#security.py (Used by api_main.py)

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import config # Import the centralized configuration

# Define the security scheme: expecting the API key in a custom header
API_KEY_HEADER_SCHEME = APIKeyHeader(name=config.API_KEY_HEADER, auto_error=True)

# Replace this with a secure method of storing and fetching API keys (e.g., database, Vault)
# For demonstration, we use a single hardcoded key
ALLOWED_API_KEY = "super_secret_master_key_123"

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER_SCHEME)):
    """
    Dependency function to validate the API key provided in the request header.
    """
    if api_key_header == ALLOWED_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

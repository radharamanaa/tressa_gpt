import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(
    title="Model Downloader API", 
    description="API to download .pt model files"
)

# Directory where the .pt files are stored. 
# Defaults to the current working directory if not specified.
MODELS_DIR = os.getenv("MODELS_DIR", ".")

@app.get("/download/{filename}")
async def download_pt_file(filename: str):
    """
    Download a .pt file by providing its name.
    """
    # Ensure only .pt files are requested
    if not filename.endswith(".pt"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file extension. Only .pt files are allowed."
        )
    
    # Prevent directory traversal attacks by extracting just the basename
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(MODELS_DIR, safe_filename)
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    # Return the file as a downloadable response
    return FileResponse(
        path=file_path, 
        filename=safe_filename, 
        media_type="application/octet-stream"
    )

if __name__ == "__main__":
    import uvicorn
    # Run the server when executed directly as a script
    # Alternatively, use: uvicorn src.api:app --reload
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

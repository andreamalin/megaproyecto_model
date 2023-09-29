from app_process import *
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

class Resource(BaseModel):
  key: str

CDN_BUCKET = "https://storage.googleapis.com/cdn-deaflens/"
BACKEND_API = "https://deaflens-xa3v62ac2q-uc.a.run.app/"

@app.post("/predict")
async def predictModel(video: Resource):
  try:
    video_path = f'{CDN_BUCKET}{video.key}'
    result = process_video(video_path)
    id = video.key.split('.')[0]
    
    # Construct payload for PUT request
    data = {
      "_id": id,
      "options": result,
      "apiPassword": '7AuY-YmfBv-624MKj2sQ',
      "apiUsername": 'W4B.n-sdsh',
    }
    
    # Send PUT request to BACKEND_API
    async with httpx.AsyncClient() as client:
      response = await client.put(f"{BACKEND_API}/model", json=data)
    
    # You can handle the response as required, for now, I'll just return it
    return response.json()
  except Exception as e:
    # Return error details for easier debugging
    return {"error": str(e)}

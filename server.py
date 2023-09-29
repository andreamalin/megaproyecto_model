from app_process import *
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

class Resource(BaseModel):
  key: str

CDN_BUCKET = "https://storage.googleapis.com/cdn-deaflens/"
BACKEND_API = "https://deaflens-xa3v62ac2q-uc.a.run.app/v1"

@app.post("/predict")
async def predictModel(video: Resource):
  try:
    video_path = f'{CDN_BUCKET}{video.key}'
    id = video.key.split('.')[0]

    result = process_video(video_path)

    # Construct payload for PUT request
    data = {
      "_id": id,
      "options": result[0],
      "apiPassword": '7AuY-YmfBv-624MKj2sQ',
      "apiUsername": 'W4B.n-sdsh',
    }

    # Send PUT request to BACKEND_API
    async with httpx.AsyncClient() as client:
      response = await client.put(f"{BACKEND_API}/resources/model", json=data)
    
    # You can handle the response as required, for now, I'll just return it
    return response.json()
  except Exception as e:
    print(e)
    # Return error details for easier debugging
    return {"error": str(e)}

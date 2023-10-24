from app_process import *
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage
import httpx

app = FastAPI()

class Resource(BaseModel):
  key: str
  langId: str
  isVideo: bool

CDN_BUCKET_NAME = "cdn-deaflens"
CDN_BUCKET = f"https://storage.googleapis.com/{CDN_BUCKET_NAME}/"
BACKEND_API = "https://deaflens-xa3v62ac2q-uc.a.run.app/v1"

@app.post("/predict")
async def predictModel(resource: Resource):
  try:
    video_path = f'{CDN_BUCKET}{resource.langId}/{resource.key}'
    id = resource.key.split('.')[0]

    if (resource.isVideo):
      if (resource.langId == '651b5d410ae23c0573d7953e'):
        result = process_video(video_path, id)
      else:
        result = process_video_asl(video_path, id)
    else:
      storage_client = storage.Client()
      bucket = storage_client.bucket(CDN_BUCKET_NAME)
      blobs = bucket.list_blobs(prefix=f'{resource.langId}/{resource.key}/')
      image_names = [blob.name for blob in blobs]

      result = []
      for index, image_name in enumerate(image_names):
        image_id = f'{id}-{index}'
        result.append(
          process_image(
            f'{CDN_BUCKET}{image_name}',
            image_id,
          )[0]
        )
      print(result)

    # # Construct payload for PUT request
    # data = {
    #   "_id": id,
    #   "options": result[0],
    #   "apiPassword": '7AuY-YmfBv-624MKj2sQ',
    #   "apiUsername": 'W4B.n-sdsh',
    # }

    # # Send PUT request to BACKEND_API
    # async with httpx.AsyncClient() as client:
    #   response = await client.put(f"{BACKEND_API}/resources/model", json=data)
    
    # # You can handle the response as required, for now, I'll just return it
    # return response.json()
  except Exception as e:
    print(e)
    # Return error details for easier debugging
    return {"error": str(e)}

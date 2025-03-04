# backend/main.py (FastAPI)
from fastapi import FastAPI
from routes import router

app = FastAPI(title="Topic Modeling API", version="1.0")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

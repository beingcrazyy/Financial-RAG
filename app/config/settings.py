import os 
from dotenv import load_dotenv

load_dotenv()

Model = os.getenv("MODEL_NAME", "gpt-4o-mini")
Temprature = os.getenv("TEMPRATURE",0)
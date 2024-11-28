from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
RESULTS_PATH = os.getenv("RESULTS_PATH")

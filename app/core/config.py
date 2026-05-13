from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    app_name: str = "ARC Financial Distress Analyzer"
    model_path: str = os.getenv("MODEL_PATH", "ml_models/lightgbm.pkl")
    sec_user_agent: str = "ARC FinancialAnalyzer research@example.com"
    sec_delay: float = 0.25
    sec_retries: int = 4
    sec_backoff: float = 2.0
    cache_dir: str = "data/raw/sec"

    class Config:
        env_file = ".env"


settings = Settings()

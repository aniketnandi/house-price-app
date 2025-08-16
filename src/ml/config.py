from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "house-price-api"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"
    model_path: str = "models/model_v1.joblib"
    metrics_enabled: bool = True
    redis_url: str | None = None
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", protected_namespaces=("settings_")
    )


settings = Settings()

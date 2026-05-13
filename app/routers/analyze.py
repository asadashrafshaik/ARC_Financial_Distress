from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.prediction_service import analyze_ticker
from app.core.model_store import model_store

router = APIRouter()


def _check_model():
    if not model_store.is_loaded:
        raise HTTPException(status_code=503,
            detail="Model not loaded. Run train.py then restart the server.")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_post(request: AnalyzeRequest) -> AnalyzeResponse:
    _check_model()
    try:
        return analyze_ticker(request.ticker.strip().upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.get("/analyze/{ticker}", response_model=AnalyzeResponse)
async def analyze_get(ticker: str) -> AnalyzeResponse:
    _check_model()
    try:
        return analyze_ticker(ticker.strip().upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

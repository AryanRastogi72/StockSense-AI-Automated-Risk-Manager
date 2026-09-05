from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from utils.model_loader import (
    SUPPORTED_TICKERS, ALL_VARIANTS,
    validate_variant, generate_plotly_graph, predict_tree, predict_lstm,
)
from utils.live_features import get_live_features
from utils.risk_utils import compute_risk_for_ticker, compute_portfolio_risk
from utils.explainability import explain_regression, explain_classification

app = FastAPI(title="Stock-Ticker Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Hello FastAPI"}



@app.get("/companies")
def list_companies():
    return {"companies": SUPPORTED_TICKERS}


@app.get("/companies/{ticker}/models")
def list_models(ticker: str):
    ticker = ticker.upper()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not supported. Available: {SUPPORTED_TICKERS}",
        )
    return {"ticker": ticker, "models": ALL_VARIANTS}


@app.get("/companies/{ticker}/models/{algorithm}/{task}/{tuning}/graphs")
def get_graph(ticker: str, algorithm: str, task: str, tuning: str):
    ticker = ticker.upper()
    algorithm = algorithm.lower()
    task = task.lower()
    tuning = tuning.lower()

    valid, error_msg = validate_variant(ticker, algorithm, task, tuning)
    if not valid:
        raise HTTPException(status_code=404, detail=error_msg)

    try:
        live_data = get_live_features(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        if algorithm in ("rf", "xgb"):
            result = predict_tree(ticker, algorithm, task, tuning, live_data["latest_row"])
        else:
            result = predict_lstm(ticker, task, tuning, live_data["sequence"])
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model files not found for {algorithm}/{task}/{tuning}. Run the training script first. ({exc})",
        )

    fig = generate_plotly_graph(ticker, algorithm, task, tuning, live_data, result)
    return HTMLResponse(content=fig.to_html(full_html=True), status_code=200)


@app.get("/companies/{ticker}/models/{algorithm}/{task}/{tuning}/predict")
def predict(ticker: str, algorithm: str, task: str, tuning: str):
    ticker = ticker.upper()
    algorithm = algorithm.lower()
    task = task.lower()
    tuning = tuning.lower()

    valid, error_msg = validate_variant(ticker, algorithm, task, tuning)
    if not valid:
        raise HTTPException(status_code=404, detail=error_msg)

    try:
        live_data = get_live_features(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        if algorithm in ("rf", "xgb"):
            result = predict_tree(ticker, algorithm, task, tuning, live_data["latest_row"])
        else:
            result = predict_lstm(ticker, task, tuning, live_data["sequence"])
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model files not found for {algorithm}/{task}/{tuning}. Run the training script first. ({exc})",
        )

    prediction_type = "regression" if task == "reg" else "classification"
    model_name = f"{algorithm}_{task}_{tuning}"

    response = {
        "ticker": ticker,
        "model": model_name,
        "prediction_type": prediction_type,
        "last_available_date": live_data["last_date"],
        "prediction_for": f"next trading session after {live_data['last_date']}",
    }

    if task == "reg":
        response["predicted_close"] = result["predicted_close"]
    else:
        response["predicted_direction"] = result["predicted_direction"]

    return response

@app.get("/risk/{ticker}")
def get_risk(ticker: str):
    ticker = ticker.upper()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not supported. Available: {SUPPORTED_TICKERS}",
        )

    try:
        live_data = get_live_features(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        risk_result = compute_risk_for_ticker(ticker, live_data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return risk_result


@app.get("/portfolio/risk")
def get_portfolio_risk(tickers: str = "LT"):
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    for t in ticker_list:
        if t not in SUPPORTED_TICKERS:
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{t}' is not supported. Available: {SUPPORTED_TICKERS}",
            )

    live_data_map = {}
    for t in ticker_list:
        try:
            live_data_map[t] = get_live_features(t)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    try:
        portfolio_result = compute_portfolio_risk(ticker_list, live_data_map)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return portfolio_result


@app.get("/explain/{ticker}")
def explain_ticker(ticker: str, task: str, predicted_class: str = None):
    ticker = ticker.upper()
    if ticker not in SUPPORTED_TICKERS:
        raise HTTPException(status_code=404, detail="Ticker not supported")
        
    try:
        live_data = get_live_features(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
        
    if task == "reg":
        return explain_regression(ticker, live_data)
    elif task == "class":
        if not predicted_class:
            raise HTTPException(status_code=400, detail="predicted_class is required for classification SHAP")
        return explain_classification(ticker, live_data, predicted_class)
    else:
        raise HTTPException(status_code=400, detail="task must be 'reg' or 'class'")


if __name__ == "__main__":
    import os
    import sys
    import uvicorn
    from utils.paths import PROJECT_ROOT, MODELS_DIR

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    sys.path.insert(0, backend_dir)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / ".venv"
FRONTEND_DIR = ROOT / "frontend"
MODEL = joblib.load(MODEL_DIR / "best_model.pkl")
FEATURE_NAMES = json.loads((MODEL_DIR / "feature_names.json").read_text(encoding="utf-8"))
CATEGORY_VALUES = json.loads((MODEL_DIR / "category_values.json").read_text(encoding="utf-8"))

FURNISHED_MAP = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
TRANSPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}
NOMINAL_COLUMNS = ["State", "City", "Property_Type", "Facing", "Owner_Type"]
CHOICES = {
    **CATEGORY_VALUES,
    "Furnished_Status": list(FURNISHED_MAP),
    "Public_Transport_Accessibility": list(TRANSPORT_MAP),
    "Parking_Space": ["No", "Yes"],
    "Security": ["No", "Yes"],
    "Availability_Status": ["Ready_to_Move", "Under_Construction"],
}
NUMERIC_FIELDS = (
    "BHK",
    "Size_in_SqFt",
    "Year_Built",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
)
NUMERIC_LIMITS = {
    "BHK": (1, 20),
    "Size_in_SqFt": (100, 20000),
    "Year_Built": (1900, 2100),
    "Floor_No": (0, 200),
    "Total_Floors": (1, 200),
    "Age_of_Property": (0, 150),
    "Nearby_Schools": (0, 100),
    "Nearby_Hospitals": (0, 100),
}


class InvalidInput(ValueError):
    pass


def encode_record(record):
    row = {}
    for field in NUMERIC_FIELDS:
        value = record.get(field)
        if isinstance(value, bool):
            raise InvalidInput(f"{field.replace('_', ' ')} must be a number.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise InvalidInput(f"{field.replace('_', ' ')} must be a number.") from None
        minimum, maximum = NUMERIC_LIMITS[field]
        if not minimum <= number <= maximum:
            raise InvalidInput(
                f"{field.replace('_', ' ')} must be between {minimum} and {maximum}."
            )
        if not number.is_integer():
            raise InvalidInput(f"{field.replace('_', ' ')} must be a whole number.")
        row[field] = number

    if record["Floor_No"] > record["Total_Floors"]:
        raise InvalidInput("Floor number cannot exceed the total number of floors.")

    for field, options in CHOICES.items():
        value = record.get(field)
        if value not in options:
            raise InvalidInput(f"Choose a valid {field.replace('_', ' ').lower()}.")

    row["Furnished_Status"] = FURNISHED_MAP[record["Furnished_Status"]]
    row["Public_Transport_Accessibility"] = TRANSPORT_MAP[
        record["Public_Transport_Accessibility"]
    ]
    row["Parking_Space"] = int(record["Parking_Space"] == "Yes")
    row["Security"] = int(record["Security"] == "Yes")
    row["Availability_Status"] = int(record["Availability_Status"] == "Ready_to_Move")

    for column in NOMINAL_COLUMNS:
        selected = record[column]
        for category in CATEGORY_VALUES[column][1:]:
            row[f"{column}_{category}"] = int(selected == category)

    return pd.DataFrame(
        [[row.get(name, 0) for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
        dtype=float,
    )


def predict(record):
    features = encode_record(record)
    predicted_class = int(MODEL.predict(features)[0])
    probabilities = MODEL.predict_proba(features)[0]
    class_probabilities = {
        str(int(label)): float(probability)
        for label, probability in zip(MODEL.classes_, probabilities)
    }
    return {
        "label": "Above-median price" if predicted_class == 1 else "Below-median price",
        "predicted_class": predicted_class,
        "confidence": class_probabilities[str(predicted_class)],
        "above_median_probability": class_probabilities["1"],
    }


class HousingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/options":
            self._send_json(200, {"choices": CHOICES})
            return

        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        asset = files.get(path)
        if asset is None:
            self.send_error(404, "Not found")
            return

        filename, content_type = asset
        content = (FRONTEND_DIR / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= 16_384:
                raise InvalidInput("Request body must be between 1 byte and 16 KB.")
            body = self.rfile.read(content_length)
            record = json.loads(body.decode("utf-8"))
            if not isinstance(record, dict):
                raise InvalidInput("Request body must be a JSON object.")
            result = predict(record)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must contain valid JSON."})
        except InvalidInput as error:
            self._send_json(400, {"error": str(error)})
        except Exception:
            logging.exception("Prediction request failed")
            self._send_json(500, {"error": "Prediction failed. Check the server log."})
        else:
            self._send_json(200, result)

    def _send_json(self, status, payload):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string, *args):
        logging.info("%s - %s", self.address_string(), format_string % args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), HousingHandler)
    logging.info("Housing insights frontend running at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping housing insights frontend")
    finally:
        server.server_close()

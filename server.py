import threading
import grpc
import numpy as np
from collections import defaultdict
from concurrent import futures
from sklearn.ensemble import IsolationForest

import telemetry_pb2
import telemetry_pb2_grpc

NORMAL_RANGES = {
    "Temperature": (20.0, 30.0),
    "Voltage":     (12.0, 12.5),
    "Radiation":   (0.0,  2.0),
}

WARMUP_SAMPLES = 50
RETRAIN_EVERY  = 20
MAX_BUFFER     = 300

# Drift için score eşiği: model belirsizse (score sıfıra yakın) → drift
DRIFT_SCORE_THRESHOLD = 0.05


def classify_anomaly_type(
    value: float,
    sensor_type: str,
    model_score: float
) -> int:
    """
    Anomali tipini belirler.
    Warmup döneminde (model_score=None) sadece yön bakılır.

    SPIKE → ani yükseliş  (üst sınırı geçti, belirgin)
    DROP  → ani düşüş    (alt sınırı geçti, belirgin)
    DRIFT → yavaş sapma  (score sıfıra yakın → model belirsiz → sınıra yakın)
    NONE  → normal
    """
    low, high   = NORMAL_RANGES.get(sensor_type, (0.0, 100.0))
    center      = (low + high) / 2.0
    half_range  = (high - low) / 2.0

    # Model score küçükse (sınıra yakın ama anomali flaglendi) → drift
    if model_score is not None and abs(model_score) < DRIFT_SCORE_THRESHOLD:
        return telemetry_pb2.DRIFT

    # Değer merkeze göre hangi yönde?
    if value > center:
        return telemetry_pb2.SPIKE
    else:
        return telemetry_pb2.DROP


class AnomalyDetector:
    def __init__(self):
        self._lock    = threading.Lock()
        self._buffers = defaultdict(list)
        self._models  = {}
        self._counts  = defaultdict(int)

    def _needs_retrain(self, sensor_type: str) -> bool:
        buf   = self._buffers[sensor_type]
        count = self._counts[sensor_type]
        if len(buf) < WARMUP_SAMPLES:
            return False
        return (len(buf) == WARMUP_SAMPLES) or (count % RETRAIN_EVERY == 0)

    def _train(self, sensor_type: str):
        buf = self._buffers[sensor_type]
        X   = np.array(buf).reshape(-1, 1)
        model = IsolationForest(
            n_estimators=100,
            contamination=0.10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        self._models[sensor_type] = model
        print(f"[ML] {sensor_type} modeli eğitildi ({len(buf)} örnek)")

    def predict(
        self, value: float, sensor_type: str
    ) -> tuple[bool, float, int]:
        with self._lock:
            buf = self._buffers[sensor_type]
            buf.append(value)
            if len(buf) > MAX_BUFFER:
                buf.pop(0)
            self._counts[sensor_type] += 1

            if self._needs_retrain(sensor_type):
                self._train(sensor_type)

            model = self._models.get(sensor_type)

        # Warmup fallback
        if model is None:
            low, high  = NORMAL_RANGES.get(sensor_type, (0.0, 100.0))
            is_anomaly = value < low or value > high
            anomaly_type = (
                classify_anomaly_type(value, sensor_type, None)
                if is_anomaly else telemetry_pb2.NONE
            )
            remaining = max(0, WARMUP_SAMPLES - len(self._buffers[sensor_type]))
            print(
                f"[WARMUP] {sensor_type}: {value:.4f} "
                f"| anomaly={is_anomaly} type={anomaly_type} "
                f"| {remaining} örnek kaldı"
            )
            return is_anomaly, 0.60, anomaly_type

        # Model hazır
        X          = np.array([[value]])
        prediction = model.predict(X)[0]
        score      = float(model.decision_function(X)[0])
        is_anomaly = (prediction == -1)

        confidence = float(np.clip(0.55 + abs(score) * 1.5, 0.55, 0.97))

        anomaly_type = (
            classify_anomaly_type(value, sensor_type, score)
            if is_anomaly else telemetry_pb2.NONE
        )

        type_names = {0: "NONE", 1: "SPIKE", 2: "DROP", 3: "DRIFT"}
        print(
            f"[ML] {sensor_type}: {value:.4f} "
            f"| score={score:.3f} "
            f"| anomaly={is_anomaly} "
            f"| type={type_names[anomaly_type]} "
            f"| conf={confidence:.2f}"
        )
        return is_anomaly, confidence, anomaly_type


detector = AnomalyDetector()


def process_cosmic_data(
    raw_value: float, sensor_type: str
) -> tuple[float, bool, float, int]:
    cleaned = raw_value * 0.985
    is_anomaly, confidence, anomaly_type = detector.predict(raw_value, sensor_type)
    return cleaned, is_anomaly, confidence, anomaly_type


class TelemetryProcessor(telemetry_pb2_grpc.TelemetryProcessorServicer):

    def ProcessStream(self, request_iterator, context):
        print("gRPC stream connected.")
        type_names = {0: "NONE", 1: "SPIKE", 2: "DROP", 3: "DRIFT"}

        for request in request_iterator:
            cleaned, is_anomaly, confidence, anomaly_type = process_cosmic_data(
                request.raw_value, request.sensor_type
            )
            sat_id = request.satellite_id or "UNKNOWN-SAT"

            if is_anomaly:
                msg = (
                    f"CRITICAL [{type_names[anomaly_type]}]: "
                    f"{request.sensor_type} anomaly on {sat_id}!"
                )
            else:
                msg = f"OK: {sat_id} {request.sensor_type} nominal."

            yield telemetry_pb2.TelemetryResponse(
                timestamp     = request.timestamp,
                cleaned_value = cleaned,
                is_anomaly    = is_anomaly,
                confidence    = confidence,
                message       = msg,
                anomaly_type  = anomaly_type,
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    telemetry_pb2_grpc.add_TelemetryProcessorServicer_to_server(
        TelemetryProcessor(), server
    )
    server.add_insecure_port("[::]:50051")
    print("--------------------------------------------------")
    print("Python ML Sunucusu 50051 portunda")
    print(f"Warmup: {WARMUP_SAMPLES} | Retrain: her {RETRAIN_EVERY} pakette")
    print("--------------------------------------------------")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
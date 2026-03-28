import threading
import grpc
import numpy as np
from collections import defaultdict
from concurrent import futures
from sklearn.ensemble import IsolationForest

import telemetry_pb2
import telemetry_pb2_grpc

# Threshold fallback — warmup döneminde kullanılır
NORMAL_RANGES = {
    "Temperature": (20.0, 30.0),
    "Voltage":     (12.0, 12.5),
    "Radiation":   (0.0,  2.0),
}

WARMUP_SAMPLES = 50   # model eğitimi için minimum örnek
RETRAIN_EVERY  = 20   # her N yeni örnekte retrain
MAX_BUFFER     = 300  # sensör başına maksimum örnek


class AnomalyDetector:
    """
    Her sensör tipi için ayrı IsolationForest modeli tutar.
    Warmup tamamlanana kadar threshold tabanlı fallback kullanır.
    Thread-safe: birden fazla gRPC isteği eşzamanlı çağırabilir.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._buffers = defaultdict(list)   # sensor_type → [float]
        self._models  = {}                  # sensor_type → IsolationForest
        self._counts  = defaultdict(int)    # sensor_type → toplam işlenen paket

    def _needs_retrain(self, sensor_type: str) -> bool:
        buf   = self._buffers[sensor_type]
        count = self._counts[sensor_type]
        if len(buf) < WARMUP_SAMPLES:
            return False
        # İlk warmup tamamlandığında ya da her RETRAIN_EVERY pakette
        return (len(buf) == WARMUP_SAMPLES) or (count % RETRAIN_EVERY == 0)

    def _train(self, sensor_type: str):
        buf = self._buffers[sensor_type]
        X   = np.array(buf).reshape(-1, 1)
        model = IsolationForest(
            n_estimators=100,
            contamination=0.10,  # veri setinin ~%10'unun anomali olduğunu varsay
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        self._models[sensor_type] = model
        print(
            f"[ML] {sensor_type} modeli eğitildi "
            f"({len(buf)} örnek, contamination=0.10)"
        )

    def predict(self, value: float, sensor_type: str) -> tuple[bool, float]:
        with self._lock:
            # Buffer güncelle
            buf = self._buffers[sensor_type]
            buf.append(value)
            if len(buf) > MAX_BUFFER:
                buf.pop(0)

            self._counts[sensor_type] += 1

            # Gerekirse retrain
            if self._needs_retrain(sensor_type):
                self._train(sensor_type)

            model = self._models.get(sensor_type)

        # --- Warmup dönemi: threshold fallback ---
        if model is None:
            remaining = WARMUP_SAMPLES - len(self._buffers[sensor_type])
            low, high = NORMAL_RANGES.get(sensor_type, (0.0, 100.0))
            is_anomaly = value < low or value > high
            print(
                f"[WARMUP] {sensor_type}: {value:.4f} "
                f"| anomaly={is_anomaly} "
                f"| {remaining} örnek kaldı"
            )
            return is_anomaly, 0.60

        # --- Model hazır: IsolationForest ---
        X          = np.array([[value]])
        prediction = model.predict(X)[0]          # 1=normal, -1=anomaly
        score      = model.decision_function(X)[0] # pozitif=normal, negatif=anomali

        is_anomaly = (prediction == -1)

        # Confidence: score'un mutlak değerini [0.55, 0.97] aralığına normalize et
        # Skor 0'a yakın → belirsiz (~0.55), uzaklaştıkça → daha emin (~0.97)
        confidence = float(np.clip(0.55 + abs(score) * 1.5, 0.55, 0.97))

        print(
            f"[ML] {sensor_type}: {value:.4f} "
            f"| score={score:.3f} "
            f"| anomaly={is_anomaly} "
            f"| conf={confidence:.2f}"
        )
        return is_anomaly, confidence


# Global detector — tüm stream'ler paylaşır
detector = AnomalyDetector()


def process_cosmic_data(
    raw_value: float, sensor_type: str
) -> tuple[float, bool, float]:
    noise_filter = 0.985
    cleaned      = raw_value * noise_filter

    is_anomaly, confidence = detector.predict(raw_value, sensor_type)
    return cleaned, is_anomaly, confidence


class TelemetryProcessor(telemetry_pb2_grpc.TelemetryProcessorServicer):

    def ProcessStream(self, request_iterator, context):
        print("gRPC stream connected.")
        for request in request_iterator:
            cleaned, is_anomaly, confidence = process_cosmic_data(
                request.raw_value, request.sensor_type
            )
            sat_id = request.satellite_id or "UNKNOWN-SAT"
            msg = (
                f"CRITICAL: {request.sensor_type} anomaly on {sat_id}!"
                if is_anomaly else
                f"OK: {sat_id} {request.sensor_type} nominal."
            )
            yield telemetry_pb2.TelemetryResponse(
                timestamp     = request.timestamp,
                cleaned_value = cleaned,
                is_anomaly    = is_anomaly,
                confidence    = confidence,
                message       = msg,
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    telemetry_pb2_grpc.add_TelemetryProcessorServicer_to_server(
        TelemetryProcessor(), server
    )
    server.add_insecure_port("[::]:50051")
    print("--------------------------------------------------")
    print("Python ML Sunucusu 50051 portunda")
    print(f"Warmup: {WARMUP_SAMPLES} örnek/sensör")
    print(f"Retrain: her {RETRAIN_EVERY} pakette")
    print("--------------------------------------------------")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

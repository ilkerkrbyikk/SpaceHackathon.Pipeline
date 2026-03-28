import grpc
from concurrent import futures
import telemetry_pb2
import telemetry_pb2_grpc

# Hafif bir gürültü temizliği ve sensöre göre temel anomali kontrolü uygular.
def process_cosmic_data(raw_value, sensor_type):
    # Basit gürültü filtreleme simülasyonu.
    noise_filter = 0.985 
    cleaned = raw_value * noise_filter
    
    # Anomali kontrolünde sensör tipine göre eşik kullanılır.
    threshold = 5.0 if sensor_type == "Temperature" else 2.0
    is_anomaly = abs(raw_value - cleaned) > threshold
    
    confidence = 0.94 if not is_anomaly else 0.72
    
    return cleaned, is_anomaly, confidence

class TelemetryProcessor(telemetry_pb2_grpc.TelemetryProcessorServicer):
    
    def ProcessStream(self, request_iterator, context):
        print("gRPC stream connected, telemetry flow started.")
        
        # Çift yönlü akıştaki istekleri sırayla işler.
        for request in request_iterator:
            # Gelen telemetri verisini işler.
            cleaned, is_anomaly, conf = process_cosmic_data(request.raw_value, request.sensor_type)
            
            # Uydu kimliği boşsa varsayılan bir değer kullanılır.
            sat_id = request.satellite_id if request.satellite_id else "UNKNOWN-SAT"
            
            if is_anomaly:
                msg = f"CRITICAL: {request.sensor_type} anomaly detected on {sat_id}!"
            else:
                msg = f"Status: {sat_id} signal is stable and cleaned."

            # Alan adları telemetry.proto ile birebir uyumlu olmalıdır.
            yield telemetry_pb2.TelemetryResponse(
                timestamp=request.timestamp,
                cleaned_value=cleaned,
                is_anomaly=is_anomaly,
                confidence=conf,
                message=msg
            )

def serve():
    # Stream işleme için küçük bir thread havuzu ile gRPC sunucusu başlatır.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    telemetry_pb2_grpc.add_TelemetryProcessorServicer_to_server(TelemetryProcessor(), server)
    
    # .NET istemci ayarıyla aynı port kullanılmalıdır.
    server.add_insecure_port('[::]:50051')
    print("--------------------------------------------------")
    print("Python Sunucusu 50051 portunda")
    print("--------------------------------------------------")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
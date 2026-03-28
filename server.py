import grpc
from concurrent import futures
import telemetry_pb2
import telemetry_pb2_grpc
import time

# MÜNİR NOTU: Algoritma, uydunun kalbidir. 
# Gürültüyü siler, anomaliyi (radyasyon çarpması) teşhis eder.
def process_cosmic_data(raw_value, sensor_type):
    # Basit bir filtreleme simülasyonu
    noise_filter = 0.985 
    cleaned = raw_value * noise_filter
    
    # Anomali tespiti (Eğer veri normalden çok sapıyorsa)
    # Farklı sensörler için farklı toleranslar koyabilirsin
    threshold = 5.0 if sensor_type == "Temperature" else 2.0
    is_anomaly = abs(raw_value - cleaned) > threshold
    
    confidence = 0.94 if not is_anomaly else 0.72
    
    return cleaned, is_anomaly, confidence

class TelemetryProcessor(telemetry_pb2_grpc.TelemetryProcessorServicer):
    
    def ProcessStream(self, request_iterator, context):
        print("Münir Üstat: .NET hattı bağlandı, uzaydan sinyal akıyor...")
        
        # Çift yönlü akış (Bi-directional streaming)
        for request in request_iterator:
            # Gelen veriyi (Request) işliyoruz
            cleaned, is_anomaly, conf = process_cosmic_data(request.raw_value, request.sensor_type)
            
            # 'msg' değişkenini burada oluşturuyoruz ki Response içine koyabilelim.
            sat_id = request.satellite_id if request.satellite_id else "UNKNOWN-SAT"
            
            if is_anomaly:
                msg = f"CRITICAL: {request.sensor_type} anomaly detected on {sat_id}!"
            else:
                msg = f"Status: {sat_id} signal is stable and cleaned."

            # Sonucu (Response) geri fırlatıyoruz
            # MÜNİR NOTU: Proto'daki isimlerle Python'daki isimler aynı olmalı!
            yield telemetry_pb2.TelemetryResponse(
                timestamp=request.timestamp,
                cleaned_value=cleaned,
                is_anomaly=is_anomaly,
                confidence=conf,
                message=msg  # Artık 'msg' tanımlı, hata vermez!
            )

def serve():
    # 10 koldan (thread) veri işleyebilecek güçte bir sunucu
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    telemetry_pb2_grpc.add_TelemetryProcessorServicer_to_server(TelemetryProcessor(), server)
    
    # .NET tarafında 50051 demiştik, sakın portu şaşırma!
    server.add_insecure_port('[::]:50051')
    print("--------------------------------------------------")
    print("Python Sunucusu 50051 portunda")
    print("--------------------------------------------------")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
import time
import requests
import random

API_URL = "http://localhost:8000/detect"  # FastAPI server URL

def generate_gps_data():
    """Simulate normal GPS data with occasional extreme spoofed jumps."""
    
    spoof_chance = random.random()
    
    if spoof_chance > 0.5:  # 50% chance of extreme spoofing
        gps_data = {
            "GPS_lat": 37.7749 + random.uniform(-100, 100),  # Large jumps
            "GPS_long": -122.4194 + random.uniform(-100, 100),
            "vx": random.uniform(-100, 100),  # Unrealistic high speed
            "vy": random.uniform(-100, 100),
            "ax": random.uniform(-10, 10),  # Unrealistic acceleration
            "ay": random.uniform(-10, 10)
        }
        spoofed = True
    else:  # Normal movement (small variations)
        gps_data = {
            "GPS_lat": 37.7749 + random.uniform(-0.001, 0.001),
            "GPS_long": -122.4194 + random.uniform(-0.001, 0.001),
            "vx": random.uniform(-3, 3),
            "vy": random.uniform(-3, 3),
            "ax": random.uniform(-1, 1),
            "ay": random.uniform(-1, 1)
        }
        spoofed = False

    return gps_data, spoofed

def send_gps_data():
    while True:
        gps_data, is_spoofed = generate_gps_data()

        try:
            response = requests.post(API_URL, json=gps_data)
            
            if response.status_code == 200:
                result_data = response.json()
                print(f"{'⚠️ Spoofed' if is_spoofed else '✅ Real'} - GPS: {gps_data} -> Result: {result_data['result']}, Confidence: {result_data['confidence']:.2f}")
            else:
                print(f"⚠️ Error: Received status code {response.status_code} from the API")
        
        except requests.ConnectionError:
            print("❌ Unable to connect to the server. Is it running?")
        except requests.exceptions.Timeout:
            print("❌ Request timed out.")
        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred: {e}")

        # Random delay between requests (0.5 to 2 seconds)
        time.sleep(random.uniform(0.5, 2))

if __name__ == "__main__":
    send_gps_data()

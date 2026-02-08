import cv2
import torch
import numpy as np
import time
import warnings
import serial
import json
from datetime import datetime
import serial.tools.list_ports

warnings.filterwarnings('ignore')

class STM32NucleoCommunication:
    def __init__(self, port='COM7', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False
        self.connect_serial()
    
    def connect_serial(self):
        """Initialize serial connection to STM32 Nucleo"""
        try:
            print(f"🔌 Connecting to {self.port}...")
            
            # Try to connect to the specified port
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=2,
                    write_timeout=2
                )
            except Exception as e:
                if "PermissionError" in str(e) or "Access is denied" in str(e):
                    print(f"❌ {self.port} is busy! Please close other programs using this port.")
                    self.connected = False
                    return
                else:
                    raise e
            
            # Wait for connection to stabilize
            print("⏳ Waiting for STM32 connection...")
            time.sleep(3)
            
            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Read STM32 startup messages
            print("📥 Reading STM32 messages...")
            time.sleep(2)
            
            startup_messages = []
            while self.serial_conn.in_waiting:
                try:
                    message = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if message:
                        startup_messages.append(message)
                        print(f"   {message}")
                except:
                    break
            
            if startup_messages:
                self.connected = True
                print(f"✅ Successfully connected to STM32 on {self.port}")
            else:
                # Try to trigger a response
                print("🔄 Testing communication...")
                test_command = '{"people_count": 0, "spray_duration": 0, "threshold": 3}\n'
                self.serial_conn.write(test_command.encode())
                time.sleep(1)
                
                if self.serial_conn.in_waiting:
                    response = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if response:
                        print(f"📥 Response: {response}")
                        self.connected = True
                        print(f"✅ Connected to STM32 on {self.port}")
                else:
                    print("⚠️  Connected but no response from STM32")
                    self.connected = True  # Still try to use it
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.connected = False
            self.serial_conn = None
    
    def send_disinfection_command(self, people_count, threshold=3):
        """Send disinfection command to STM32 Nucleo"""
        # Determine spray duration based on people count
        if people_count == 0:
            spray_duration = 0  # No spraying if no people
        elif people_count <= threshold:
            spray_duration = 20  # 20 seconds for low crowd
        else:
            spray_duration = 30  # 30 seconds for high crowd
        
        # Create command packet
        command = {
            'people_count': people_count,
            'spray_duration': spray_duration,
            'threshold': threshold,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        # Convert to JSON string
        command_str = json.dumps(command) + '\n'
        
        try:
            if self.connected and self.serial_conn and self.serial_conn.is_open:
                # Debug: print the JSON being sent
                print(f"🔧 DEBUG JSON: {command_str.strip()}")
                
                self.serial_conn.write(command_str.encode())
                print(f"📤 Sent to STM32: People={people_count}, Spray={spray_duration}s")
                
                # Increased delay for better serial handling
                time.sleep(0.5)  # Give STM32 time to process
                
                # Read all available responses
                start_time = time.time()
                responses = []
                while time.time() - start_time < 2:  # Read for up to 2 seconds
                    if self.serial_conn.in_waiting:
                        ack = self.serial_conn.readline().decode().strip()
                        if ack:
                            responses.append(ack)
                            print(f"📥 STM32: {ack}")
                    time.sleep(0.1)
                
                # If no responses, print warning
                if not responses:
                    print("⚠️  No response from STM32")
                    
            else:
                print(f"💻 SIMULATION: People={people_count}, Would spray={spray_duration}s")
                
        except Exception as e:
            print(f"❌ Serial write error: {e}")
            self.connected = False

class YOLOPeopleCounter:
    def __init__(self, stm32_comm):
        print("🚀 Initializing YOLO People Counter with STM32 Nucleo...")
        self.stm32_comm = stm32_comm
        self.model = self.load_yolo_model()
        self.people_count = 0
        self.threshold = 3
        self.paused = False  # Add paused as instance variable
        print("✅ System initialized successfully!")
        print(f"🎯 Using device: {self.get_device()}")
        
    def load_yolo_model(self):
        """Load YOLO model with proper error handling"""
        try:
            # Try loading YOLOv8 (recommended)
            from ultralytics import YOLO
            print("📥 Loading YOLOv8 model...")
            model = YOLO('yolov8n.pt')  # Nano version for speed
            print("✅ YOLOv8 loaded successfully!")
            return model
        except Exception as e:
            print(f"⚠️  YOLOv8 failed: {e}")
            try:
                # Fallback to YOLOv5
                print("📥 Loading YOLOv5 model...")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                print("✅ YOLOv5 loaded successfully!")
                return model
            except Exception as e2:
                print(f"❌ YOLOv5 failed: {e2}")
                print("🚨 Using fallback detection...")
                return None
    
    def get_device(self):
        """Check if GPU is available"""
        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name()})"
        else:
            return "CPU"
    
    def detect_people(self, frame):
        """Detect people in frame using YOLO"""
        people_count = 0
        processed_frame = frame.copy()
        
        if self.model is None:
            return self.fallback_detection(frame)
        
        try:
            # YOLOv8
            if hasattr(self.model, 'predict'):
                results = self.model(frame, verbose=False)
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        # Class 0 is 'person' in COCO dataset
                        if cls == 0 and conf > 0.5:  # Confidence threshold 50%
                            people_count += 1
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(processed_frame, f'Person {conf:.2f}', 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # YOLOv5
            else:
                results = self.model(frame)
                for detection in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = detection
                    if int(cls) == 0 and conf > 0.5:  # Person class
                        people_count += 1
                        # Draw bounding box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f'Person {conf:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        except Exception as e:
            print(f"⚠️  YOLO detection error: {e}")
            return self.fallback_detection(frame)
        
        return people_count, processed_frame
    
    def fallback_detection(self, frame):
        """Fallback detection using Haar cascades"""
        people_count = 0
        processed_frame = frame.copy()
        
        try:
            # Load pre-trained Haar cascade for full body
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect people
            people = cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
            people_count = len(people)
            
            # Draw bounding boxes
            for (x, y, w, h) in people:
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(processed_frame, 'Person (Fallback)', 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                          
        except Exception as e:
            print(f"⚠️  Fallback detection error: {e}")
        
        return people_count, processed_frame
    
    def run_detection_system(self):
        """Main function to run the detection system"""
        print("\n📷 Starting webcam detection system...")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎯 Intelligent Disinfection System Active!")
        print("⏹️  Press 'q' to quit")
        print("⏸️  Press 'p' to pause/resume")
        print("🔧 Press 't' to change threshold")
        print("🔧 Press 'r' to reset spray cooldown")
        print("-" * 50)
        
        # System variables
        total_frames = 0
        fps = 0
        last_spray_time = 0
        spray_cooldown = 15  # Minimum seconds between sprays
        start_time = time.time()
        
        while True:
            if not self.paused:
                frame_start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame!")
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Detect people
                self.people_count, processed_frame = self.detect_people(frame)
                
                # Calculate performance metrics
                detection_time = time.time() - frame_start_time
                total_frames += 1
                fps = 1.0 / detection_time if detection_time > 0 else 0
                
                # Send command to STM32 with cooldown
                current_time = time.time()
                if current_time - last_spray_time > spray_cooldown:
                    self.stm32_comm.send_disinfection_command(self.people_count, self.threshold)
                    last_spray_time = current_time
                
                # Display information on frame
                self.display_info(processed_frame, fps, detection_time)
                
                # Show the frame
                cv2.imshow('Intelligent Disinfection System - YOLO + STM32', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RESUMED"
                print(f"⏸️  System {status}")
            elif key == ord('t'):
                self.change_threshold()
            elif key == ord('r'):
                last_spray_time = 0
                print("🔧 Spray cooldown reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        total_time = time.time() - start_time
        print("\n" + "=" * 50)
        print("📊 SESSION SUMMARY")
        print("=" * 50)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Frames processed: {total_frames}")
        print(f"Average FPS: {total_frames/total_time:.1f}")
        print(f"Final people count: {self.people_count}")
        print("🛑 System stopped successfully")
        print("=" * 50)
    
    def display_info(self, frame, fps, detection_time):
        """Display system information on the frame"""
        # People count with color coding
        if self.people_count == 0:
            color = (0, 255, 0)  # Green - no people
        elif self.people_count <= self.threshold:
            color = (0, 165, 255)  # Orange - low crowd
        else:
            color = (0, 0, 255)  # Red - high crowd
        
        cv2.putText(frame, f'People Count: {self.people_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # System info
        info_lines = [
            f'Threshold: {self.threshold}',
            f'FPS: {fps:.1f}',
            f'Detection: {detection_time*1000:.1f}ms',
            f'STM32: {"Connected" if self.stm32_comm.connected else "Simulation"}'
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 60 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status indicator
        status_color = (0, 255, 0) if not self.paused else (0, 0, 255)
        status_text = "LIVE" if not self.paused else "PAUSED"
        cv2.putText(frame, f'Status: {status_text}', 
                   (10, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Instructions
        instructions = [
            "q: Quit",
            "p: Pause/Resume", 
            "t: Change Threshold",
            "r: Reset Cooldown"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 80 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Footer
        cv2.putText(frame, "Intelligent Disinfection System - YOLO + STM32", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def change_threshold(self):
        """Change the crowd threshold"""
        try:
            print(f"\n🔧 Current threshold: {self.threshold}")
            new_threshold = int(input("Enter new threshold (1-10): "))
            if 1 <= new_threshold <= 10:
                self.threshold = new_threshold
                print(f"✅ Threshold updated to: {self.threshold}")
            else:
                print("❌ Threshold must be between 1-10")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def main():
    print("=" * 70)
    print("           INTELLIGENT DISINFECTION CONTROL SYSTEM")
    print("=" * 70)
    print("🔍 Real-time People Detection using YOLO")
    print("💻 STM32 Nucleo L476RG Microcontroller")
    print("💧 Automatic Disinfectant Spray Control")
    print("🎯 Adaptive Spray Duration based on Crowd Density")
    print("=" * 70)
    
    # Initialize STM32 communication - Change COM7 to your actual port if different
    stm32_comm = STM32NucleoCommunication(port='COM7')
    
    # Create people counter system
    counter = YOLOPeopleCounter(stm32_comm)
    
    # Start detection system
    counter.run_detection_system()

if __name__ == "__main__":
    main()
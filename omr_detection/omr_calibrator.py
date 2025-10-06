"""
OMR Calibration Tool for 100% Accuracy
Interactive calibration system to manually fine-tune bubble detection
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class OMRCalibrator:
    def __init__(self):
        self.image = None
        self.processed_image = None
        self.calibration_points = []
        self.current_question = 1
        self.total_questions = 30
        
        # Calibration parameters
        self.bubble_radius = 12
        self.threshold = 0.65
        
        # Initialize GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the calibration GUI"""
        self.root = tk.Tk()
        self.root.title("OMR Calibration Tool")
        self.root.geometry("1200x800")
        
        # Create frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control widgets
        ttk.Label(self.control_frame, text="OMR Calibration Tool", font=('Arial', 14, 'bold')).pack(pady=10)
        
        ttk.Button(self.control_frame, text="Load Image", command=self.load_image).pack(pady=5)
        
        ttk.Label(self.control_frame, text="Bubble Radius:").pack(pady=(20,5))
        self.radius_var = tk.IntVar(value=self.bubble_radius)
        ttk.Scale(self.control_frame, from_=8, to=20, variable=self.radius_var, 
                 orient=tk.HORIZONTAL, command=self.update_radius).pack(pady=5)
        
        ttk.Label(self.control_frame, text="Detection Threshold:").pack(pady=(10,5))
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        ttk.Scale(self.control_frame, from_=0.3, to=0.9, variable=self.threshold_var,
                 orient=tk.HORIZONTAL, command=self.update_threshold).pack(pady=5)
        
        ttk.Label(self.control_frame, text="Current Question:").pack(pady=(20,5))
        self.question_var = tk.IntVar(value=1)
        ttk.Scale(self.control_frame, from_=1, to=30, variable=self.question_var,
                 orient=tk.HORIZONTAL, command=self.update_question).pack(pady=5)
        
        # Calibration buttons
        ttk.Button(self.control_frame, text="Auto Calibrate", 
                  command=self.auto_calibrate).pack(pady=(20,5))
        ttk.Button(self.control_frame, text="Manual Calibrate", 
                  command=self.manual_calibrate).pack(pady=5)
        ttk.Button(self.control_frame, text="Test Detection", 
                  command=self.test_detection).pack(pady=5)
        ttk.Button(self.control_frame, text="Save Calibration", 
                  command=self.save_calibration).pack(pady=5)
        ttk.Button(self.control_frame, text="Load Calibration", 
                  command=self.load_calibration).pack(pady=5)
        
        # Results display
        self.results_text = tk.Text(self.control_frame, height=15, width=40)
        self.results_text.pack(pady=(20,10), fill=tk.BOTH, expand=True)
        
        # Image display
        self.canvas = tk.Canvas(self.image_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
    
    def load_image(self):
        """Load an OMR image for calibration"""
        file_path = filedialog.askopenfilename(
            title="Select OMR Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.processed_image = self.preprocess_image(self.image)
            self.display_image()
    
    def preprocess_image(self, image):
        """Preprocess image for better visualization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        return enhanced
    
    def display_image(self):
        """Display the current image on canvas"""
        if self.processed_image is not None:
            # Resize image to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                image_height, image_width = self.processed_image.shape
                
                # Calculate scaling factor
                scale_x = canvas_width / image_width
                scale_y = canvas_height / image_height
                scale = min(scale_x, scale_y, 1.0)  # Don't upscale
                
                new_width = int(image_width * scale)
                new_height = int(image_height * scale)
                
                resized = cv2.resize(self.processed_image, (new_width, new_height))
                
                # Convert to PIL Image and display
                pil_image = Image.fromarray(resized)
                self.photo = ImageTk.PhotoImage(pil_image)
                
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                       image=self.photo, anchor=tk.CENTER)
                
                # Store scaling factor for coordinate conversion
                self.scale_factor = scale
                self.offset_x = (canvas_width - new_width) // 2
                self.offset_y = (canvas_height - new_height) // 2
    
    def on_canvas_click(self, event):
        """Handle canvas clicks for manual calibration"""
        if self.processed_image is not None:
            # Convert canvas coordinates to image coordinates
            canvas_x = event.x - self.offset_x
            canvas_y = event.y - self.offset_y
            
            image_x = int(canvas_x / self.scale_factor)
            image_y = int(canvas_y / self.scale_factor)
            
            self.log_message(f"Clicked at image coordinates: ({image_x}, {image_y})")
            
            # Analyze bubble at clicked position
            self.analyze_bubble_at_position(image_x, image_y)
    
    def analyze_bubble_at_position(self, x, y):
        """Analyze bubble characteristics at given position"""
        if self.processed_image is None:
            return
        
        radius = self.radius_var.get()
        
        # Create circular mask
        mask = np.zeros(self.processed_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Calculate statistics
        total_pixels = np.sum(mask > 0)
        bubble_region = cv2.bitwise_and(self.processed_image, self.processed_image, mask=mask)
        dark_pixels = np.sum((bubble_region < 100) & (mask > 0))
        fill_percentage = dark_pixels / total_pixels if total_pixels > 0 else 0
        
        # Calculate average intensity
        masked_pixels = self.processed_image[mask > 0]
        avg_intensity = np.mean(masked_pixels) if len(masked_pixels) > 0 else 255
        
        is_filled = fill_percentage > self.threshold_var.get()
        
        result = f"Position: ({x}, {y})\\n"
        result += f"Fill %: {fill_percentage:.3f}\\n"
        result += f"Avg Intensity: {avg_intensity:.1f}\\n"
        result += f"Status: {'FILLED' if is_filled else 'EMPTY'}\\n"
        result += f"Threshold: {self.threshold_var.get():.2f}\\n"
        result += "-" * 30 + "\\n"
        
        self.log_message(result)
    
    def auto_calibrate(self):
        """Automatically calibrate bubble positions"""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        self.log_message("Starting automatic calibration...")
        
        # Use the precise OMR detector for auto calibration
        from precise_omr_detector import PreciseOMRDetector
        detector = PreciseOMRDetector()
        detector.debug = False  # Disable debug output during calibration
        
        try:
            calibration = detector.manual_calibration(self.image_path)
            self.calibration_data = calibration
            
            self.log_message("Automatic calibration completed!")
            self.log_message(f"Grid bounds: {calibration['grid_bounds']}")
            self.log_message(f"Detected {len(calibration['bubble_positions'])} questions")
            
        except Exception as e:
            messagebox.showerror("Error", f"Auto calibration failed: {str(e)}")
    
    def manual_calibrate(self):
        """Start manual calibration mode"""
        self.log_message("Manual calibration mode activated.")
        self.log_message("Click on bubbles to analyze their detection accuracy.")
        self.log_message("Adjust radius and threshold as needed.")
    
    def test_detection(self):
        """Test the current calibration settings"""
        if not hasattr(self, 'calibration_data'):
            messagebox.showwarning("Warning", "Please perform calibration first!")
            return
        
        self.log_message("Testing detection with current settings...")
        
        # Update detection parameters
        from precise_omr_detector import PreciseOMRDetector
        detector = PreciseOMRDetector()
        detector.bubble_threshold = self.threshold_var.get()
        detector.debug = False
        
        try:
            results = detector.extract_answers(self.image_path)
            
            # Display results
            self.log_message("\\nDetection Results:")
            answers = results['answers']
            
            for q in range(1, 31):
                answer = answers.get(q, 'BLANK')
                self.log_message(f"Q{q:2d}: {answer}")
            
            # Count answer types
            filled = sum(1 for a in answers.values() if a not in ['BLANK', ''] and '*' not in str(a))
            blank = sum(1 for a in answers.values() if a == 'BLANK')
            multiple = sum(1 for a in answers.values() if '*' in str(a))
            
            self.log_message(f"\\nSummary:")
            self.log_message(f"Filled: {filled}")
            self.log_message(f"Blank: {blank}")
            self.log_message(f"Multiple: {multiple}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection test failed: {str(e)}")
    
    def save_calibration(self):
        """Save calibration settings to file"""
        if not hasattr(self, 'calibration_data'):
            messagebox.showwarning("Warning", "No calibration data to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            calibration_settings = {
                'bubble_radius': self.radius_var.get(),
                'threshold': self.threshold_var.get(),
                'calibration_data': self.calibration_data
            }
            
            with open(file_path, 'w') as f:
                json.dump(calibration_settings, f, indent=2, default=str)
            
            messagebox.showinfo("Success", f"Calibration saved to {file_path}")
    
    def load_calibration(self):
        """Load calibration settings from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    calibration_settings = json.load(f)
                
                self.radius_var.set(calibration_settings.get('bubble_radius', 12))
                self.threshold_var.set(calibration_settings.get('threshold', 0.65))
                self.calibration_data = calibration_settings.get('calibration_data', {})
                
                messagebox.showinfo("Success", f"Calibration loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load calibration: {str(e)}")
    
    def update_radius(self, value):
        """Update bubble radius"""
        self.bubble_radius = self.radius_var.get()
        self.log_message(f"Bubble radius updated to: {self.bubble_radius}")
    
    def update_threshold(self, value):
        """Update detection threshold"""
        self.threshold = self.threshold_var.get()
        self.log_message(f"Detection threshold updated to: {self.threshold:.2f}")
    
    def update_question(self, value):
        """Update current question"""
        self.current_question = self.question_var.get()
        self.log_message(f"Current question: {self.current_question}")
    
    def log_message(self, message):
        """Log message to results text area"""
        self.results_text.insert(tk.END, message + "\\n")
        self.results_text.see(tk.END)
    
    def run(self):
        """Run the calibration tool"""
        self.root.mainloop()

if __name__ == "__main__":
    calibrator = OMRCalibrator()
    calibrator.run()
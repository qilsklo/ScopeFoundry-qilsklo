import time
import numpy as np
import cv2
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ScopeFoundry import Measurement

class FlakeFinder(Measurement):
    
    name = "flake_finder"
    
    def setup(self):
        # Define logged quantities
        self.settings.New("search_width", dtype=float, initial=5000.0, unit="um")
        self.settings.New("search_height", dtype=float, initial=5000.0, unit="um")
        self.settings.New("step_size", dtype=float, initial=100.0, unit="um")
        self.settings.New("contrast_threshold", dtype=float, initial=150.0)
        self.settings.New("calibration_constant", dtype=float, initial=1.0, unit="um/px")
        
        # Mode selection
        self.settings.New("search_mode", dtype=str, initial="Blob", choices=("Blob", "Template"))
        self.settings.New("template_path", dtype="file", initial="")
        
        # Shared data between run() and update_display()
        self.current_image = None
        self.current_processed_image = None
        self.detected_targets = []  # List of dicts with 'x', 'y' (pixels)
        self.found_flakes_stage_coords = [] # List of tuples (stage_x, stage_y)
        
        # Add a custom UI button for calibration
        self.add_operation("Calibrate Pixels to um", self.calibrate_pixels_to_um)

    def setup_figure(self):
        # Create a QWidget and layout for the UI
        self.ui = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout()
        self.ui.setLayout(self.layout)
        
        # Pyqtgraph ImageView for displaying the camera feed and CV overlay
        self.image_view = pg.ImageView()
        self.layout.addWidget(self.image_view)
        
        # Add a scatter plot item to draw circles around detected flakes
        self.scatter_plot = pg.ScatterPlotItem(size=10, pen=pg.mkPen('r'), brush=pg.mkBrush(255, 0, 0, 100))
        self.image_view.getView().addItem(self.scatter_plot)
        
        # Also add a target item for the center
        self.center_target = pg.TargetItem(pos=(0, 0), size=20, movable=False, pen='g')
        self.image_view.getView().addItem(self.center_target)

    def run(self):
        self.found_flakes_stage_coords = []
        
        stage = self.app.hardware['stage']
        cam = self.app.hardware['toupcam']
        
        start_x = stage.settings['x_position']
        start_y = stage.settings['y_position']
        
        search_width = self.settings['search_width']
        search_height = self.settings['search_height']
        step_size = self.settings['step_size']
        
        # Create a basic snake raster path
        num_x_steps = max(1, int(search_width / step_size))
        num_y_steps = max(1, int(search_height / step_size))
        
        x_positions = np.linspace(start_x - search_width/2, start_x + search_width/2, num_x_steps)
        y_positions = np.linspace(start_y - search_height/2, start_y + search_height/2, num_y_steps)
        
        self.template = None
        if self.settings['search_mode'] == 'Template' and self.settings['template_path']:
            self.template = cv2.imread(self.settings['template_path'], cv2.IMREAD_GRAYSCALE)
        
        for i, y in enumerate(y_positions):
            # Snake pattern: alternate direction
            current_x_positions = x_positions if i % 2 == 0 else x_positions[::-1]
            
            for x in current_x_positions:
                if self.interrupt_measurement_called:
                    break
                
                # Move stage
                stage.settings['x_position'] = x
                stage.settings['y_position'] = y
                time.sleep(0.5) # Wait for stage to settle
                
                # Read image
                img = cam.settings['last_image']
                if img is None:
                    continue
                
                self.current_image = img.copy()
                
                # Process image
                targets = self.process_image(img)
                self.detected_targets = targets
                
                # If targets are found, auto-center on the best one (largest or highest confidence)
                if targets:
                    best_target = targets[0] # Assume sorted or just take first
                    
                    cal_const = self.settings['calibration_constant']
                    
                    h, w = img.shape[:2]
                    center_px_x = w / 2.0
                    center_px_y = h / 2.0
                    
                    # Calculate offset in pixels
                    offset_px_x = best_target['x'] - center_px_x
                    offset_px_y = best_target['y'] - center_px_y
                    
                    # Convert to microns
                    offset_um_x = offset_px_x * cal_const
                    offset_um_y = offset_px_y * cal_const
                    
                    # Move stage to center flake
                    new_x = stage.settings['x_position'] + offset_um_x
                    new_y = stage.settings['y_position'] + offset_um_y
                    
                    stage.settings['x_position'] = new_x
                    stage.settings['y_position'] = new_y
                    time.sleep(0.5) # Wait for centering
                    
                    # Record the coordinates
                    self.found_flakes_stage_coords.append((new_x, new_y))
                    
                    # Re-take image after centering to verify (optional)
                    self.current_image = cam.settings['last_image'].copy()
                    
            if self.interrupt_measurement_called:
                break
                
        # Return to start position
        stage.settings['x_position'] = start_x
        stage.settings['y_position'] = start_y
        
        # Save results
        if len(self.found_flakes_stage_coords) > 0:
            self.open_new_h5_file()
            coords_array = np.array(self.found_flakes_stage_coords)
            self.h5_meas_group.create_dataset('flake_coordinates', data=coords_array)

    def process_image(self, img):
        # 1. Extract Green channel for MoS2 contrast on SiO2
        if len(img.shape) == 3 and img.shape[2] >= 3:
            green_channel = img[:, :, 1]
        else:
            green_channel = img # Assume already grayscale
            
        # 2. Median Blur
        blurred = cv2.medianBlur(green_channel, 5)
        self.current_processed_image = blurred
        
        targets = []
        
        mode = self.settings['search_mode']
        if mode == 'Blob':
            # Segmentation
            # Using adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cal_const = self.settings['calibration_constant']
            
            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                area_um = area_px * (cal_const ** 2)
                
                # Check area (e.g., 10 to 50 um^2 or similar depending on size criteria)
                # Adjust criteria based on actual um^2 size or diameter
                if 10 < area_um < 5000:  # arbitrary range for example
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        targets.append({'x': cx, 'y': cy, 'area': area_um})
                        
            # Sort by area descending
            targets.sort(key=lambda t: t['area'], reverse=True)
            
        elif mode == 'Template' and self.template is not None:
            # Template matching
            res = cv2.matchTemplate(blurred, self.template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8 # Example threshold
            loc = np.where(res >= threshold)
            
            h, w = self.template.shape
            for pt in zip(*loc[::-1]):
                # Center of the matched template
                cx = pt[0] + w // 2
                cy = pt[1] + h // 2
                targets.append({'x': cx, 'y': cy})
                
        return targets

    def update_display(self):
        if self.current_image is not None:
            # Pyqtgraph expects image axes as (x, y, color) or (x, y)
            # OpenCV is (y, x, color)
            img_to_show = np.transpose(self.current_image, axes=(1, 0, 2) if len(self.current_image.shape)==3 else (1,0))
            self.image_view.setImage(img_to_show, autoRange=False, autoLevels=False)
            
            h, w = self.current_image.shape[:2]
            self.center_target.setPos(w/2, h/2)
            
        # Update scatter plot with detected targets
        if hasattr(self, 'detected_targets'):
            points = [{'pos': (t['x'], t['y'])} for t in self.detected_targets]
            self.scatter_plot.setData(points)

    def calibrate_pixels_to_um(self):
        """
        Move stage by known distance, use phase correlation to find pixel shift.
        """
        stage = self.app.hardware['stage']
        cam = self.app.hardware['toupcam']
        
        move_dist_um = 50.0
        
        # Get image 1
        img1 = cam.settings['last_image']
        if img1 is None:
            print("Cannot calibrate: No camera image available.")
            return
            
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            
        img1_gray = np.float32(img1_gray)
            
        # Move stage
        start_x = stage.settings['x_position']
        stage.settings['x_position'] = start_x + move_dist_um
        time.sleep(1.0) # Wait for move to complete
        
        # Get image 2
        img2 = cam.settings['last_image']
        
        if len(img2.shape) == 3:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = img2
            
        img2_gray = np.float32(img2_gray)
            
        # Move stage back
        stage.settings['x_position'] = start_x
        
        # Calculate shift
        shift, response = cv2.phaseCorrelate(img1_gray, img2_gray)
        
        dx_px, dy_px = shift
        dist_px = np.sqrt(dx_px**2 + dy_px**2)
        
        if dist_px > 0:
            cal_const = move_dist_um / dist_px
            self.settings['calibration_constant'] = cal_const
            print(f"Calibration successful: {cal_const:.4f} um/px")
        else:
            print("Calibration failed: No pixel shift detected.")

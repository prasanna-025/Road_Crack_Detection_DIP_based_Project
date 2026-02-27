import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class RoadCrackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Crack Detection System - AI Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")

        # Color Palette
        self.colors = {
            "bg": "#1e1e2e",
            "sidebar": "#181825",
            "accent": "#89b4fa",
            "text": "#cdd6f4",
            "success": "#a6e3a1",
            "danger": "#f38ba8"
        }

        self.setup_ui()
        self.current_image = None
        self.processed_image = None

    def setup_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, bg=self.colors["sidebar"], width=250)
        sidebar.pack(side="left", fill="y")

        title = tk.Label(sidebar, text="DIP CRACK DETECTOR", bg=self.colors["sidebar"], 
                         fg=self.colors["accent"], font=("Helvetica", 14, "bold"), pady=30)
        title.pack()

        # Buttons
        btn_style = {"bg": "#313244", "fg": "white", "font": ("Helvetica", 10), "activebackground": self.colors["accent"], "bd": 0, "pady": 10}
        
        self.btn_load = tk.Button(sidebar, text="📂 Load Image", command=self.load_image, **btn_style)
        self.btn_load.pack(fill="x", padx=20, pady=5)

        self.btn_realtime = tk.Button(sidebar, text="🎥 IP Camera Feed", command=self.not_implemented, **btn_style)
        self.btn_realtime.pack(fill="x", padx=20, pady=5)

        self.btn_process = tk.Button(sidebar, text="⚡ Detect Cracks", command=self.process_current, **btn_style)
        self.btn_process.pack(fill="x", padx=20, pady=30)

        # Stats Panel in Sidebar
        stats_box = tk.Frame(sidebar, bg="#313244", pady=10)
        stats_box.pack(fill="x", padx=20, pady=10)
        
        tk.Label(stats_box, text="Analysis Data", bg="#313244", fg=self.colors["text"], font=("Helvetica", 9, "bold")).pack()
        self.status_label = tk.Label(stats_box, text="Ready", bg="#313244", fg=self.colors["success"], font=("Helvetica", 9))
        self.status_label.pack()

        # Main Content Area
        self.main_content = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_content.pack(side="right", fill="both", expand=True)

        # Image Display Area
        self.display_frame = tk.Frame(self.main_content, bg=self.colors["bg"])
        self.display_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.canvas_orig = tk.Canvas(self.display_frame, bg="#11111b", bd=0, highlightthickness=0)
        self.canvas_orig.place(relx=0.05, rely=0.1, relwidth=0.4, relheight=0.6)
        tk.Label(self.display_frame, text="Original View", bg=self.colors["bg"], fg=self.colors["text"]).place(relx=0.05, rely=0.05)

        self.canvas_proc = tk.Canvas(self.display_frame, bg="#11111b", bd=0, highlightthickness=0)
        self.canvas_proc.place(relx=0.55, rely=0.1, relwidth=0.4, relheight=0.6)
        tk.Label(self.display_frame, text="Processed Detection", bg=self.colors["bg"], fg=self.colors["text"]).place(relx=0.55, rely=0.05)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.show_image(self.current_image, self.canvas_orig)
            self.status_label.config(text="Image Loaded", fg=self.colors["success"])

    def show_image(self, img, canvas):
        # Resize image to fit canvas while maintaining aspect ratio
        h, w = img.shape[:2]
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w < 10: canvas_w = 400 # Default if not yet rendered
        if canvas_h < 10: canvas_h = 400

        scale = min(canvas_w/w, canvas_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.delete("all")
        canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=img_tk)
        canvas.image = img_tk # Keep reference

    def process_current(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        self.status_label.config(text="Processing...", fg=self.colors["accent"])
        self.root.update_idletasks()

        # Simplified DIP pipeline for the GUI
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Create overlay
        overlay = self.current_image.copy()
        overlay[closed > 0] = [0, 0, 255] # Red cracks
        
        self.show_image(overlay, self.canvas_proc)
        self.status_label.config(text="Detection Complete", fg=self.colors["success"])

    def not_implemented(self):
        messagebox.showinfo("Feature", "IP Camera feed integration in GUI coming soon!")

if __name__ == "__main__":
    root = tk.Tk()
    app = RoadCrackApp(root)
    root.mainloop()

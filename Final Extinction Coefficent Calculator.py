#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import linregress
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

class ExtinctionCoefficientApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Extinction Coefficient Calculator")
        self.geometry("1200x600")

        self.path_length = None
        self.concentrations = []
        self.data = None

        self.load_button = tk.Button(self, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=10)

        self.plot_button = tk.Button(self, text="Plot Traces", command=self.plot_traces, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        self.calculate_button = tk.Button(self, text="Calculate Extinction Coefficient", command=self.calculate_extinction, state=tk.DISABLED)
        self.calculate_button.pack(pady=10)

        self.save_button = tk.Button(self, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        self.range_button = tk.Button(self, text="Set Wavelength Range", command=self.set_wavelength_range, state=tk.DISABLED)
        self.range_button.pack(pady=10)

        self.figure = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, figure=self.figure)

        self.ax1 = self.figure.add_subplot(gs[0])
        self.ax2 = self.figure.add_subplot(gs[1])

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.extinction_coeffs = []
        self.r_squared_values = []

        self.wavelength_min = None
        self.wavelength_max = None

        # Connect the canvas to Matplotlib events
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            raw_data = pd.read_csv(file_path)
            num_spectra = raw_data.shape[1] // 2

            # Separate the data into a list of DataFrames, each containing wavelength and absorbance
            data_list = []
            for i in range(num_spectra):
                data = raw_data.iloc[:, [2*i, 2*i+1]].dropna()
                data.columns = ['Wavelength', f'Absorbance_{i+1}']
                data_list.append(data)

            self.data = data_list
            self.path_length = float(simpledialog.askstring("Input", "Enter the path length (l) in cm:"))
            concentration_str = simpledialog.askstring("Input", f"Enter the concentrations (C) for each spectrum (total {num_spectra}), separated by commas:")
            self.concentrations = list(map(float, concentration_str.split(',')))

            if len(self.concentrations) != num_spectra:
                messagebox.showerror("Error", "The number of concentrations does not match the number of spectra.")
                return

            self.plot_button.config(state=tk.NORMAL)
            self.calculate_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.range_button.config(state=tk.NORMAL)

    def plot_traces(self):
        self.ax1.clear()
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=len(self.data)-1)

        for i, data in enumerate(self.data):
            color = cmap(norm(i))
            self.ax1.plot(data['Wavelength'], data[f'Absorbance_{i+1}'], label=f'C = {self.concentrations[i]} M', color=color)

        self.ax1.set_xlabel('Wavelength (nm)', fontname='Arial')
        self.ax1.set_ylabel('Absorbance (O.D.)', fontname='Arial')
        self.ax1.legend()
        self.ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))
        if self.wavelength_min and self.wavelength_max:
            self.ax1.set_xlim(self.wavelength_min, self.wavelength_max)
        self.canvas.draw()

    def set_wavelength_range(self):
        min_wavelength = float(simpledialog.askstring("Input", "Enter the minimum wavelength (nm):"))
        max_wavelength = float(simpledialog.askstring("Input", "Enter the maximum wavelength (nm):"))

        if min_wavelength < max_wavelength:
            self.wavelength_min = min_wavelength
            self.wavelength_max = max_wavelength
            self.slice_data()
            self.plot_traces()
            self.calculate_extinction()
        else:
            messagebox.showerror("Error", "Minimum wavelength must be less than maximum wavelength.")

    def slice_data(self):
        # Filter each DataFrame in self.data based on the wavelength range
        for i, data in enumerate(self.data):
            self.data[i] = data[(data['Wavelength'] >= self.wavelength_min) & (data['Wavelength'] <= self.wavelength_max)]

    def calculate_extinction(self):
        self.extinction_coeffs = []
        self.r_squared_values = []

        # Ensure data slicing if wavelength range is set
        if self.wavelength_min and self.wavelength_max:
            self.slice_data()

        wavelengths = self.data[0]['Wavelength']

        for wavelength in wavelengths:
            absorbances = [data[data['Wavelength'] == wavelength].iloc[0, 1] for data in self.data]
            slope, intercept, r_value, _, _ = linregress(self.concentrations, absorbances)
            self.extinction_coeffs.append(slope / self.path_length)
            self.r_squared_values.append(r_value**2)

        self.ax2.clear()
        self.ax2.plot(wavelengths, self.extinction_coeffs, label='Extinction Coefficient (ε)', color='black')
        self.ax2.set_xlabel('Wavelength (nm)', fontname='Arial')
        self.ax2.set_ylabel('Extinction Coefficient (ε) ($cm^{-1}$ $M^{-1}$)', fontname='Arial')
        self.ax2.legend()
        self.ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both'))
        if self.wavelength_min and self.wavelength_max:
            self.ax2.set_xlim(self.wavelength_min, self.wavelength_max)
        self.canvas.draw()

    def save_results(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            wavelengths = self.data[0]['Wavelength']
            results = pd.DataFrame({
                'Wavelength (nm)': wavelengths,
                'Extinction Coefficient (ε) (cm⁻¹ M⁻¹)': self.extinction_coeffs,
                'R²': self.r_squared_values
            })
            results.to_csv(file_path, index=False)

            self.figure.savefig(file_path.replace('.csv', '.png'), dpi=300)
            self.figure.savefig(file_path.replace('.csv', '.pdf'), dpi=300)

            messagebox.showinfo("Success", "Results saved successfully!")

    def on_click(self, event):
        pass  # Placeholder for potential future interactive features

    def on_key_press(self, event):
        pass  # Placeholder for potential future interactive features

if __name__ == "__main__":
    app = ExtinctionCoefficientApp()
    app.mainloop()


# In[ ]:





# In[ ]:





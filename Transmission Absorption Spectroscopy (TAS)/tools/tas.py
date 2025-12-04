import os
from os import listdir
import sys
import struct
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import yaml
import ipywidgets as ipw
from ipywidgets import Button, Layout
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cmath
import re
import yaml
from scipy.optimize import curve_fit
from scipy.special import erfc
import matplotlib.pyplot as plt


settingsFile = 'tools/settings.yaml'

class dataTools :
    
    def __init__(self) :

        pass

    def loadData(self,folder,file) :
        
        npz_file = np.load(folder+'/'+file)                        # arrays stored in npz files are accessed in the same way you access python dictionaries
        data = {key: npz_file[key] for key in npz_file.files}
        npz_file.close()
        data['wl'] = 1240/data["energy_axis"]
        
        return data
                
    def plotData(self,data) :
        
        wl_max = 600
        wl_min = 530
        wl_selected = np.logical_and(data['wl'] < wl_max, data['wl'] > wl_min)
        index = np.nonzero(wl_selected)[0]

        min_level = np.min(data["TR_RC_avg"][:, index])
        max_level = np.max(data["TR_RC_avg"][:, index])
        level = np.max([np.abs(min_level), np.abs(max_level)])

        plt.figure()
        plt.contourf(data['wl'][index], data["time_axis"], data["TR_RC_avg"][:, index], levels=np.linspace(-level,level, 1000), cmap = 'seismic')
        plt.title("Data")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Time Delay (ps)')
        plt.colorbar(label="signal")
        plt.show()


class analysisTools :
    
    def __init__(self) :

        pass

    def findTau(self, data, wl_cut) :
        
        wl = data['wl']
        t = data['time_axis']
        
        def exponental(t, A, tau, C):
            return A*np.exp(-(t)/tau) + C  
        
        cut_indices = []
        for loc in [wl_cut]:
            close_indices = np.where(np.abs(wl-loc) < 1)
            cut_indices.append(close_indices[0][len(close_indices)//2+1])
        cols = plt.cm.viridis(np.linspace(0,1,len(cut_indices)))

        plt.figure()
        for i, ind in enumerate(cut_indices):
            dataRC_t = np.transpose(data['TR_RC_avg'])
            y = dataRC_t[ind]

            peak_idx = np.argmax(np.abs(y - np.median(y))) # find peak
            peak_val = y[peak_idx]
            peak_time = t[peak_idx]
            t_fit = t[peak_idx:] - peak_time
            y_fit = y[peak_idx:]

            # initial guess
            A0 = peak_val - y_fit[-1]
            tau0 = 2.0
            C0 = y_fit[-1]
            guess = [A0, tau0, C0]
            popt, pcov = curve_fit(exponental, t_fit, y_fit,guess)
            t_smooth = np.linspace(t_fit.min(), t_fit.max(), 81503)
            y_smooth = exponental(t_smooth, *popt)
            
            col = cols[i]
            plt.plot(t, y,'.',color=col, label =f"{round(wl[ind],2)} nm")
            plt.plot(t_smooth+peak_time, y_smooth, '-', color=col)

            sd = np.sqrt(np.diag(pcov))
            print(f"Wavelength {wl[ind]:.2f} nm, tau = {popt[1]:.3f} ± {sd[1]:.3f} ps")

        plt.xlabel("Time Delay (ps)")
        plt.ylabel("Signal")
        plt.title("Au Time-Resolved Electrodynamics")
        plt.legend(title="wl (nm)")
        plt.show()


class UI :
    
    def __init__(self) :

        dt = dataTools()
        at = analysisTools()

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        
        with open(settingsFile, 'r') as stream :
            settings = yaml.safe_load(stream)
            
        if os.path.isdir(settings['folders']['current']) :
            self.cwd = settings['folders']['current']
        else :
            self.cwd = str(Path(os.getcwd()))
        
        out = ipw.Output()
        out2 = ipw.Output()

        def gotoCurFolder(address):
            address = Path(address)
            if address.is_dir():
                currentFolder.value = str(address)
                SelectFolder.unobserve(selecting, names='value')
                SelectFolder.options = self.get_folder_contents(folder=address)[0]
                SelectFolder.observe(selecting, names='value')
                SelectFolder.value = None
                selectFile.options = self.get_folder_contents(folder=address)[1]
                settings['folders']['current'] = str(address)
                with open(settingsFile, 'w') as f:
                    yaml.dump(settings, f)

        def newCurFolder(value):
            gotoCurFolder(currentFolder.value)
        currentFolder = ipw.Text(value=str(self.cwd),
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder')
        currentFolder.on_submit(newCurFolder)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(currentFolder.value)
                newpath = path / value['new']
                if newpath.is_dir():
                    gotoCurFolder(newpath)
                elif newpath.is_file():
                    #some other condition
                    pass
        
        SelectFolder = ipw.Select(
            options=self.get_folder_contents(self.cwd)[0],
            rows=5,
            value=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Subfolders')
        SelectFolder.observe(selecting, names='value')
        
        selectFile = ipw.Select(
            options=self.get_folder_contents(self.cwd)[1],
            rows=10,
            values=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Files')

        def parent(value):
            new = Path(currentFolder.value).parent
            gotoCurFolder(new)
        up_button = ipw.Button(description='Up',layout=Layout(width='10%'))
        up_button.on_click(parent)

        def ShowData_Clicked(b) :
            with out :
                clear_output(True)
                data = dt.loadData(currentFolder.value,selectFile.value)
                self.data = data
                dt.plotData(data)
                display(ipw.HBox([findTau,tau_wl]))
            with out2 :
                clear_output(True)
                data = self.data
                at.findTau(data,tau_wl.value)
        ShowData = ipw.Button(description="Load data")
        ShowData.on_click(ShowData_Clicked)

        def findTau_Clicked(b) :
            with out2 :
                clear_output(True)
                data = self.data
                at.findTau(data,tau_wl.value)
        findTau = ipw.Button(description="Find tau")
        findTau.on_click(findTau_Clicked)
        
        tau_wl = ipw.BoundedFloatText(
            value=settings['tau']['wl'],
            min=500,
            max=1000,
            description='Wavelength')
        
        display(ipw.HBox([currentFolder]))
        display(ipw.HBox([SelectFolder,up_button]))
        display(ipw.HBox([selectFile]))
        display(ipw.HBox([ShowData]))

        display(out)
        display(out2)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)

import numpy as np
from scipy import integrate
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
from lmfit.models import SkewedGaussianModel
import ipywidgets as ipw
from ipywidgets import Layout
from IPython.display import clear_output
import os
from pathlib import Path
import io


settingsFile = 'tools/settings.yaml'

class dataTools :

    def __init__(self) :

        pass

    def loadData(self,folder,file) :
        
        filepath = folder + '/' + file
        
        try :
            with open(filepath) as f:
                Content = f.readlines()
            dataLength = list()
            for index in range(len(Content)):
                dataLength.append(len(Content[index].split('\t')))
            dataStart = list()
            dataEnd = list()
            Counter = 0
            for index in range(len(dataLength)):
                if dataLength[index] == 1 :
                    if Counter > 1 : dataEnd.append(index-1)
                    Counter = 0
                else :
                    if Counter == 0 : dataStart.append(index+3)
                    Counter = Counter + 1
            Header = list()
            dataSets = 0
            for index in range(len(dataStart)):
                dataSets = dataSets + dataLength[dataStart[index]]
            data = np.zeros((dataSets,dataEnd[0]-dataStart[0]+1))
            for index in range(len(dataStart)):
                for x in range(dataLength[dataStart[index]]):
                    Header.append(Content[dataStart[index]-2].split('\t')[x])
                    for y in range(dataEnd[0]-dataStart[0]+1):
                        if Content[dataStart[index]+y].split('\t')[x] != '' :
                            data[x + index * dataLength[dataStart[index-1]]][y] = Content[dataStart[index]+y].split('\t')[x]
            for index in range(2,int(len(data)/2)+1):
                data = np.delete(data,index,axis=0)
                Header.remove(Header[index-1])
            Header.remove(Header[-1])
            Header.insert(0,'X')
            data = pd.DataFrame(data=np.transpose(data),columns=Header)
            data = data[data.columns.drop(list(data.filter(regex='ExCorr')))]
            data.set_index('X', inplace=True)
            print('Data file loaded successfully.')
        
        except :
            
            data = pd.DataFrame(columns=[''])
            print('Error loading data file.')
        
        return data
    
    def createSpectra(self,data,Runs,Buffer='None',CamCorrection=False) :
        
        try:
            Runs = np.array(Runs, dtype=float)
        except:
            pass
        
        if Buffer != 'None' :
            data = data.sub(data[Buffer], axis=0)
        
        if CamCorrection :
            
            with open(settingsFile, 'r') as stream :
                settings = yaml.safe_load(stream)
            file_path = settings['files']['calmodulin']
            try:
                with open(file_path, 'r'):
                    Cam = pd.read_csv(file_path)
                    Cam.set_index('X', inplace=True)
                    for name in data.columns :
                        scaling = np.average(data[name].values[0:3] / Cam['Y'].values[0:3])
                        data[name] -= scaling*Cam['Y']
                        
            except FileNotFoundError:
                print("Calmodulin data does not exist. Skipping tyrosine removal.")
        
        data = data.filter(items=Runs)
        
        return data
    
    def plot(self,data,Title='') :
        
        fontsize = 16
        fig, ax = plt.subplots(figsize=(10,8))
        for name in data.columns :
            plt.plot(data[name],label=name)
        plt.plot(data)
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), fontsize=fontsize*0.75)
        plt.xlabel('Wavelength (nm)',fontsize=fontsize)
        plt.ylabel('Intensity (au)',fontsize=fontsize)
        plt.title(Title, fontsize=fontsize)
        ax.tick_params(axis='both',which='both',labelsize=fontsize,direction="in")
        ax.minorticks_on()
        plt.show()
        
        return fig

    def plot2D(self,data,Title='') :
        
        fontsize = 14
        data = data.T
        plt.figure(figsize=(10,6))
        im = plt.imshow(data.values, 
                        origin='lower',         # Lower-left corner is (0,0)
                        cmap='turbo',           # Colormap
                        aspect='auto',          # Adjust aspect ratio
                        extent=[data.columns.min(), data.columns.max(),
                                data.index.min(), data.index.max()])
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label("Intensity", fontsize=fontsize)
        plt.xlabel('Wavelength (nm)',fontsize=fontsize)
        plt.ylabel('Run',fontsize=fontsize)
        plt.tick_params(axis='both', which='both', labelsize=fontsize, direction="in")
        plt.yticks([])
        plt.show()


class analysisTools :

    def __init__(self) :

        pass

    def Integrate(self, data, limits):
        data = data.loc[(data.index >= min(limits)) & (data.index <= max(limits))]
        integratedValues = list()
        for idx, column in enumerate(data) :
            x = data.index.to_numpy()
            y = data[column].to_numpy()
            integratedValues.append(integrate.trapezoid(y,x=x))
        integratedValues = pd.DataFrame(data=integratedValues,index=data.columns,columns=['Integrated'])
        fontsize = 14
        plt.figure(figsize=(12,4))
        plt.xlabel('Run',fontsize=fontsize), plt.ylabel('Integrated Value',fontsize=fontsize)
        plt.plot(integratedValues, '.-')
        plt.tick_params(axis="x", labelsize=fontsize, rotation=-90)
        plt.tick_params(axis="y", labelsize=fontsize)
        plt.show()
        
        return integratedValues

    def Fit(self,data,limits) :
        
        fontsize = 14
        integratedAreas = list()
        peakValues = list()
        data2fit = data.loc[(data.index >= min(limits)) & (data.index <= max(limits))]
        plt.figure(figsize=(10, 6))
        x_vals = data2fit.index.to_numpy()
        for col in data2fit.columns:
            y_vals = data2fit[col].to_numpy()
            model = SkewedGaussianModel()
            params = model.guess(y_vals, x=x_vals)
            result = model.fit(y_vals, params, x=x_vals)
            plt.scatter(data.index.to_numpy(), data[col].to_numpy(), s=20)
            plt.plot(x_vals, result.best_fit, linewidth=2, label=f'{col}')
            
            integratedAreas.append(result.params['amplitude'].value)
            peakValues.append(x_vals[np.argmax(result.best_fit)])

        # Finalize plot
        plt.xlabel('Wavelength (nm)',fontsize=fontsize)
        plt.ylabel('Intensity (au)',fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize)
        plt.tick_params(axis="y", labelsize=fontsize)
        plt.title('Fits')
        plt.legend(frameon=False)
        plt.grid(True)
        plt.show()
        
        # Plot integrated areas
        plt.figure(figsize=(10, 3))
        plt.plot(data2fit.columns, integratedAreas, 'o-')
        plt.xlabel('Wavelength (nm)',fontsize=fontsize)
        plt.ylabel('Intensity (au)',fontsize=fontsize)
        plt.tick_params(axis="both", labelsize=fontsize)
        plt.title('Integrated areas')
        plt.grid(True)
        plt.show()
        
        # Plot peak values
        plt.figure(figsize=(10, 3))
        plt.plot(data2fit.columns, peakValues, 'o-')
        plt.xlabel('Wavelength (nm)',fontsize=fontsize)
        plt.ylabel('Intensity (au)',fontsize=fontsize)
        plt.tick_params(axis="both", labelsize=fontsize)
        plt.title('Peak values')
        plt.grid(True)
        plt.show()
        
        df = pd.DataFrame()
        df[']Integrated Areas'] = integratedAreas
        df['Peak Values'] = peakValues
        df.index = data.columns
        
        return df


class UI :
    
    def __init__(self) :

        dt = dataTools()
        at = analysisTools()
        
        self.BufferNames = ['None']
        self.Names = ['']
        self.titration = None

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        self.settingsFile = settingsFile
        
        with open(settingsFile, 'r') as stream :
            settings = yaml.safe_load(stream)

        if os.path.isdir(settings['folders']['data']) :
            self.cwd = settings['folders']['data']
        else :
            self.cwd = str(Path(os.getcwd()))

        out = ipw.Output()
        anout = ipw.Output()

        def go_to_address(address):
            address = Path(address)
            if address.is_dir():
                folderField.value = str(address)
                SelectFolder.unobserve(selecting, names='value')
                SelectFolder.options = self.get_folder_contents(folder=address)[0]
                SelectFolder.observe(selecting, names='value')
                SelectFolder.value = None
                SelectFile.options = self.get_folder_contents(folder=address)[1]
                settings['folders']['data'] = str(address)
                with open(settingsFile, "w") as outfile:
                    yaml.dump(settings,outfile,default_flow_style=False)

        def newaddress(value):
            go_to_address(folderField.value)
        folderField = ipw.Text(
            value=self.cwd,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder'
            )
        folderField.on_submit(newaddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(folderField.value)
                newpath = path / value['new']
                if newpath.is_dir():
                    go_to_address(newpath)
                elif newpath.is_file():
                    pass
        
        SelectFolder = ipw.Select(
            options=self.get_folder_contents(self.cwd)[0],
            rows=5,
            value=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Subfolders')
        SelectFolder.observe(selecting, names='value')
        
        SelectFile = ipw.Select(
            options=self.get_folder_contents(self.cwd)[1],
            rows=10,
            values=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Files')

        def parent(value):
            new = Path(folderField.value).parent
            go_to_address(new)
        up_button = ipw.Button(description='Up',layout=Layout(width='10%'))
        up_button.on_click(parent)
            
        def load(b):
            with out :
                clear_output()
            with anout :
                clear_output()
            self.Buffer.value = 'None'
            self.Runs_Selected.value = []
            self.data = dt.loadData(folderField.value,SelectFile.value)
            self.filename = SelectFile.value
            RunList()
            Upload_Titration.value = ()
            Upload_Titration._counter=0
        load_button = ipw.Button(description='Load',layout=Layout(width='10%'))
        load_button.on_click(load)

        def RunList():
            self.Runs = list(self.data.columns.values.astype(str))
            Runs = [k for k in self.Runs if self.Filter.value in k]
            self.Runs_Selected.options = Runs
            Runs.insert(0,'None')
            self.Buffer.options = Runs

        self.Filter = ipw.Text(
            value='',
            placeholder='Type something',
            description='Filter:',
            style = {'description_width': '150px'},
            disabled=False
        )
        
        def Update_RunList_Clicked(b):
            RunList()
        Update_RunList = ipw.Button(description="Update run list")
        Update_RunList.on_click(Update_RunList_Clicked)

        # Create upload widget
        Upload_Titration = ipw.FileUpload(
            accept='.csv, .xlsx',  # allowed file types
            multiple=False,
            description='Select Titration File',
            layout=ipw.Layout(width='170px')
        )

        def on_uploadTitration_change(change):
            data = self.data
            titration = self.titration
            if not Upload_Titration.value:
                print("⚠️ No file uploaded.")
                return
            for file_info in Upload_Titration.value:
                filename = file_info['name']
                content = file_info['content']
                print(f"📂 Uploaded: {filename}")
                try:
                    if filename.lower().endswith('.csv'):
                        titration = pd.read_csv(io.BytesIO(content))
                    elif filename.lower().endswith('.xlsx'):
                        titration = pd.read_excel(io.BytesIO(content))
                    else:
                        print("❌ Unsupported file type.")
                        titration = None
                        return
                    
                    print("✅ File loaded successfully!")
                except Exception as e:
                    print(f"❌ Error reading file: {e}")
                    titration = None
            self.titration = titration
            if data.shape[1] == titration.shape[0]:
                data.columns = titration.iloc[:,0].values
            else:
                print('Titration data length does not match number of runs.')
            RunList()
            print('ok')
        Upload_Titration.observe(on_uploadTitration_change, names='value')

        self.Runs_Selected = ipw.SelectMultiple(
            options='',
            style = {'width': '100px','description_width': '150px'},
            rows=20,
            layout=Layout(width='70%'),
            description='Runs',
            disabled=False
        )

        self.Buffer = ipw.Dropdown(
            options=self.BufferNames,
            value='None',
            layout=Layout(width='70%'),
            description='Buffer',
            style = {'description_width': '150px'},
            disabled=False,
        )
        
        self.CamCorrection = ipw.Checkbox(
            value=False,
            description='Subtract CaM?',
            disabled=False,
            indent=False
        )

        def Plot_Clicked(b):
            with out :
                data = self.data
                clear_output()
                spectra = dt.createSpectra(data,self.Runs_Selected.value,Buffer=self.Buffer.value,CamCorrection=self.CamCorrection.value)
                self.fig = dt.plot(spectra)
                display(ipw.Box([SavePlot,SpectraToClipboard,Plot2D]))
                display(ipw.Box([Integrate,Fit]))
                display(LowLim)
                display(UpLim)
            with anout :
                clear_output()
            LowLim.max = max(spectra.index)
            LowLim.min = min(spectra.index)
            LowLim.value = min(spectra.index)
            UpLim.max = max(spectra.index)
            UpLim.min = min(spectra.index)
            UpLim.value = max(spectra.index)
            self.spectra = spectra
        Plot = ipw.Button(description="Plot")
        Plot.on_click(Plot_Clicked)
        
        def Plot2d_Clicked(b):
            with anout :
                clear_output()
                dt.plot2D(self.spectra)
        Plot2D = ipw.Button(description="Plot 2D")
        Plot2D.on_click(Plot2d_Clicked)

        def Integrate_clicked(b):
            with anout :
                clear_output()
                limits = [LowLim.value, UpLim.value]
                self.integratedValues = at.Integrate(self.spectra, limits)
                display(IntegratedToClipboard)
        Integrate = ipw.Button(description="Integrate")
        Integrate.on_click(Integrate_clicked)
        
        def Fit_clicked(b):
            with anout :
                clear_output()
                limits = [LowLim.value, UpLim.value]
                self.fitAnalysis = at.Fit(self.spectra, limits)
                display(FitAnalysisToClipboard)
        Fit = ipw.Button(description="Fit")
        Fit.on_click(Fit_clicked)

        def SavePlot_Clicked(b):
            self.fig.savefig(self.filename.replace('.txt','.jpg'),bbox_inches='tight')
        SavePlot = ipw.Button(description="Save Plot")
        SavePlot.on_click(SavePlot_Clicked)
        
        def SpectraToClipboard_Clicked(b):
            dataToSave = self.spectra
            dataToSave.to_clipboard()
        SpectraToClipboard = ipw.Button(description="Copy Spectra")
        SpectraToClipboard.on_click(SpectraToClipboard_Clicked)

        def IntegratedToClipboard_Clicked(b):
            dataToSave = self.integratedValues
            display(dataToSave)
            dataToSave.to_clipboard()
        IntegratedToClipboard = ipw.Button(description="Copy Integrated Data")
        IntegratedToClipboard.on_click(IntegratedToClipboard_Clicked)
        
        def FitAnalysisToClipboard_Clicked(b):
            dataToSave = self.fitAnalysis
            display(dataToSave)
            dataToSave.to_clipboard()
        FitAnalysisToClipboard = ipw.Button(description="Copy Fit Analysis")
        FitAnalysisToClipboard.on_click(FitAnalysisToClipboard_Clicked)
        
        LowLim = ipw.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Lower Limit:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
            )

        UpLim = ipw.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Upper Limit:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
            )

        display(ipw.HBox([folderField]))
        display(ipw.HBox([SelectFolder,up_button]))
        display(ipw.HBox([SelectFile,load_button]))
        display(ipw.Box([self.Filter,Update_RunList,Upload_Titration]))
        display(ipw.Box([self.Runs_Selected,Plot]))
        display(ipw.Box([self.Buffer,self.CamCorrection]))

        display(out)
        display(anout)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)
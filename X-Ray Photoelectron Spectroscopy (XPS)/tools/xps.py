import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import ipywidgets as ipw
from ipywidgets import Button, Layout
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel
from lmfitxps.models import ShirleyBG
import re
import os
from os import listdir
from os.path import isfile, join, dirname
import sys
from pathlib import Path
from IPython.display import clear_output
from pylab import rc

pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.95,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 0,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)',
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=50, #top margin
        )
))

settingsFile = 'tools/settings.yaml'

class dataTools () :

    def __init__(self) :

        self.regions = ['C1s','O1s','Pt4f','Ef','Overview']
    
    def loadData(self, par, normalize=True) :
        
        data = {}
        data['run'] = par['run']
        for region in par :
            if region in self.regions :
                data[region] = {}
                channel = par[region]['Channel']
                newdata = pd.read_csv(Path(par['folder'] +'/'+par['run']+'_'+str(channel)+'.csv'))
                for idx, col in enumerate(newdata) :
                    if idx == 0 :
                        data[region]['x'] = newdata[col].values
                        if 'xOffset' in par :
                            data[region]['x'] -= par['xOffset']
                            data[region]['x'] = -data[region]['x']
                    else :
                        if idx == 1 :
                            data[region]['y'] = newdata[col].values.copy()
                        else :
                            data[region]['y'] += newdata[col].values.copy()
                        data[region]['y'+str(idx)] = newdata[col].values
                data[region]['y'] /= idx
                if normalize :
                    for col in data[region] :
                        if 'y' in col :
                            normalization = np.mean(data[region][col][-11:-1])
                            data[region][col] = data[region][col] / normalization - 1
                if 'yFactor' in par[region] :
                    for y in data[region] :
                        if 'y' in y :
                            data[region][y] /= par[region]['yFactor']
        return data
    
    def plotData(self,data) :

        for region in data :
            if region in self.regions :
                fig = go.Figure()
                fig.update_layout(xaxis_title="Energy (eV)",yaxis_title="Intensity (au)",title=region+', '+data['run'],font=dict(size=18),
                    autosize=False,width=1000,height=600)
                x = data[region]['x']
                y = data[region]['y']
                fig.add_trace(go.Scatter(x=x, y=y))
                fig.show()


class analysisTools :
    
    def __init__(self, data, par) :

        self.data = data
        self.par = par

    def setModel(self, region) :

        data = self.data[region]
        par = self.par[region]['Models']
        
        ModelString = list()
        for Peak in par :
            ModelString.append((Peak,par[Peak]['model']))
        
        for Model in ModelString :
            try :
                fitModel
            except :
                if Model[1] == 'Constant' :
                    fitModel = ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    fitModel = LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    fitModel = GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    fitModel = SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    fitModel = VoigtModel(prefix=Model[0]+'_')
                if Model[1] == 'Shirley' :
                    fitModel = ShirleyBG(prefix=Model[0]+'_')
            else :
                if Model[1] == 'Constant' :
                    fitModel = fitModel + ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    fitModel = fitModel + LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    fitModel = fitModel + GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    fitModel = fitModel + SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    fitModel = fitModel + VoigtModel(prefix=Model[0]+'_')
                if Model[1] == 'Shirley' :
                    fitModel = ShirleyBG(prefix=Model[0]+'_')
        
        modelParameters = fitModel.make_params()
        names = list()
        for col in data :
            if 'y' in col :
                names.append(col)
        fitParameters = df(index=modelParameters.keys(),columns=names)
        
        self.fitModel[region] = fitModel
        self.modelParameters[region] = modelParameters
        self.fitParameters[region] = fitParameters
    
    def setParameters(self, region) :

        data = self.data[region]
        par = self.par[region]['Models']
        modelParameters = self.modelParameters[region]
        
        ParameterList = ['intercept','offset','amplitude','center','sigma', 'const', 'k']
        Parameters = {'Standard': par}

        for Dictionary in Parameters :
            for Peak in Parameters[Dictionary] :
                for Parameter in Parameters[Dictionary][Peak] :
                    if Parameter in ParameterList :
                        for Key in Parameters[Dictionary][Peak][Parameter] :
                            if Key != 'set' :
                                exec('modelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                            else :
                                exec('modelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))
        
        names = list()
        for col in data :
            if 'y' in col :
                names.append(col)
        self.modelParameters[region] = modelParameters
        self.fitParameters[region] = df(index=modelParameters.keys(),columns=names)
    
    def fit(self, region, **kwargs) :

        data = self.data[region]
        self.setModel(region)
        modelParameters = self.modelParameters[region]
        fitParameters = self.fitParameters[region]
        fitModel = self.fitModel[region]

        names = list()
        for col in data :
            if 'y' in col :
                names.append(col)
        
        fits = df(index=data['x'],columns=names)
        fitResults = list()
        fitComponents = list()
        
        for idx,col in enumerate(data) :

            if 'y' in col :
                self.setParameters(region)
                x = data['x']
                y = data[col]
                fit_results = fitModel.fit(y, modelParameters, x=x, nan_policy='omit')
                fit_comps = fit_results.eval_components(fit_results.params, x=x)
                fit_y = fit_results.eval(x=x)
                ParameterNames = [i for i in fit_results.params.keys()]
                for Parameter in (ParameterNames) :
                    fitParameters.loc[Parameter, col] = fit_results.params[Parameter].value
                fits[names[idx-1]] = fit_y
                fitResults.append(fit_results)
                fitComponents.append(fit_comps)
                
                sys.stdout.write(("\rFitting %i out of "+str(len(data)-1)) % (idx))
                sys.stdout.flush()
        
        self.fits[region] = fits
        self.fitParameters[region] = fitParameters
        self.fitResults[region] = fitResults
        self.fitComponents[region] = fitComponents
    
    def fitData(self) :
        
        data = self.data
        par = self.par

        self.fits = {}
        self.fitParameters = {}
        self.fitResults = {}
        self.fitComponents = {}
        self.fitModel = {}
        self.modelParameters = {}
        
        print('Data: '+data['run'])
        
        ##### Fit Data #####

        regions = ['C1s','O1s','Pt4f','Ef']
        
        for region in par :
            if region in regions :
                if 'Models' in par[region] :
                    self.fit(region)
                    fits = self.fits[region]
                    fitParameters = self.fitParameters[region]
                    fitComponents = self.fitComponents[region]
        
                    print('\n'+100*'_')
        
                    for idx,col in enumerate(data[region]) :
                        if 'y' in col :
                            plt.figure(figsize = [12,4])
                            plt.plot(data[region]['x'], data[region][col],'k.', label='Data')
                            plt.plot(fits.index, fits[col], 'r-', label='Fit')
                            plt.xlabel('Energy (eV)'), plt.ylabel('Intensity (au)')
                            plt.title(region+', '+str(col))
                            for Component in fitComponents[idx-1] :
                                Peak = Component[:-1]
                                if not isinstance(fitComponents[idx-1][Component],float) :
                                    if 'assignment' in par[region]['Models'][Peak] :
                                        label = par[region]['Models'][Peak]['assignment']
                                    else :
                                        label = Peak
                                    plt.fill(fits.index, fitComponents[idx-1][Component], '--', label=label, alpha=0.5)
                            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
                            plt.show()
                            
                            Peaks = list()
                            for Parameter in fitParameters.index :
                                Name = Parameter.split('_')[0]
                                if Name not in Peaks :
                                    Peaks.append(Name)
                            string = ''
                            for Peak in Peaks :
                                if 'assignment' in par[region]['Models'][Peak] :
                                    string += par[region]['Models'][Peak]['assignment'] + ' | '
                                else :
                                    string += Peak + ' | '
                                for Parameter in fitParameters.index :
                                    if Peak == Parameter.split('_')[0] : 
                                        string += Parameter.split('_')[1] + ': ' + str(round(fitParameters[col][Parameter],2))
                                        string += ', '
                                string = string[:-2] + '\n'
                            print(string)
                            print(100*'_')


class UI :
    
    def __init__(self) :

        dt = dataTools()

        self.cwd = Path(os.getcwd())

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        self.parFile = settingsFile

        with open(settingsFile, 'r') as stream :
            self.folders = yaml.safe_load(stream)['folders']
        
        out = ipw.Output()
        anout = ipw.Output()

        dataFolder = ipw.Text(value=self.folders['data'],
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Data Folder')

        def changeDataFolder(value) :
            if value['new'] :
                with open(self.parFile, 'r') as f :
                    data = yaml.safe_load(f)
                data['folders']['data'] = dataFolder.value
                self.folders['data'] = dataFolder.value
                with open(self.parFile, 'w') as f:
                    yaml.dump(data, f)
                print('cool')
        dataFolder.observe(changeDataFolder, names='value')

        def goToAddress(address):
            address = Path(address)
            if address.is_dir():
                currentFolder.value = str(address)
                selectFolder.unobserve(selecting, names='value')
                selectFolder.options = self.getFolderContents(folder=address)[0]
                selectFolder.observe(selecting, names='value')
                selectFolder.value = None
                selectFile.options = self.getFolderContents(folder=address)[1]

        def newAddress(value):
            goToAddress(currentFolder.value)
        currentFolder = ipw.Text(value=str(self.cwd),
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder')
        currentFolder.on_submit(newAddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(currentFolder.value)
                newpath = path / value['new']
                if newpath.is_dir():
                    goToAddress(newpath)
                elif newpath.is_file():
                    pass
        
        selectFolder = ipw.Select(
            options=self.getFolderContents(self.cwd)[0],
            rows=5,
            value=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Subfolders')
        selectFolder.observe(selecting, names='value')
        
        selectFile = ipw.Select(
            options=self.getFolderContents(self.cwd)[1],
            rows=10,
            values=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Files')

        def parent(value):
            new = Path(currentFolder.value).parent
            goToAddress(new)
        up_button = ipw.Button(description='Up',layout=Layout(width='10%'))
        up_button.on_click(parent)

        def load(b):
            self.loadRuns(selectFile.value,currentFolder.value)
            selectRun.options = self.runNames
        load_button = ipw.Button(description='Load',layout=Layout(width='10%'))
        load_button.on_click(load)

        selectRun = ipw.Dropdown(
            options=[''],
            description='Select Run',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )
        selectRun.observe(selecting, names='value')

        def showData_Clicked(b) :
            with out :
                clear_output(True)
                par = self.runs[selectRun.value]
                par['run'] = selectRun.value
                par['folder'] = dataFolder.value
                self.data = dt.loadData(par)
                dt.plotData(self.data)
        showData = ipw.Button(description="Show Data")
        showData.on_click(showData_Clicked)
        
        def fitData_Clicked(b) :
            with out :
                clear_output(True)
                par = self.runs[selectRun.value]
                par['run'] = selectRun.value
                par['folder'] = dataFolder.value
                self.data = dt.loadData(par)
                self.fits = analysisTools(self.data, par)
                self.fits.fitData()
        fitData = ipw.Button(description="Fit Data")
        fitData.on_click(fitData_Clicked)
        
        display(ipw.HBox([dataFolder]))
        display(ipw.HBox([currentFolder]))
        display(ipw.HBox([selectFolder,up_button]))
        display(ipw.HBox([selectFile,load_button]))
        display(ipw.HBox([selectRun]))
        display(ipw.HBox([showData,fitData]))

        display(out)
        display(anout)
    
    def getFolderContents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)

    def loadRuns(self,file,folder='') :
        self.file = file
        self.folder = folder
        with open(str(folder)+'/'+file, 'r') as stream:
            self.runs = yaml.safe_load(stream)
        self.runNames = list()
        for runName in self.runs :
            self.runNames.append(runName)
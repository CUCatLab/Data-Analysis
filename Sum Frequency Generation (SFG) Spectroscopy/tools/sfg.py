import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import cmath
import yaml
import re
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as ipw
from ipywidgets import Button, Layout
from IPython.display import clear_output
from . import sif_open as sif

settingsFile = 'tools/settings.yaml'


class dataTools :
    
    def __init__(self) :

        pass

    def loadData(self, file, folder) :
        
        def FolderPath(file, folder) :
            date = re.split(r'sfg|_',file)[1]
            return folder+'/'+'20'+date[0:2]+'/'+'20'+date[0:2]+'.'+date[2:4]+'.'+date[4:6]

        with open(file, 'r') as stream :
            parameters = yaml.safe_load(stream)
        
        if 'FolderPath' not in parameters :
            parameters['FolderPath'] = FolderPath(file, folder)
        
        Data = self.loadSFG(parameters)
        Threshold = parameters['Background']['Threshold']
        data = self.removeEmptyDataSets(Data,Threshold)
        
        if 'Files' in parameters['Background'] :
            xlist = list()
            ylist = list()
            for bgFile in parameters['Background']['Files'] :
                print('Loading background file: '+parameters['Background']['Files'][bgFile]['FileName'])
                if 'FolderPath' not in parameters['Background']['Files'][bgFile] :
                    parameters['Background']['Files'][bgFile]['FolderPath'] = FolderPath(parameters['Background']['Files'][bgFile]['FileName'])
                TempData = self.loadSFG(parameters['Background']['Files'][bgFile])
                xlist.append(TempData.index.values)
                ylist.append(np.transpose(TempData.values)[0])
            background = np.array((np.average(xlist,axis=0),np.average(ylist,axis=0)))
        
        return data, parameters
    
    def loadBackground(self, file, folder) :

        def FolderPath(file, folder) :
            date = re.split(r'sfg|_',file)[1]
            return folder+'/'+'20'+date[0:2]+'/'+'20'+date[0:2]+'.'+date[2:4]+'.'+date[4:6]

        with open(file, 'r') as stream :
            parameters = yaml.safe_load(stream)
        
        if 'FolderPath' not in parameters :
            parameters['FolderPath'] = FolderPath(file, folder)
            
        xlist = list()
        ylist = list()
        for bgFile in parameters['Background']['Files'] :
            print('Loading background file: '+parameters['Background']['Files'][bgFile]['FileName'])
            if 'FolderPath' not in parameters['Background']['Files'][bgFile] :
                parameters['Background']['Files'][bgFile]['FolderPath'] = FolderPath(parameters['Background']['Files'][bgFile]['FileName'])
            TempData = self.loadSFG(parameters['Background']['Files'][bgFile])
            xlist.append(TempData.index.values)
            ylist.append(np.transpose(TempData.values)[0])
        background = np.array((np.average(xlist,axis=0),np.average(ylist,axis=0)))
        
        return background
    
    def loadSFG(self,Parameters) :
    
        FolderPath = Parameters['FolderPath']
        FileName = Parameters['FileName']

        if FileName.endswith('.ibw') :
            d = binarywave.load(FolderPath + '/' + FileName)
            y = np.transpose(d['wave']['wData'])
            Start = d['wave']['wave_header']['sfB']
            Delta = d['wave']['wave_header']['sfA']
            x = np.arange(Start[0],Start[0]+y.shape[1]*Delta[0]-Delta[0]/2,Delta[0])
            z = np.arange(Start[1],Start[1]+y.shape[0]*Delta[1]-Delta[1]/2,Delta[1])
            print('Igor binary data loaded')
        elif FileName.endswith('.itx') :
            y = np.loadtxt(FolderPath + '/' + FileName,comments =list(string.ascii_uppercase))
            y = np.transpose(y)
            with open(FolderPath + '/' + FileName) as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if 'SetScale/P' in row[0]:
                        SetScale = row[0]
            xScale = re.findall(r' x (.*?),"",', SetScale)
            xScale = xScale[0].split(',')
            zScale = re.findall(r' y (.*?),"",', SetScale)
            zScale = zScale[0].split(',')
            Start = [float(xScale[0]),float(zScale[0])]
            Delta = [float(xScale[1]),float(zScale[1])]
            x = np.arange(Start[0],Start[0]+y.shape[1]*Delta[0]-Delta[0]/2,Delta[0])
            z = np.arange(Start[1],Start[1]+y.shape[0]*Delta[1]-Delta[1]/2,Delta[1])
            print('Igor text data loaded')
        elif FileName.endswith('sif') :
            FileData = sif.xr_open(FolderPath + '/' + FileName)
            y = FileData.values[:,0,:]
            x = [i for i in range(len(np.transpose(y)))]
            z = [i+1 for i in range(len(y))]
            
            try :
                FileData.attrs['WavelengthCalibration0']
                FileData.attrs['WavelengthCalibration1']
                FileData.attrs['WavelengthCalibration2']
                FileData.attrs['WavelengthCalibration3']
            except :
                print('Warning: Wavelength calibration not found')
            else :
                c0 = FileData.attrs['WavelengthCalibration0']
                c1 = FileData.attrs['WavelengthCalibration1']
                c2 = FileData.attrs['WavelengthCalibration2']
                c3 = FileData.attrs['WavelengthCalibration3']
                for i in x :
                    x[i] = c0 + c1*i + c2*1**2 + c3*i**3
                x = np.array(x)
                x = 1e7 / x - 12500
            
            try :
                Frame = Parameters['Heating']['Frame']
                Temperature = Parameters['Heating']['Temperature']
            except :
                print('Warning: Temperature data not found')
            else :
                FitModel = QuadraticModel()
                ModelParameters = FitModel.make_params()
                FitResults = FitModel.fit(Temperature, ModelParameters,x=Frame)
                idx = np.array(z)
                z = FitResults.eval(x=idx)
                z = np.round(z,1)
        
        Data = df(np.transpose(y),index=x,columns=z)
            
        return Data
    
    def removeEmptyDataSets(self,Data,Threshold) :
    
        Index = list()
        for i in Data.columns :
            if np.mean(Data[i]) < Threshold :
                Index.append(i)
        for i in Index :
            del Data[i]
        
        return Data

    def trimData(self,Data,Min,Max) :
        
        Mask = np.all([Data.index.values>Min,Data.index.values<Max],axis=0)
        Data = Data[Mask]
        
        return Data

    def reduceResolution(self,Data,Resolution=1) :
        
        Counter = 0
        ReducedData = df()
        if Resolution > 1 :
            for i in range(int(len(Data.columns.values)/Resolution)) :
                Column = round(np.mean(Data.columns[Counter:Counter+Resolution]),1)
                ReducedData[Column] = Data[Data.columns[Counter:Counter+Resolution]].mean(axis=1)
                Counter = Counter + Resolution
            Data = ReducedData

        return Data


class analysisTools :
    
    def __init__(self,data,par,background='') :
        
        self.data = data
        self.par = par
        self.background = background
    
    def fit(self,data,fitParameters,**kwargs) :
        
        dt = dataTools()

        def BuiltInModels(fitParameters) :
        
            ModelString = list()
            for key in fitParameters['Models'] :
                ModelString.append((key,fitParameters['Models'][key]['model']))
            
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

            return fitModel
            
        def SFGModel(fitParameters) :
            
            ModelString = list()
            for key in fitParameters['Models'] :
                ModelString.append([key])
            
            if len(ModelString) == 2 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            elif len(ModelString) == 3 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                                Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            elif len(ModelString) == 4 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                                Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                                Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                    Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            elif len(ModelString) == 5 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                                Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                                Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                                Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                    Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                    Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            elif len(ModelString) == 6 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                                Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                                Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                                Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                                Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                    Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                    Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                    Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            elif len(ModelString) == 7 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                                Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                                Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                                Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                                Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                                Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                    Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                    Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                    Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                    Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            elif len(ModelString) == 8 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                                Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                                Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                                Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                                Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                                Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                                Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                    Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                    Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                    Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                    Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                    Peaks+= Peak7_amp*(cmath.exp(Peak7_phi*1j)/(x-Peak7_omega+Peak7_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            elif len(ModelString) == 9 :
                def SFGFunction(x,NonRes_amp,
                                Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                                Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                                Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                                Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                                Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                                Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                                Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma,
                                Peak8_amp,Peak8_phi,Peak8_omega,Peak8_gamma) :
                    Peaks = NonRes_amp
                    Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                    Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                    Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                    Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                    Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                    Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                    Peaks+= Peak8_amp*(cmath.exp(Peak8_phi*1j)/(x-Peak8_omega+Peak8_gamma*1j))
                    return np.real(Peaks*np.conjugate(Peaks))
            
            fitModel = Model(SFGFunction)

            return fitModel
        
        def setParameters(fitParameters, fitModel, Value=None) :

            modelParameters = fitModel.make_params()
            
            ParameterList = ['amp','phi','omega','gamma','center','sigma','c']
            Parameters = {'Standard': fitParameters['Models']}

            if 'Cases' in fitParameters and Value != None:
                for Case in fitParameters['Cases'] :
                    if Value >= min(fitParameters['Cases'][Case]['zRange']) and Value <= max(fitParameters['Cases'][Case]['zRange']) :
                        Parameters[Case] = fitParameters['Cases'][Case]
            
            for Dictionary in Parameters :
                for Peak in Parameters[Dictionary] :
                    for Parameter in Parameters[Dictionary][Peak] :
                        if Parameter in ParameterList :
                            for Key in Parameters[Dictionary][Peak][Parameter] :
                                if Key != 'set' :
                                    exec('modelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                                else :
                                    exec('modelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))

            return modelParameters
        
        try :
            fitParameters['ModelType']
            fitParameters['Models']
        except:
            fitParameters['ModelType'] = 'None'
            fitParameters['Models'] = ''
        else :
            if fitParameters['ModelType'] == 'BuiltIn' :
                fitModel = BuiltInModels(fitParameters)
            if fitParameters['ModelType'] == 'SFG' :
                fitModel = SFGModel(fitParameters)

        if 'xRange' in fitParameters :
            data = dt.trimData(data, fitParameters['xRange'][0], fitParameters['xRange'][1])
        x = data.index.values
        if 'fit_x' in kwargs :
            for kwarg in kwargs :
                if kwarg == 'fit_x':
                    fit_x = kwargs[kwarg]
        else :
            fit_x = x
        
        for idx, Column in enumerate(data) :
            
            modelParameters = setParameters(fitParameters, fitModel)
            
            y = data[Column].values
            fitResults = fitModel.fit(y, modelParameters, x=x, nan_policy='omit')
            fit_comps = fitResults.eval_components(fitResults.params, x=fit_x)
            fit_y = fitResults.eval(x=fit_x)
            ParameterNames = [i for i in fitResults.params.keys()]
            if idx == 0 :
                fits = df(index=fit_x,columns=data.columns.values)
                fitsParameters = df(index=modelParameters.keys(),columns=data.columns.values)
                fitsResults = list()
                fitsComponents = list()
            for Parameter in (ParameterNames) :
                fitsParameters.at[Parameter,Column] = fitResults.params[Parameter].value
            fits[Column] = fit_y
            fitsResults.append(fitResults)
            fitsComponents.append(fit_comps)
            
            sys.stdout.write(("\rFitting %i out of "+str(data.shape[1])) % (idx+1))
            sys.stdout.flush()
        
        return fits, fitsComponents, fitsParameters, fitsResults
        
    def fitData(self) :

        data = self.data
        par = self.par
        background = self.background

        dt = dataTools()
        
        print('Data: '+par['FileName'])
        print('Description: '+par['Description'])
        
        ##### Prepare Data #####
        
        if background != '' :
            for column in data :
                data[column] = data[column]-background[1]
        if 'Level' in par['Background'] :
            xRange = par['Background']['Level']['xRange']
            Mean = data
            Mean = Mean[Mean.index>min(xRange)]
            Mean = Mean[Mean.index<max(xRange)]
            Mean = Mean.mean().mean()
            data -= Mean
        
        backgroundT = par['Background']['zRange']
        DataNames = list()
        for i in data.columns :
            if i >= min(backgroundT) and i <= max(backgroundT) :
                DataNames.append(i)
        nonres = df(data[DataNames].mean(axis=1),columns=['Data'])
        
        data = dt.reduceResolution(data,par['Resolution'])
        
        ##### Fit Data #####

        try :
            par['Background']['Models']
        except :
            data2Fit = data.divide(nonres['Data'],axis=0)
        else :
            print('Fitting Background')
            fitNonres = self.fit(nonres,par['Background'])
            fitNonres[3][0].plot_fit(xlabel='Wavenumber (cm$^{-1}$)',ylabel='Intensity (au)',title='Non-Resonant Background')
            nonres['Fit'] = fitNonres[0]
            data2Fit = data.divide(nonres['Fit'],axis=0)

        if 'xRange' in par['Fit'] :
            data2Fit = dt.trimData(data2Fit,par['Fit']['xRange'][0],par['Fit']['xRange'][1])
        
        if 'zRange' in par['Fit'] :
            T_mask = []
            T_mask.append(data.columns<=max(par['Fit']['zRange']))
            T_mask.append(data.columns>=min(par['Fit']['zRange']))
            T_mask = np.all(T_mask, axis=0)
            data2Fit = data2Fit.T[T_mask].T
        
        fits, fitsComponents, fitsParameters, fitsResults = self.fit(data2Fit,par['Fit'])
        
        if 'Fit' in nonres :
            fitsData = fits.multiply(nonres['Fit'],axis=0)
        else :
            fitsData = fits.multiply(nonres['Data'],axis=0)
        
        print('\n'+100*'_')
        
        ##### Peak Assignments #####
        
        PeakList = list()
        AssignmentList = list()
        for Peak in par['Fit']['Models'] :
            PeakList.append(Peak)
            if 'assignment' in par['Fit']['Models'][Peak] :
                AssignmentList.append(par['Fit']['Models'][Peak]['assignment'])
            else :
                AssignmentList.append(Peak)
        FitsAssignments = df(AssignmentList,index=PeakList,columns=['Assignment'])
        
        #### Show Fits & Data #####
        
        if 'ShowFits' in par['Fit'] :
            ShowFits = par['Fit']['ShowFits']
        else :
            ShowFits = True

        if ShowFits :
            for Column in data2Fit :

                plt.figure(figsize = [12,4])

                plt.subplot(1, 2, 1)
                plt.plot(data.index, data[Column],'k.', label='Data')
                plt.plot(fitsData.index, fitsData[Column], 'r-', label='Fit')
                plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
                plt.title('Temperature: '+str(Column)+' K')

                plt.subplot(1, 2, 2)
                plt.plot(data2Fit.index, data2Fit[Column],'k.', label='Data')
                plt.plot(fits.index, fits[Column], 'r-', label='Fit')
                plt.xlabel('WaveNumber (cm$^{-1}$)'), plt.ylabel('Intensity (au)')
                if 'xRange' in par['Fit'] :
                    plt.xlim(par['Fit']['xRange'][0],par['Fit']['xRange'][1])

                plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
                plt.show()

                Peaks = list()
                for Parameter in fitsParameters.index :
                    Name = Parameter.split('_')[0]
                    if Name not in Peaks :
                        Peaks.append(Name)

                string = ''
                for Peak in Peaks :
                    if 'assignment' in par['Fit']['Models'][Peak] :
                        string += par['Fit']['Models'][Peak]['assignment'] + ' | '
                    else :
                        string += Peak + ' | '
                    for Parameter in fitsParameters.index :
                        if Peak == Parameter.split('_')[0] : 
                            string += Parameter.split('_')[1] + ': ' + str(round(fitsParameters[Column][Parameter],2))
                            string += ', '
                    string = string[:-2] + '\n'
                print(string)
                print(100*'_')
        
        fitsParameters = fitsParameters.T
        fitsParameters = fitsParameters[np.concatenate((fitsParameters.columns.values[1:],fitsParameters.columns.values[0:1]))]
        
        # Plot 2D Data & Fits
        
        plt.figure(figsize = [8,12])
        
        plt.subplot(2, 1, 1)
        x = data.index.values
        y = data.columns.values
        z = np.transpose(data.values)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Data: '+par['FileName'], fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.subplot(2, 1, 2)
        x = fitsData.index.values
        y = fitsData.columns.values
        z = np.transpose(fitsData.values)
        plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        plt.ylabel('Temperature (K)', fontsize=16)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plt.title('Fits: '+par['FileName'], fontsize=16)
        pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        
        plt.show()
        
        # Plot Trends
        
        UniqueParameters = []
        [UniqueParameters.append(x.split('_')[1]) for x in fitsParameters.columns if x.split('_')[1] not in UniqueParameters][0]
        for uniqueParameter in UniqueParameters :
            fig = go.Figure()
            for parameter in fitsParameters :
                if uniqueParameter in parameter :
                    Name = parameter.split('_')[0]
                    if 'assignment' in par['Fit']['Models'][Name] :
                        Name = par['Fit']['Models'][Name]['assignment']
                    fig.add_trace(go.Scatter(x=fitsParameters.index,y=fitsParameters[parameter],name=Name,mode='lines+markers'))
            fig.update_layout(xaxis_title='Temperature (K)',yaxis_title=uniqueParameter,legend_title='',width=800,height=400)
            fig.show()
        
        ##### Widgets #####

        Results = {}
        Results['fits'] = fits
        Results['fitsData'] = fitsData
        Results['fitsComponents'] = fitsComponents
        Results['fitsParameters'] = fitsParameters
        Results['fitsResults'] = fitsResults
        Results['FitsAssignments'] = FitsAssignments
        Results['nonres'] = nonres

        return Results

class UI :
    
    def __init__(self) :

        dt = dataTools()
        
        self.cwd = Path(os.getcwd())

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        self.settingsFile = settingsFile

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
                with open(self.settingsFile, 'r') as f :
                    data = yaml.safe_load(f)
                data['folders']['data'] = dataFolder.value
                self.folders['data'] = dataFolder.value
                with open(self.settingsFile, 'w') as f:
                    yaml.dump(data, f)
                print('cool')
        dataFolder.observe(changeDataFolder, names='value')

        def go_to_address(address):
            address = Path(address)
            if address.is_dir():
                currentFolder_field.value = str(address)
                selectFolder.unobserve(selecting, names='value')
                selectFolder.options = self.get_folder_contents(folder=address)[0]
                selectFolder.observe(selecting, names='value')
                selectFolder.value = None
                selectFile.options = self.get_folder_contents(folder=address)[1]

        def newaddress(value):
            go_to_address(currentFolder_field.value)
        currentFolder_field = ipw.Text(value=str(self.cwd),
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder')
        currentFolder_field.on_submit(newaddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(currentFolder_field.value)
                newpath = path / value['new']
                if newpath.is_dir():
                    go_to_address(newpath)
                elif newpath.is_file():
                    #some other condition
                    pass
        
        selectFolder = ipw.Select(
            options=self.get_folder_contents(self.cwd)[0],
            rows=5,
            value=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Subfolders')
        selectFolder.observe(selecting, names='value')
        
        selectFile = ipw.Select(
            options=self.get_folder_contents(self.cwd)[1],
            rows=10,
            values=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Files')

        def parent(value):
            new = Path(currentFolder_field.value).parent
            go_to_address(new)
        up_button = ipw.Button(description='Up',layout=Layout(width='10%'))
        up_button.on_click(parent)

        def ShowData_Clicked(b) :
            with out :
                clear_output(True)
                data, parameters = dt.loadData(selectFile.value,dataFolder.value)
                self.data = data
                self.parameters = parameters
                plt.figure(figsize = [8,6])
                x = data.index.values
                y = data.columns.values
                z = np.transpose(data.values)
                plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
                plt.ylabel('Temperature (K)', fontsize=16)
                plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
                plt.title(selectFile.value, fontsize=16)
                pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
                plt.show()
        ShowData = ipw.Button(description="Show Data")
        ShowData.on_click(ShowData_Clicked)
        
        def FitData_Clicked(b) :
            with out :
                clear_output(True)
                data, parameters = dt.loadData(selectFile.value,dataFolder.value)
                self.data = data
                self.parameters = parameters
                if 'Files' in parameters['Background'] :
                    background = dt.loadBackground(selectFile.value,dataFolder.value)
                    at = analysisTools(data,parameters,background)
                else :
                    at = analysisTools(data,parameters)
                self.fits = at.fitData()
                def CopyData_Clicked(b) :
                    data.to_clipboard()
                CopyData = ipw.Button(description="Copy Data")
                CopyData.on_click(CopyData_Clicked)
                def CopyFits_Clicked(b) :
                    self.fits['fits'].to_clipboard()
                CopyFits = ipw.Button(description="Copy Fits")
                CopyFits.on_click(CopyFits_Clicked)
                def CopyParameters_Clicked(b) :
                    self.fits['fitsParameters'].to_clipboard()
                CopyParameters = ipw.Button(description="Copy Parameters")
                CopyParameters.on_click(CopyParameters_Clicked)
                display(ipw.Box([CopyData,CopyFits,CopyParameters]))
        FitData = ipw.Button(description="Fit Data")
        FitData.on_click(FitData_Clicked)
        
        display(ipw.HBox([dataFolder]))
        display(ipw.HBox([currentFolder_field]))
        display(ipw.HBox([selectFolder,up_button]))
        display(ipw.HBox([selectFile]))
        display(ipw.HBox([ShowData,FitData]))

        display(out)
        display(anout)
    
    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)
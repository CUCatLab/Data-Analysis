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
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import cmath
import re
import tempfile


# Plotly settings
import plotly.graph_objects as go
import plotly.io as pio
# import plotly.express as px
pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.9,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 1,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)'
))

settingsFile = 'tools/settings.yaml'

class dataTools :
    
    def __init__(self) :

        pass

    def loadData(self, tpdFile, dataFolder) :

        def loadTpd_SU(parameters) :
            
            folderPath = parameters['folderPath']
            fileName = parameters['FileName']
            Masses = parameters['Masses']

            with open(folderPath + '/' + fileName, mode='rb') as file:
                fileContent = file.read()

            NumChan = len(Masses) + 1
            dataLength = int((len(fileContent)-5)/(46*NumChan))
            data = np.zeros((int(1+NumChan),dataLength))

            for i in range(len (data)) :
                for j in range(len (data[0])) :
                    if i == 0 :
                        index = int(31+j*46*NumChan)
                        data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]/1000
                    else :
                        index = int(43+j*46*NumChan + (i-1)*46)
                        data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]
            
            Header = list()
            Header.append('Time (s)')
            if 'TChan' in parameters :
                TChan = parameters['TChan']
            else :
                TChan = 0
            for idx in range(NumChan) :
                if idx == TChan :
                    Header.append('Temperature (K)')
                else :
                    Header.append(str(Masses[idx-1]))
            
            data = df(np.transpose (data),columns=Header)
            if 'Temperature (K)' in data:
                data = data.set_index('Temperature (K)')
            else:
                data = data.set_index('Temperature')
            if 'TScale' in parameters :
                data.index = data.index * parameters['TScale']
            
            parameters['HeatingRate'] = np.mean(np.diff (data.index)/np.diff (data['Time (s)']))
                
            return data, parameters

        def loadTpd(parameters) :
            
            folderPath = parameters['folderPath']
            fileName = parameters['FileName']
            
            if fileName.endswith('.yml') or fileName.endswith('.yaml'):
                with open(folderPath + '/' + fileName, mode='rb') as file:
                    fileContent = file.read()
                allData = yaml.safe_load(fileContent)
                data = dict()
                for key in allData:
                    try:
                        data[float(key)] = allData[key]
                    except ValueError:
                        data[key] = allData[key]
            else:
                with open(folderPath + '/' + fileName, mode='rb') as file:
                    data = pd.read_csv(file)
            
            if 'Parameters' in data:
                del data['Parameters']
            if 'Pressure (Torr)' in data:
                del data['Pressure (Torr)']
            if 'Pressure' in data:
                del data['Pressure']
            if 'Mass 15.0' in data:
                data['CH3'] = data['Mass 15.0']
                del data['Mass 15.0']
            if 'Mass 17.0' in data:
                data['OH'] = data['Mass 17.0']
                del data['Mass 17.0']
            if 'Mass 18.0' in data:
                data['H2O'] = data['Mass 18.0']
                del data['Mass 18.0']
            if 'Mass 2.0' in data:
                data['H2'] = data['Mass 2.0']
                del data['Mass 2.0']
            if 'Mass 28.0' in data:
                data['CO'] = data['Mass 28.0']
                del data['Mass 28.0']
            if 'Mass 32.0' in data:
                data['O2'] = data['Mass 32.0']
                del data['Mass 32.0']
            if 'Mass 44.0' in data:
                data['CO2'] = data['Mass 44.0']
                del data['Mass 44.0']
                
            if type (data) == dict:
                data = df.from_dict (data)
            if 'Temperature (K)' in data:
                data = data.set_index('Temperature (K)')
            else:
                data = data.set_index('Temperature')
            
            if 'Time (s)' in data:
                parameters['HeatingRate'] = np.mean(np.diff (data.index)/np.diff (data['Time (s)']))
            else:
                parameters['HeatingRate'] = np.mean(np.diff (data.index)/np.diff (data['Time']))
            
            return data, parameters
        
        if tpdFile[1].endswith('.yaml') or tpdFile[1].endswith('.yml') :
            with open(tpdFile[0]+'/'+tpdFile[1], 'r') as stream :
                parameters = yaml.safe_load(stream)
            if 'FileName' not in parameters :
                parameters['FileName'] = tpdFile
            
            if 'folderPath' not in parameters :
                if os.path.exists(dataFolder+'/'+parameters['FileName']) :
                    parameters['folderPath'] = dataFolder
                else :
                    if 'TPD' in parameters['FileName'] :
                        date = re.split(r'TPD|_',parameters['FileName'])[1]
                    if 'tpd' in parameters['FileName'] :
                        date = re.split(r'tpd|_',parameters['FileName'])[1]
                    if len(date) == 6 :
                        date = '20'+date
                    parameters['folderPath'] = dataFolder+'/'+'20'+date[2:4]+'/'+'20'+date[2:4]+'.'+date[4:6]+'.'+date[6:8]
        elif tpdFile[1].endswith('.csv') :
            parameters = {}
            parameters['folderPath'] = tpdFile[0]
            parameters['FileName'] = tpdFile[1]

        print('File: '+parameters['folderPath']+'/'+parameters['FileName'])
        try :
            data, parameters = loadTpd(parameters)
        except :
            data, parameters = loadTpd_SU(parameters)
        
        return data, parameters
    
    def plotData(
            self,
            data,
            filename=''
        ):

        traces = [
            c for c in data.columns
            if c not in [
                'Time',
                'Time (s)'
            ]
        ]

        default_labels = '\n'.join(
            f'{trace}={trace}'
            for trace in traces
        )

        title_default = (
            Path(filename).stem
            if filename else
            'TPD Analysis'
        )

        TraceSelector = ipw.SelectMultiple(
            options=traces,
            value=tuple(traces),
            rows=min(10, len(traces)),
            description='Traces'
        )

        BaselineCorrection = ipw.Checkbox(
            value=False,
            description='Baseline Correct'
        )

        BaselineMin = ipw.FloatText(
            value=600,
            description='Baseline Min'
        )

        BaselineMax = ipw.FloatText(
            value=700,
            description='Baseline Max'
        )

        ShowBaselineRegion = ipw.Checkbox(
            value=False,
            description='Show Region'
        )

        Title = ipw.Text(
            value=title_default,
            description='Title'
        )

        XLabel = ipw.Text(
            value='Temperature (K)',
            description='X Label'
        )

        YLabel = ipw.Text(
            value='Counts (au)',
            description='Y Label'
        )

        Labels = ipw.Textarea(
            value=default_labels,
            description='Labels',
            layout=ipw.Layout(
                width='700px',
                height='250px'
            )
        )

        UpdateButton = ipw.Button(
            description='Update Plot',
            button_style='success'
        )

        ShowOptions = ipw.Button(
            description='Plot Options',
            button_style='info'
        )

        output = ipw.Output()

        def parse_labels():

            mapping = {}

            for line in Labels.value.splitlines():

                if '=' in line:

                    old, new = line.split(
                        '=',
                        1
                    )

                    mapping[
                        old.strip()
                    ] = new.strip()

            return mapping

        def update_plot(b=None):

            with output:

                clear_output(wait=True)

                fig = go.Figure()

                label_map = parse_labels()

                for trace in TraceSelector.value:

                    y = data[
                        trace
                    ].astype(float).copy()

                    if BaselineCorrection.value:

                        baseline_mask = (
                            (data.index >= BaselineMin.value)
                            &
                            (data.index <= BaselineMax.value)
                        )

                        if np.sum(baseline_mask) > 0:

                            baseline = np.mean(
                                y[
                                    baseline_mask
                                ]
                            )

                            y = y - baseline

                    fig.add_trace(

                        go.Scatter(

                            x=data.index,

                            y=y,

                            mode='lines',

                            name=label_map.get(
                                trace,
                                trace
                            ),

                            line=dict(
                                width=3
                            )
                        )
                    )

                if (
                    BaselineCorrection.value
                    and
                    ShowBaselineRegion.value
                ):

                    fig.add_vrect(
                        x0=BaselineMin.value,
                        x1=BaselineMax.value,
                        fillcolor='lightgray',
                        opacity=0.15,
                        layer='below',
                        line_width=0
                    )

                fig.add_hline(
                    y=0,
                    line_dash='dash',
                    line_color='black'
                )

                fig.update_layout(

                    template='simple_white',

                    height=900,

                    title=dict(
                        text=Title.value,
                        font=dict(
                            size=34
                        )
                    ),

                    xaxis_title=
                    XLabel.value,

                    yaxis_title=
                    YLabel.value,

                    font=dict(
                        size=28
                    ),

                    legend=dict(

                        x=0.98,
                        y=0.98,

                        xanchor='right',
                        yanchor='top',

                        bgcolor='rgba(0,0,0,0)',

                        borderwidth=0,

                        font=dict(
                            size=28
                        )
                    ),

                    margin=dict(
                        l=100,
                        r=50,
                        b=90,
                        t=130
                    )
                )

                fig.update_xaxes(

                    showline=True,
                    linewidth=2,
                    mirror=True,
                    ticks='inside',

                    tickfont=dict(
                        size=28
                    ),

                    title_font=dict(
                        size=28
                    )
                )

                fig.update_yaxes(

                    showline=True,
                    linewidth=2,
                    mirror=True,
                    ticks='inside',

                    tickfont=dict(
                        size=28
                    ),

                    title_font=dict(
                        size=28
                    )
                )

                fig.show()

        UpdateButton.on_click(
            update_plot
        )

        OptionsBox = ipw.VBox([

            ipw.HBox([

                TraceSelector,

                ipw.VBox([

                    BaselineCorrection,
                    BaselineMin,
                    BaselineMax,
                    ShowBaselineRegion

                ])

            ]),

            Title,
            XLabel,
            YLabel,

            Labels,

            UpdateButton

        ])

        OptionsBox.layout.display = 'none'

        def ToggleOptions(b):

            if (
                OptionsBox.layout.display
                == 'none'
            ):

                OptionsBox.layout.display = ''

                ShowOptions.description = (
                    'Hide Plot Options'
                )

            else:

                OptionsBox.layout.display = 'none'

                ShowOptions.description = (
                    'Plot Options'
                )

        ShowOptions.on_click(
            ToggleOptions
        )

        display(
            ipw.VBox([

                ShowOptions,

                OptionsBox,

                output

            ])
        )

        update_plot()


class analysisTools :
    
    def __init__(self) :

        pass

    def simulateData(self, data, parameters, Rate) :
        
        if 'Simulations' in parameters :

            # Initial parameters
            kB = 8.617e-5                 # eV/K
            T0 = 100                      # K
            
            Temperature, deltaT = np.linspace(min (data.index),max (data.index),1001,retstep =True)
            Time = Temperature / Rate
            deltat = deltaT / Rate
            Size = len(Temperature)
            
            Traces = df(index=Temperature)
            Coverages = df(index=Temperature)
            
            # Calculate traces
            for Mass in parameters['Simulations'] :
                Trace = np.zeros((Size))
                Coverage = np.zeros((Size))
                for idx, Peak in enumerate(parameters['Simulations'][Mass]) :
                    PeakParameters = parameters['Simulations'][Mass][Peak]
                    Offset = PeakParameters['Offset']
                    Scaling = PeakParameters['Scaling']
                    Ni = PeakParameters['Coverage']
                    Ea = PeakParameters['Barrier']
                    nu = PeakParameters['Prefactor']
                    n = PeakParameters['Order']
                    
                    PeakTrace = np.zeros((Size))
                    PeakCoverage = np.zeros((Size))
                    IntRate = 0
                    for idx, T in enumerate(Temperature) :
                        PeakTrace[idx] = nu*(Ni - IntRate)**n * np.exp(-Ea/(kB*T))
                        IntRate += PeakTrace[idx] * deltat
                        PeakCoverage[idx] = Ni - IntRate
                        if IntRate >= Ni :
                            IntRate = Ni
                            PeakCoverage[idx] = 0
                        if PeakCoverage[idx] < 0 or PeakCoverage[idx] > Ni :
                            PeakCoverage[idx] = 0
                            PeakTrace[idx] = 0
                    Trace += PeakTrace * Scaling + Offset
                    Coverage += PeakCoverage
                
                Traces[Mass] = Trace
                Coverages[Mass] = Coverage
                
        return Traces, Coverages


class UI:

    def __init__(self):

        dt = dataTools()
        at = analysisTools()

        try:
            with open(settingsFile, 'r') as stream:
                settings = yaml.safe_load(stream) or {}
        except:
            settings = {
                'folders': {
                    'data': str(Path.cwd())
                }
            }

        savedFolder = settings.get(
            'folders',
            {}
        ).get(
            'data',
            str(Path.cwd())
        )

        if Path(savedFolder).is_dir():
            self.cwd = savedFolder
        else:
            self.cwd = str(Path.home())

        self.loadedFiles = {}
        self.loadedData = {}
        self.loadedParameters = {}

        out = ipw.Output()

        # ----------------------------------
        # Folder selection
        # ----------------------------------

        currentFolderLabel = ipw.Label(
            value=self.cwd,
            layout=Layout(width='85%')
        )

        SelectFolder = ipw.Text(
            value=self.cwd,
            description='Folder:',
            layout=Layout(width='90%')
        )

        SelectFolderBox = ipw.VBox(
            [SelectFolder]
        )

        SelectFolderBox.layout.display = 'none'

        SelectFolderButton = ipw.Button(
            description='Select Folder',
            button_style='primary',
            layout=Layout(width='15%')
        )

        def SelectFolderButton_Clicked(b):

            if SelectFolderBox.layout.display == 'none':
                SelectFolderBox.layout.display = ''
            else:
                SelectFolderBox.layout.display = 'none'

        SelectFolderButton.on_click(
            SelectFolderButton_Clicked
        )

        def FolderChanged(change):

            folder = Path(
                SelectFolder.value
            )

            if folder.is_dir():

                currentFolderLabel.value = str(folder)

                settings['folders'][
                    'data'
                ] = str(folder)

                with open(
                    settingsFile,
                    'w'
                ) as f:
                    yaml.dump(
                        settings,
                        f
                    )

            else:

                print(
                    f'Folder does not exist: '
                    f'{folder}'
                )

        SelectFolder.observe(
            FolderChanged,
            names='value'
        )

        # ----------------------------------
        # File Upload
        # ----------------------------------

        DataFileUpload = ipw.FileUpload(
            accept='.csv,.yaml,.yml',
            multiple=True,
            description='Select Analysis Files',
            layout=Layout(width='220px')
        )

        LoadedDataFile = ipw.Dropdown(
            options=[],
            value=None,
            description='Active File:',
            layout=Layout(width='600px'),
            style={
                'description_width':
                '100px'
            }
        )

        ClearLoadedFiles = ipw.Button(
            description='Clear Files',
            layout=Layout(width='120px')
        )

        UploadStatus = ipw.HTML(
            value='<i>No files loaded.</i>'
        )

        # ----------------------------------
        # Helpers
        # ----------------------------------

        def _uploaded_entries(value):

            if isinstance(value, dict):

                return [
                    (
                        name,
                        item.get(
                            'content',
                            b''
                        )
                    )
                    for name, item
                    in value.items()
                ]

            entries = []

            for item in (value or []):

                entries.append(
                    (
                        item.get('name'),
                        item.get(
                            'content',
                            b''
                        )
                    )
                )

            return entries

        def _activate_file(name):

            if (
                name is None or
                name not in self.loadedData
            ):
                return

            self.data = self.loadedData[
                name
            ]

            self.parameters = (
                self.loadedParameters[
                    name
                ]
            )

            self.filename = name

        # ----------------------------------
        # Upload callback
        # ----------------------------------

        def FilesUploaded(change):

            entries = _uploaded_entries(
                change['new']
            )

            if len(entries) == 0:
                return

            errors = []

            for (
                filename,
                content
            ) in entries:

                try:

                    raw = (
                        content.tobytes()
                        if hasattr(
                            content,
                            'tobytes'
                        )
                        else bytes(
                            content
                        )
                    )

                    with tempfile.TemporaryDirectory() as tmpdir:

                        path = (
                            Path(tmpdir)
                            / filename
                        )

                        path.write_bytes(
                            raw
                        )

                        data, parameters = (
                            dt.loadData(
                                [
                                    str(
                                        path.parent
                                    ),
                                    path.name
                                ],
                                currentFolderLabel.value
                            )
                        )

                    self.loadedData[
                        filename
                    ] = data

                    self.loadedParameters[
                        filename
                    ] = parameters

                except Exception as e:

                    errors.append(
                        f'{filename}: {e}'
                    )

            names = sorted(
                self.loadedData.keys()
            )

            LoadedDataFile.options = names

            if len(names):

                LoadedDataFile.value = (
                    names[-1]
                )

                UploadStatus.value = (
                    f'<b>{len(names)}</b> '
                    'file(s) loaded'
                )

            if errors:

                UploadStatus.value += (
                    '<br>'
                    '<span style="color:red">'
                    + '<br>'.join(errors)
                    + '</span>'
                )

        DataFileUpload.observe(
            FilesUploaded,
            names='value'
        )

        # ----------------------------------
        # Active file changed
        # ----------------------------------

        def ActiveFileChanged(change):

            if change['new']:

                _activate_file(
                    change['new']
                )

        LoadedDataFile.observe(
            ActiveFileChanged,
            names='value'
        )

        # ----------------------------------
        # Clear files
        # ----------------------------------

        def ClearFilesClicked(b):

            self.loadedData.clear()

            self.loadedParameters.clear()

            LoadedDataFile.options = []

            LoadedDataFile.value = None

            UploadStatus.value = (
                '<i>No files loaded.</i>'
            )

        ClearLoadedFiles.on_click(
            ClearFilesClicked
        )

        # ----------------------------------
        # Show Data
        # ----------------------------------

        def ShowData_Clicked(b):

            with out:

                clear_output(True)

                if not hasattr(
                    self,
                    'data'
                ):

                    print(
                        'No file loaded.'
                    )

                    return

                dt.plotData(
                    self.data,
                    self.filename
                )

                display(
                    Data2Clipboard
                )

        ShowData = ipw.Button(
            description='Show Data',
            button_style='success'
        )

        ShowData.on_click(
            ShowData_Clicked
        )

        # ----------------------------------
        # Copy Data
        # ----------------------------------

        def Data2Clipboard_Clicked(b):

            if hasattr(
                self,
                'data'
            ):

                self.data.to_clipboard()

                print(
                    'Copied to clipboard.'
                )

        Data2Clipboard = ipw.Button(
            description='Copy Data'
        )

        Data2Clipboard.on_click(
            Data2Clipboard_Clicked
        )

        # ----------------------------------
        # Simulate
        # ----------------------------------

        def SimulateTrace_Clicked(b):

            with out:

                clear_output(True)

                if not hasattr(
                    self,
                    'data'
                ):

                    print(
                        'No file loaded.'
                    )

                    return

                simulatedData, (
                    SimulatedCoverages
                ) = at.simulateData(
                    self.data,
                    self.parameters,
                    self.parameters[
                        'HeatingRate'
                    ]
                )

                self.simulatedData = (
                    simulatedData
                )

                allData = pd.concat(
                    [
                        allData,
                        simulatedData
                    ]
                )

                dt.plotData(
                    self.data,
                    self.filename
                )

        SimulateTrace = ipw.Button(
            description='Simulate Traces'
        )

        SimulateTrace.on_click(
            SimulateTrace_Clicked
        )

        # ----------------------------------
        # Display
        # ----------------------------------

        display(
            ipw.HBox([
                SelectFolderButton,
                currentFolderLabel
            ])
        )

        display(
            SelectFolderBox
        )

        display(
            ipw.HBox([
                DataFileUpload,
                LoadedDataFile,
                ClearLoadedFiles
            ])
        )

        display(
            UploadStatus
        )

        display(
            ipw.HBox([
                ShowData,
                SimulateTrace
            ])
        )

        display(out)
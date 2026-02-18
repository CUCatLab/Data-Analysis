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
# New for .gxz debug + parsing
import gzip
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import gzip, re, xml.etree.ElementTree as ET
from itertools import combinations
from typing import List, Tuple, Optional

settingsFile = 'tools/settings.yaml'


class dataTools :

    def __init__(self) :
        
        pass

    def loadText(self,folder,file) :
        
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

    def loadGXZ(self, folder: str, file: str) -> pd.DataFrame:
        
        def _to_floats(txt: Optional[str]) -> List[float]:
            _NUM_SPLIT = re.compile(r"[,\s;]+")
            """Parse a whitespace/comma/semicolon separated string into floats."""
            if not txt:
                return []
            vals: List[float] = []
            for tok in _NUM_SPLIT.split(txt.strip()):
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except ValueError:
                    try:
                        vals.append(float(tok.replace(',', '.')))
                    except Exception:
                        pass
            return vals

        def _local(tag: str) -> str:
            """Strip XML namespace."""
            return tag.split('}', 1)[-1] if '}' in tag else tag

        def _elem_text(e: Optional[ET.Element]) -> str:
            
            return "" if e is None else (e.text or "")

        def _is_monotonic_increasing(xs: List[float]) -> bool:
            
            return all(xs[i] < xs[i+1] for i in range(len(xs)-1))

        def _collect_parent_map(root: ET.Element) -> dict:
            """Build {child: parent} so we can walk upward with stdlib ElementTree."""
            parent_of = {}
            stack = [root]
            while stack:
                p = stack.pop()
                for c in list(p):
                    parent_of[c] = p
                    stack.append(c)
            return parent_of

        def _find_name_down(node: ET.Element) -> Optional[str]:
            NAME_TAGS = {
                "RecordName", "TraceName", "SeriesName", "Label",
                "DetectorName", "SampleName", "Name"
            }
            """Search downward (subtree) for any naming tag."""
            for tag in NAME_TAGS:
                f = node.find(f".//{tag}")
                if f is not None and f.text and f.text.strip():
                    return f.text.strip()
            return None

        def _guess_name(node: ET.Element, parent_of: dict, fallback: str) -> str:
            """
            Choose a human-readable label by searching:
            1) Downward within 'node'
            2) Upward through ancestors (up to 5 levels), then downward within each
            """
            name = _find_name_down(node)
            if name:
                return name
            steps, cur = 0, node
            while cur in parent_of and steps < 5:
                cur = parent_of[cur]
                steps += 1
                name = _find_name_down(cur)
                if name:
                    return name
            return fallback

        def _first_child_by_tag_set(parent: ET.Element, tagset: set) -> Optional[ET.Element]:
            
            for ch in list(parent):
                if _local(ch.tag) in tagset:
                    return ch
            return None

        def _extract_pattern_a(root: ET.Element, parent_of: dict) -> List[Tuple[str, List[float], List[float]]]:
            
            WAVEL_TAGS = {"Wavelengths", "Wavelength", "X", "XAxis", "WL"}
            INTENS_TAGS = {"Intensities", "Intensity", "Y", "YAxis", "Data", "Values"}
            
            spectra: List[Tuple[str, List[float], List[float]]] = []
            for node in root.iter():
                w0 = _first_child_by_tag_set(node, WAVEL_TAGS)
                y0 = _first_child_by_tag_set(node, INTENS_TAGS)

                # If not found directly, try one level deeper
                if w0 is None or y0 is None:
                    for child in list(node):
                        w0 = w0 or _first_child_by_tag_set(child, WAVEL_TAGS)
                        y0 = y0 or _first_child_by_tag_set(child, INTENS_TAGS)
                        if w0 is not None and y0 is not None:
                            node = child
                            break

                if w0 is None or y0 is None:
                    continue

                X = _to_floats(_elem_text(w0))
                Y = _to_floats(_elem_text(y0))
                if len(X) >= 4 and len(X) == len(Y):
                    fallback = "Trace"
                    name = _guess_name(node, parent_of, fallback)
                    # disambiguate duplicates
                    existing = [n for (n, _, _) in spectra]
                    if name in existing:
                        k = 2
                        nn = f"{name} ({k})"
                        while nn in existing:
                            k += 1
                            nn = f"{name} ({k})"
                        name = nn
                    spectra.append((name, X, Y))
            return spectra

        # ---------- Pattern B: per-point children (<Point><X>..</X><Y>..</Y></Point>) ----------
        def _leaf_numeric_map(elem: ET.Element) -> dict:
            """
            For a 'point' element, return {local_tag: float_value} for all numeric leaves under it.
            If a tag appears multiple times, keep the first.
            """
            out = {}
            for leaf in elem.iter():
                if list(leaf):  # has children → not a leaf
                    continue
                t = _local(leaf.tag)
                val = _elem_text(leaf).strip()
                if not val:
                    continue
                try:
                    f = float(val)
                except ValueError:
                    try:
                        f = float(val.replace(',', '.'))
                    except Exception:
                        continue
                if t not in out:
                    out[t] = f
            return out

        def _extract_pattern_b(root: ET.Element, parent_of: dict) -> List[Tuple[str, List[float], List[float]]]:
            
            spectra: List[Tuple[str, List[float], List[float]]] = []
            for node in root.iter():
                kids = list(node)
                if len(kids) < 8:   # need enough points to be a spectrum
                    continue

                # Build per-child numeric maps and find common numeric tags across all children
                maps = []
                common = None
                ok = True
                for ch in kids:
                    m = _leaf_numeric_map(ch)
                    if len(m) < 2:
                        ok = False
                        break
                    maps.append(m)
                    common = set(m.keys()) if common is None else (common & set(m.keys()))
                    if not common:
                        ok = False
                        break

                if not ok or not common or len(common) < 2:
                    continue

                tags = sorted(common)
                built_any = False
                for x_tag, y_tag in combinations(tags, 2):
                    X = [m[x_tag] for m in maps]
                    Y = [m[y_tag] for m in maps]
                    if len(X) == len(Y) and _is_monotonic_increasing(X):
                        fallback = _local(node.tag)
                        name = _guess_name(node, parent_of, fallback)
                        # disambiguate duplicates
                        existing = [n for (n, _, _) in spectra]
                        base, k = name, 2
                        while name in existing:
                            name = f"{base} ({k})"; k += 1
                        spectra.append((name, X, Y))
                        built_any = True
                # keep scanning; there may be multiple blocks elsewhere
            return spectra

        # ---------- public loader ----------
        def read_gxz_to_dataframe(path: str) -> pd.DataFrame:
            """
            Read a FelixGX .gxz (GZIPped XML) and return a DataFrame indexed by wavelength (X)
            with one column per spectrum. Supports both array-sibling and per-point layouts.
            """
            # Decompress + parse
            with gzip.open(path, 'rb') as g:
                xml_bytes = g.read()
            try:
                root = ET.fromstring(xml_bytes)
            except ET.ParseError:
                xml_str = xml_bytes.decode('utf-8', errors='replace').lstrip('\ufeff')
                root = ET.fromstring(xml_str)

            # Build parent map once for name resolution
            parent_of = _collect_parent_map(root)

            # Try both extraction strategies
            spectra = _extract_pattern_a(root, parent_of)
            if not spectra:
                spectra = _extract_pattern_b(root, parent_of)

            print(f"[GXZ] spectra found (combined): {len(spectra)}")
            if not spectra:
                raise ValueError("No spectra found. If needed, we can tweak tag detection for your export preset.")

            # Build DataFrame
            frames = []
            for name, X, Y in spectra:
                fr = pd.DataFrame({"X": X, name: Y})
                frames.append(fr)

            data = frames[0]
            for fr in frames[1:]:
                data = data.merge(fr, on="X", how="outer")

            data = data.sort_values("X").drop_duplicates("X").set_index("X")

            # Drop lamp-correction helper channels if they slipped in
            drop_mask = data.columns.str.contains(r"ExCorr|RCQC", case=False, na=False)
            data = data.loc[:, ~drop_mask]

            # Prefer corrected PMT traces first (ordering convenience)
            cols = list(data.columns)
            corrected = [c for c in cols if ('d1' in c.lower() and 'cor' in c.lower()) or ('[cor]' in c.lower())]
            others = [c for c in cols if c not in corrected]
            if corrected:
                data = data[[*corrected, *others]]

            return data
        
        path = str(Path(folder) / file)
        try:
            data = read_gxz_to_dataframe(path)
            print('GXZ file loaded successfully.')
        except Exception as e:
            print(f'Error loading GXZ file: {e}')
            data = pd.DataFrame()
        return data

    def createSpectra(self,data,Runs,Buffer='None',CamScale=0) :
        
        spectra = data.copy(deep=True)
        if Buffer != 'None':
            spectra = spectra.sub(spectra[Buffer], axis=0)
        if CamScale > 0:
            try:
                with open(settingsFile, 'r') as stream:
                    settings = yaml.safe_load(stream) or {}
                file_path = settings.get('files', {}).get('calmodulin', '')
                if file_path and Path(file_path).is_file():
                    Cam = pd.read_csv(file_path)
                    if 'X' not in Cam.columns:
                        raise ValueError("Calmodulin file must contain an 'X' column.")
                    Cam = Cam.set_index('X').sort_index()
                    scaling = []
                    for item in spectra.columns:
                        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", item)
                        if match:
                            scaling.append(float(match.group()))
                        else:
                            scaling.append(0)
                    cam = Cam.iloc[:, 0]  # first Y column
                    for idx, name in enumerate(spectra.columns):
                        spectra[name] = spectra[name] - CamScale * scaling[idx] * cam / 60
                else:
                    print("Calmodulin data not found in settings. Skipping calmodulin tyrosine removal.")
            except Exception as e:
                print(f"CaM subtraction skipped due to error: {e}")
        # Keep only selected runs, but preserve order
        runs = list(Runs) if isinstance(Runs, (list, tuple)) else [Runs]
        spectra = spectra.filter(items=runs)
        return spectra
    
    def plot(self, data, Title=''):
        
        fontsize = 16
        fig, ax = plt.subplots(figsize=(10, 8))
        handles = []
        labels = []
        for name in data.columns:
            ln, = ax.plot(data.index, data[name], label=name, lw=1.5)  # returns a Line2D
            handles.append(ln)
            labels.append(name)
        ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1.02, 1), fontsize=fontsize*0.75)
        ax.set_xlabel('Wavelength (nm)', fontsize=fontsize)
        ax.set_ylabel('Intensity (au)', fontsize=fontsize)
        ax.set_title(Title, fontsize=fontsize)
        ax.tick_params(axis='both', which='both', labelsize=fontsize, direction="in")
        ax.minorticks_on()
        fig.tight_layout()
        plt.show()
        return fig

    def plot2D(self, data, Title=''):
        
        fontsize = 14
        M = data.T  # rows=runs, cols=wavelength
        x = M.columns.to_numpy()
        runs = M.index.astype(str).to_list()
        y = np.arange(len(runs))

        plt.figure(figsize=(10, 6))
        im = plt.imshow(M.values, origin='lower', cmap='turbo', aspect='auto',
                        extent=[x.min(), x.max(), y.min(), y.max()])
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label("Intensity (au)", fontsize=fontsize)
        plt.xlabel('Wavelength (nm)', fontsize=fontsize)
        plt.ylabel('Run', fontsize=fontsize)
        plt.yticks(ticks=y + 0.5, labels=runs, fontsize=fontsize*0.75)  # center labels
        plt.title(Title, fontsize=fontsize)
        plt.tick_params(axis='both', which='both', labelsize=fontsize, direction="in")
        plt.tight_layout()
        plt.show()


class analysisTools :

    def __init__(self) :

        pass
    
    def LoadTitration(self,data,file) :
        try:
            if file.lower().endswith('.csv'):
                titration = pd.read_csv(file)
            elif file.lower().endswith('.xlsx'):
                titration = pd.read_excel(file)
            else:
                titration = None
                return
        except Exception as e:
            titration = None
        if data.shape[1] <= titration.shape[0]:
            columns = list()
            for idx, col in enumerate(data.columns) :
                try :
                    columns.append(str(float(titration.values[idx]))+' '+titration.columns[0])
                except :
                    columns.append(str(titration.values[idx][0]))
            data.columns = columns
        return data
    
    def ExtractTitration(self,data) :
        
        if any('D1' in col for col in data.columns):
            pass
        else:
            titration = list()
            for column in data.columns :
                try :
                    titration.append(float(column.split(' ')[0]))
                except :
                    titration.append(column)
            data.columns = titration
        return data

    def Integrate(self, data, limits):
        
        data = self.ExtractTitration(data)
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
        
        data = self.ExtractTitration(data)
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
            x_fine = np.linspace(x_vals.min(), x_vals.max(), int((x_vals.max()-x_vals.min())*20))
            y_fine = model.eval(result.params, x=x_fine)

            plt.scatter(data.index.to_numpy(), data[col].to_numpy(), s=20)
            plt.plot(x_fine, y_fine, linewidth=2, label=f'{col}')
            
            integratedAreas.append(result.params['amplitude'].value)
            peakValues.append(x_fine[np.argmax(y_fine)])

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
        df['Integrated Areas'] = integratedAreas
        df['Peak Values'] = peakValues
        df.index = data.columns
        
        return df


class UI :
    
    def __init__(self) :

        dt = dataTools()
        at = analysisTools()
        
        self.BufferNames = ['None']
        self.Names = ['']

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        self.settingsFile = settingsFile
        
        try:
            with open(self.settingsFile, 'r') as stream:
                settings = yaml.safe_load(stream) or {}
        except FileNotFoundError:
            settings = {'folders': {'data': str(Path.cwd())}, 'files': {}}

        if os.path.isdir(settings['folders']['data']) :
            self.cwd = settings['folders']['data']
        else :
            self.cwd = str(Path(os.getcwd()))

        out1 = ipw.Output()
        out2 = ipw.Output()
        out3 = ipw.Output()
        
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
            with out1 :
                clear_output()
                display(ipw.Box([Filter,Update_RunList,CamScale]))
                display(ipw.Box([TitrationFile,UpdateTitration]))
                display(ipw.Box([Runs_Selected,Plot]))
                display(ipw.Box([Buffer]))
            with out2 :
                clear_output()
            with out3 :
                clear_output()
            Buffer.value = 'None'
            Runs_Selected.value = []
            suffix = (SelectFile.value or '').lower()
            if suffix.endswith('.gxz'):
                data = dt.loadGXZ(folderField.value, SelectFile.value)
            else:
                data = dt.loadText(folderField.value, SelectFile.value)
            self.filename = SelectFile.value
            self.data = data
            RunList()
        load_button = ipw.Button(description='Load',layout=Layout(width='10%'))
        load_button.on_click(load)

        def RunList():
            self.Runs = list(self.data.columns.values.astype(str))
            Runs = [k for k in self.Runs if Filter.value in k]
            Runs_Selected.options = Runs
            Runs.insert(0,'None')
            Buffer.options = Runs

        Filter = ipw.Text(
            value='',
            placeholder='Type something',
            description='Filter',
            style = {'description_width': '150px'},
            disabled=False
        )
        
        def Update_RunList_Clicked(b):
            RunList()
        Update_RunList = ipw.Button(description="Update run list")
        Update_RunList.on_click(Update_RunList_Clicked)
        
        CamScale = ipw.BoundedFloatText(
            value=0.0,
            min=0.0,
            max=10,
            step=0.01,
            description='CaM Subtraction:',
            layout=ipw.Layout(width='200px'),
            style={'description_width': '120px'}
        )

        def UpdateTitration_Clicked(b):
            self.data = at.LoadTitration(self.data,TitrationFile.value)   
            RunList()
        UpdateTitration = ipw.Button(
            description="Update Titration",
            )
        UpdateTitration.on_click(UpdateTitration_Clicked)
        
        TitrationFile = ipw.Text(
            value=settings['files']['titration'],
            description='Titration File',
            style = {'description_width': '150px'},
            layout=Layout(width='40%'),
            disabled=False
        )

        Runs_Selected = ipw.SelectMultiple(
            options='',
            style = {'width': '100px','description_width': '150px'},
            rows=20,
            layout=Layout(width='70%'),
            description='Runs',
            disabled=False
        )

        Buffer = ipw.Dropdown(
            options=self.BufferNames,
            value='None',
            layout=Layout(width='70%'),
            description='Buffer',
            style = {'description_width': '150px'},
            disabled=False,
        )

        def Plot_Clicked(b):
            with out2 :
                data = self.data
                clear_output()
                spectra = dt.createSpectra(data,Runs_Selected.value,Buffer=Buffer.value,CamScale=CamScale.value)
                self.fig = dt.plot(spectra)
                display(ipw.Box([SavePlot,SpectraToClipboard,Plot2D]))
                display(ipw.Box([Integrate,Fit]))
                display(LowLim)
                display(UpLim)
            with out3 :
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
            with out3 :
                clear_output()
                dt.plot2D(self.spectra)
        Plot2D = ipw.Button(description="Plot 2D")
        Plot2D.on_click(Plot2d_Clicked)

        def Integrate_clicked(b):
            spectra = self.spectra.copy(deep=True)
            with out3 :
                clear_output()
                limits = [LowLim.value, UpLim.value]
                self.integratedValues = at.Integrate(spectra, limits)
                display(IntegratedToClipboard)
        Integrate = ipw.Button(description="Integrate")
        Integrate.on_click(Integrate_clicked)
        
        def Fit_clicked(b):
            spectra = self.spectra.copy(deep=True)
            with out3 :
                clear_output()
                limits = [LowLim.value, UpLim.value]
                self.fitAnalysis = at.Fit(spectra, limits)
                display(FitAnalysisToClipboard)
        Fit = ipw.Button(description="Fit")
        Fit.on_click(Fit_clicked)

        def SavePlot_Clicked(b):
            out_path = Path(self.filename).with_suffix('.png')
            self.fig.savefig(out_path, dpi=300, bbox_inches='tight')
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


        display(out1)
        display(out2)
        display(out3)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)
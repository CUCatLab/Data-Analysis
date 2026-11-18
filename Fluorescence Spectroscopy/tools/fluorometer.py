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
import tempfile
import zipfile
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
                "DetectorName", "ExperimentName", "Name"
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

    def createSpectra(self,data,Runs,Buffer='None',CamSubtraction=0) :
        
        spectra = data.copy(deep=True)
        if Buffer != 'None':
            spectra = spectra.sub(spectra[Buffer], axis=0)
        if CamSubtraction > 0:
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
                        spectra[name] = spectra[name] - CamSubtraction * scaling[idx] * cam / 60
                else:
                    print("Calmodulin data not found in settings. Skipping calmodulin tyrosine removal.")
            except Exception as e:
                print(f"CaM subtraction skipped due to error: {e}")
        # Keep only selected runs, but preserve order
        runs = list(Runs) if isinstance(Runs, (list, tuple)) else [Runs]
        spectra = spectra.filter(items=runs)
        return spectra
    
    def plot(self, data, Title=''):
        
        fontsize = 24
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
        ax.set_yticks([])
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

    def ExtractDateFromFilename(self,filename):

        filename = Path(
            filename
        ).stem
        patterns = [
            r'(\d{8})',
            r'(\d{6})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        for pattern in patterns:
            match = re.search(
                pattern,
                filename
            )
            if match:
                return match.group(1)
        return 'Unknown'

    def ExtractExperimentFromFilename(
            self,
            filename
        ):

        stem = Path(filename).stem

        if 'C-MA' in stem:
            ma = 'C-term MA'
        elif 'N-MA' in stem:
            ma = 'N-term MA'
        elif 'MA' in stem:
            ma = 'Full MA'
        else:
            ma = 'Unknown MA'

        if 'Full-CaM' in stem:
            cam = 'Full CaM'
        elif 'C-CaM' in stem:
            cam = 'C-term CaM'
        elif 'N-CaM' in stem:
            cam = 'N-term CaM'
        else:
            cam = 'Unknown CaM'

        experiment = f'{ma} + {cam}'

        if 'APO' in stem.upper():
            experiment += ' (Apo)'

        return experiment

    def AdjustConcentration(self,data,volume) :
        
        tempData = self.ExtractTitration(data.copy(deep=True))
        titration = pd.to_numeric(tempData.columns, errors='coerce').fillna(0).astype(int)
        data *= (volume+titration)/volume
        return data

    def _titrant_axis(self, columns, fallback_values=None):
        """Return titrant volumes for run columns, with deterministic fallbacks.

        Priority: volume parsed from run name, optional user-entered fallback list,
        numeric column label, then zero-based acquisition order.
        """
        cols = [str(c) for c in columns]
        manual = []
        if fallback_values:
            try:
                manual = [float(v.strip()) for v in str(fallback_values).split(',') if v.strip()]
            except ValueError:
                manual = []
        values, sources = [], []
        for i, col in enumerate(cols):
            parsed = self.ParseRunName(col)['Titrant (uL)']
            if parsed is not None:
                values.append(float(parsed)); sources.append('run name')
            elif len(manual) == len(cols):
                values.append(manual[i]); sources.append('manual fallback')
            else:
                try:
                    values.append(float(col)); sources.append('numeric label')
                except ValueError:
                    values.append(float(i)); sources.append('acquisition order')
        return np.asarray(values, float), sources

    def Integrate(self, data, limits, fallback_values=None):
        """Numerically integrate selected spectra and plot against titrant amount."""
        lower, upper = sorted(map(float, limits))
        subset = data.loc[(data.index >= lower) & (data.index <= upper)]
        amounts, sources = self._titrant_axis(subset.columns, fallback_values)
        rows = []
        for i, column in enumerate(subset.columns):
            x = np.asarray(subset.index, float)
            y = np.asarray(subset[column], float)
            finite = np.isfinite(x) & np.isfinite(y)
            area = float(integrate.trapezoid(y[finite], x=x[finite]))
            rows.append({'Run': str(column), 'Titrant (uL)': amounts[i],
                         'Integrated': area, 'Titrant source': sources[i]})
        result = pd.DataFrame(rows).sort_values('Titrant (uL)')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result['Titrant (uL)'], result['Integrated'], 'o-')
        ax.set(xlabel='Titrant added (uL)', ylabel='Integrated intensity (au)',
               title=f'Integrated fluorescence: {lower:g}-{upper:g} nm')
        ax.grid(True, alpha=.25); fig.tight_layout(); plt.show()
        return result

    def Fit(self, data, limits, method='Skewed Gaussian', r2_threshold=0.95,
            fallback_values=None):
        """Use the same lambda-max model and quality criteria as Analyze."""
        amounts, sources = self._titrant_axis(data.columns, fallback_values)
        rows, details = [], []
        for i, column in enumerate(data.columns):
            fit = self._fit_peak(data.index, data[column], limits, method, r2_threshold)
            lam, xf, yf, xfine, yfine, r2, rmse, used_method, fit_ok = fit
            rows.append({'Run': str(column), 'Titrant (uL)': amounts[i],
                         'Lambda max (nm)': lam, 'R2': r2, 'RMSE': rmse,
                         'Fit method': used_method, 'Fit OK': fit_ok,
                         'Titrant source': sources[i]})
            details.append((str(column), amounts[i], xf, yf, xfine, yfine, fit_ok))
        result = pd.DataFrame(rows).sort_values('Titrant (uL)')

        fig1, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('viridis')
        unique = sorted(set(amounts))
        colors = {a:cmap(j/max(1,len(unique)-1)) for j,a in enumerate(unique)}
        for run, amount, xf, yf, xfine, yfine, ok in details:
            c=colors[amount]
            ax.plot(xf, yf, 'o', ms=3, alpha=.45, color=c, label=f'{amount:g} uL')
            ax.plot(xfine, yfine, '-' if ok else '--', lw=2, color=c)
        handles, labels = ax.get_legend_handles_labels(); dedup=dict(zip(labels,handles))
        ax.legend(dedup.values(),dedup.keys(),title='Titrant',frameon=False)
        ax.set(xlabel='Wavelength (nm)',ylabel='Intensity (au)',title=f'{method} fits')
        ax.grid(True,alpha=.25); fig1.tight_layout(); plt.show()

        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(result['Titrant (uL)'], result['Lambda max (nm)'], 'o-')
        bad=result[~result['Fit OK']]
        if not bad.empty:
            ax2.plot(bad['Titrant (uL)'],bad['Lambda max (nm)'],'x',ms=10,mew=2,color='red',label='Below fit threshold')
            ax2.legend(frameon=False)
        ax2.set(xlabel='Titrant added (uL)',ylabel='Lambda max (nm)',title='Peak position')
        ax2.grid(True,alpha=.25); fig2.tight_layout(); plt.show()
        return result

    @staticmethod
    def CanonicalExperimentName(experiment):

        s = experiment.lower()

        s = s.replace('_', '-')
        s = s.replace(' ', '-')
        s = re.sub(r'-+', '-', s)
        s = s.strip('-')

        aliases = {

            # MA
            'c-ma'        : 'C-term MA',
            'cterm-ma'    : 'C-term MA',
            'c-term-ma'   : 'C-term MA',

            'n-ma'        : 'N-term MA',
            'nterm-ma'    : 'N-term MA',
            'n-term-ma'   : 'N-term MA',

            'ma'          : 'Full MA',
            'full-ma'     : 'Full MA',
            'fullma'      : 'Full MA',

            # CaM
            'cam'         : 'Full CaM',
            'full-cam'    : 'Full CaM',
            'fullcam'     : 'Full CaM',

            'c-cam'       : 'C-term CaM',
            'cterm-cam'   : 'C-term CaM',
            'c-term-cam'  : 'C-term CaM',

            'n-cam'       : 'N-term CaM',
            'nterm-cam'   : 'N-term CaM',
            'n-term-cam'  : 'N-term CaM'
        }

        return aliases.get(s, experiment)

    def ParseRunName(self, run_name):
        name = str(run_name)
        name = re.sub(
            r'_D\d+$',
            '',
            name
        )
        pol_match = re.search(
            r'(?:^|[_\-\s])(VV|VH)(?=$|[_\-\s])',
            name,
            re.IGNORECASE
        )
        polarization = (
            pol_match.group(1).upper()
            if pol_match
            else None
        )
        amount_match = re.search(
            r'([-+]?\d+(?:\.\d+)?)\s*(?:u|µ|μ)\s*[lL]',
            name,
            re.IGNORECASE
        )

        is_buffer = (
            'buffer' in name.lower()
        )

        is_apo = bool(
            re.search(
                r'\bapo\b',
                name,
                re.IGNORECASE
            )
        )

        # Build experiment name first
        experiment = re.sub(
            r'([-+]?\d+(?:\.\d+)?)\s*(?:u|µ|μ)\s*[lL]',
            '',
            name,
            flags=re.IGNORECASE
        )

        experiment = re.sub(
            r'_D\d+$',
            '',
            experiment,
            flags=re.IGNORECASE
        )

        experiment = re.sub(
            r'(?:^|[_\-\s])(?:VV|VH)(?=$|[_\-\s])',
            '',
            experiment,
            flags=re.IGNORECASE
        ).strip('_- ')
        experiment = self.CanonicalExperimentName(experiment)
        display_experiment = experiment
        # Determine titrant amount
        amount = None
        if amount_match:
            amount = float(
                amount_match.group(1)
            )
            # Initial cuvette species (MA) should be treated as 0 uL titrant.
            if (
                ('MA' in name.upper())
                and ('CAM' not in name.upper())
            ):
                amount = 0.0
        elif polarization and not is_buffer:
            amount = 0.0
        rep_match = re.search(
            r'(?:^|[_\-\s])(?:rep|r)?(\d+)$',
            name,
            re.IGNORECASE
        )
        replicate = (
            int(rep_match.group(1))
            if rep_match
            else 1
        )

        return {
            'Experiment': experiment,
            'Apo': is_apo,
            'DisplayExperiment':
                display_experiment,
            'Titrant (uL)': amount,
            'Polarization': polarization,
            'Replicate': replicate,
            'Buffer': is_buffer
        }

    def AutoTitration(self, data):
        records = []
        for run in data.columns:
            rec = self.ParseRunName(run)
            rec['Run'] = str(run)
            records.append(rec)
        return pd.DataFrame(records).set_index('Run')

    @staticmethod
    def _fit_peak(x, y, limits, method='Skewed Gaussian', r2_threshold=0.95):
        x, y = np.asarray(x, float), np.asarray(y, float)
        lower, upper = sorted(map(float, limits))
        mask = np.isfinite(x) & np.isfinite(y) & (x >= lower) & (x <= upper)
        x_fit, y_fit = x[mask], y[mask]
        if x_fit.size < 5:
            raise ValueError('At least five finite points are required between the fit limits.')
        if method == 'Smoothed maximum':
            window = min(11, x_fit.size if x_fit.size % 2 else x_fit.size-1)
            window = max(3, window)
            smooth = pd.Series(y_fit).rolling(window, center=True, min_periods=1).mean().to_numpy()
            i = int(np.nanargmax(smooth))
            return float(x_fit[i]), x_fit, y_fit, x_fit, smooth, np.nan, np.nan, 'Smoothed maximum', True
        model = SkewedGaussianModel()
        result = model.fit(y_fit, model.guess(y_fit, x=x_fit), x=x_fit, nan_policy='omit')
        predicted = model.eval(result.params, x=x_fit)
        residual = y_fit - predicted
        ss_res = float(np.nansum(residual**2))
        ss_tot = float(np.nansum((y_fit-np.nanmean(y_fit))**2))
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(np.nanmean(residual**2)))
        x_fine = np.linspace(x_fit.min(), x_fit.max(), max(500, int((x_fit.max()-x_fit.min())*20)))
        y_fine = model.eval(result.params, x=x_fine)
        lam = float(x_fine[int(np.nanargmax(y_fine))])
        ok = bool(np.isfinite(r2) and r2 >= r2_threshold)
        return lam, x_fit, y_fit, x_fine, y_fine, r2, rmse, 'Skewed Gaussian', ok
    
    @staticmethod
    def _experiment_count(fits):
        """
        Number of independent experiments/files contributing to the plot.
        """
        if 'Dataset' not in fits.columns:
            return 1

        n = fits['Dataset'].dropna().nunique()
        return max(1, int(n))

    @staticmethod
    def _plot_mean_sd(
        ax,
        x,
        y,
        sd,
        color,
        label=None,
        capsize=3
    ):
        """
        Plot mean ± SD.
        """
        ax.errorbar(
            x,
            y,
            yerr=sd.fillna(0),
            fmt='o-',
            color=color,
            label=label,
            capsize=capsize
        )

    def AnalyzeAnisotropy(
            self,
            data,
            fit_limits,
            integration_limits,
            dilution_volume=None,
            g_factor=1.0,
            fit_method='Skewed Gaussian',
            r2_threshold=0.95,
            error_bar_type='SD',
            dataset_map=None
        ):
        """Analyze all loaded data, validate VV/VH pairs, average replicates, and plot results."""
        if data is None or data.empty:
            raise ValueError('No fluorescence data are loaded.')
        if fit_limits[1] <= fit_limits[0] or integration_limits[1] <= integration_limits[0]:
            raise ValueError('Upper limits must be greater than lower limits.')
        if g_factor <= 0:
            raise ValueError('G factor must be greater than zero.')
        info = self.AutoTitration(data)
        experiments = info[~info['Buffer']]
        bad_pol = experiments.index[experiments['Polarization'].isna()].tolist()
        bad_amount = experiments.index[experiments['Titrant (uL)'].isna()].tolist()
        if bad_pol: raise ValueError('Could not identify VV or VH in: ' + ', '.join(bad_pol))
        if bad_amount: raise ValueError('Could not identify titrant amount in: ' + ', '.join(bad_amount))

        corrected = data.copy(deep=True)
        for pol in ('VV','VH'):
            buffers = info.index[info['Buffer'] & (info['Polarization']==pol)].tolist()
            if not buffers:
                raise ValueError(f'No {pol} buffer found. Include "buffer" and "{pol}" in its run name.')
            baseline = data[buffers].mean(axis=1)
            targets = info.index[(~info['Buffer']) & (info['Polarization']==pol)]
            corrected.loc[:, targets] = corrected.loc[:, targets].sub(baseline, axis=0)

        x_all = np.asarray(corrected.index, float)
        ilow, ihigh = sorted(map(float, integration_limits))
        imask = np.isfinite(x_all) & (x_all >= ilow) & (x_all <= ihigh)
        if imask.sum() < 2: raise ValueError('At least two points are required within the integration limits.')
        rows, details = [], []
        for run, meta in experiments.iterrows():
            y = corrected[run].copy()
            amount = float(meta['Titrant (uL)'])
            if dilution_volume is not None and dilution_volume > 0:
                y *= (dilution_volume + amount) / dilution_volume
            fit = self._fit_peak(corrected.index, y, fit_limits, fit_method, r2_threshold)
            lam, xf, yf, xfine, yfine, r2, rmse, used_method, fit_ok = fit
            yi = np.asarray(y, float)[imask]; xi = x_all[imask]
            finite = np.isfinite(xi) & np.isfinite(yi)
            area = float(integrate.simpson(yi[finite], x=xi[finite]))
            row = {'Run':run, 'Experiment':meta['Experiment'], 'Apo': meta['Apo'], 'DisplayExperiment':meta['DisplayExperiment'], 'Titrant (uL)':amount,
                   'Polarization':meta['Polarization'], 'Replicate':meta['Replicate'],
                   'Lambda max (nm)':lam, 'Integrated intensity':area,
                   'R2':r2, 'RMSE':rmse, 'Fit method':used_method, 'Fit OK':fit_ok}
            rows.append(row)
            details.append((row, xf, yf, xfine, yfine))
        fits = pd.DataFrame(rows)
        fits['Dataset'] = fits['Run'].str.extract(r'(D\d+)$')
        fits['Experiment'] = fits['Dataset'].map(
            lambda d:
                dataset_map[d]['experiment']
                if (
                    dataset_map is not None
                    and pd.notna(d)
                    and d in dataset_map
                )
                else 'Unknown'
        )
        fits['Date'] = fits['Dataset'].map(
            lambda d: dataset_map[d]['date']
            if dataset_map is not None and pd.notna(d) and d in dataset_map
            else 'Unknown'
        )
        n_experiments = self._experiment_count(fits)

        groupcols=[
            'Experiment',
            'Titrant (uL)',
            'Polarization'
        ]
        summary=fits.groupby(groupcols,as_index=False).agg(
            **{'Lambda max (nm)':('Lambda max (nm)','mean'),
               'Lambda SD':('Lambda max (nm)','std'),
               'Integrated intensity':('Integrated intensity','mean'),
               'Intensity SD':('Integrated intensity','std'),
               'n':('Run','count'), 'Fit OK':('Fit OK','all')})
        summary['Lambda SEM'] = summary['Lambda SD'] / np.sqrt(summary['n'])
        pair_counts=summary.pivot_table(
            index=['Experiment','Titrant (uL)'],
            columns='Polarization',
            values='n',
            aggfunc='sum'
        )
        missing=[]
        for key,row in pair_counts.iterrows():
            absent=[p for p in ('VV','VH') if p not in row.index or pd.isna(row.get(p))]
            if absent: missing.append(f'{key[0]}, {key[1]:g} uL: missing {"/".join(absent)}')
        complete=summary.pivot_table(
            index=['Experiment','Titrant (uL)'],
            columns='Polarization',
            values='Integrated intensity'
        )
        complete=complete.dropna(subset=['VV','VH'])
        complete['Anisotropy r']=(complete['VV']-g_factor*complete['VH'])/(complete['VV']+2*g_factor*complete['VH'])

        # Propagate replicate variability by calculating replicate-paired anisotropy when possible.
        rep = fits.pivot_table(
            index=[
                'Experiment',
                'Titrant (uL)',
                'Dataset',
                'Replicate'
            ],
            columns='Polarization',
            values='Integrated intensity'
        )   
        rep=rep.dropna(subset=['VV','VH'])
        rep['r']=(rep['VV']-g_factor*rep['VH'])/(rep['VV']+2*g_factor*rep['VH'])
        rstats=rep.groupby(level=[0,1])['r'].agg(['mean','std','count']).rename(columns={'mean':'Anisotropy mean','std':'Anisotropy SD','count':'n pairs'})
        rstats['Anisotropy SEM'] = rstats['Anisotropy SD'] / np.sqrt(rstats['n pairs'])
        results=complete.join(rstats,how='left')
        results['Anisotropy mean']=results['Anisotropy mean'].fillna(results['Anisotropy r'])

        # Coordinated fit figure with same color per titrant in VV and VH.
        fit_fig,(avv,avh)=plt.subplots(1,2,figsize=(14,6),sharex=True)
        axes={'VV':avv,'VH':avh}; amounts=sorted(fits['Titrant (uL)'].unique())
        cmap=plt.get_cmap('viridis'); colors={a:cmap(i/max(1,len(amounts)-1)) for i,a in enumerate(amounts)}
        self._fit_artists={}
        for row,xf,yf,xline,yline in details:
            ax=axes[row['Polarization']]; c=colors[row['Titrant (uL)']]
            points=ax.plot(xf,yf,'o',ms=3,alpha=.45,color=c,label=f"{row['Titrant (uL)']:g} uL")[0]
            style='-' if row['Fit OK'] else '--'; line=ax.plot(xline,yline,style,lw=2,color=c,alpha=1 if row['Fit OK'] else .75)[0]
            self._fit_artists[(row['Experiment'],row['Titrant (uL)'],row['Polarization'])]=(points,line)
        for pol,ax in axes.items():
            ax.set(title=pol,xlabel='Wavelength (nm)',ylabel='Intensity'); ax.grid(True,alpha=.2)
            handles,labels=ax.get_legend_handles_labels(); unique=dict(zip(labels,handles)); ax.legend(unique.values(),unique.keys(),title='Titrant',frameon=False,fontsize=8)
        dates = sorted(fits['Date'].unique())
        dates = sorted(
            set(dates)
        )
        experiments = sorted(
            set(
                fits[
                    'DisplayExperiment'
                ]
                if 'DisplayExperiment'
                in fits.columns
                else
                fits['Experiment']
            )
        )
        fit_fig.tight_layout()
        fit_fig.canvas.draw()
        plt.show()

        # Stacked summary, shared x-axis, error bars for replicates.
        summary_fig,(a1,a2)=plt.subplots(
            2,1,
            figsize=(8,6),
            sharex=True
        )
        self._summary_points = {}
        colors = plt.cm.tab10.colors
        for experiment_idx, experiment in enumerate(
            sorted(summary['Experiment'].unique())
        ):
            experiment_color = colors[
                experiment_idx % len(colors)
            ]
            for pol, linestyle in [
                ('VV', '-'),
                ('VH', '--')
            ]:
                part = summary[
                    (summary['Experiment'] == experiment) &
                    (summary['Polarization'] == pol)
                ].sort_values('Titrant (uL)')
                if part.empty:
                    continue
                yerr = (
                    part['Lambda SD']
                    if error_bar_type == 'SD'
                    else part['Lambda SEM']
                )
                n_rep = int(part['n'].max())
                artists = a1.errorbar(
                    part['Titrant (uL)'],
                    part['Lambda max (nm)'],
                    yerr=yerr.fillna(0),
                    fmt='o',
                    linestyle=linestyle,
                    color=experiment_color,
                    label=f'{experiment} ({pol}, n={n_rep}))',
                    capsize=3
                )
                for x in part['Titrant (uL)']:
                    self._summary_points[
                        (float(x), experiment, pol)
                    ] = artists[0]
        rr = results.reset_index()
        for experiment_idx, experiment in enumerate(
            sorted(summary['Experiment'].unique())
        ):
            experiment_color = colors[
                experiment_idx % len(colors)
            ]
            part = rr[
                rr['Experiment'] == experiment
            ].sort_values(
                'Titrant (uL)'
            )
            if part.empty:
                continue
            yerr = (
                part['Anisotropy SD']
                if error_bar_type == 'SD'
                else part['Anisotropy SEM']
            )
            n_rep = int(part['n pairs'].max()) if 'n pairs' in part.columns else 1
            a2.errorbar(
                part['Titrant (uL)'],
                part['Anisotropy mean'],
                yerr=yerr.fillna(0),
                fmt='o-',
                color=experiment_color,
                label=f'{experiment} (n={n_rep})',
                capsize=3
            )
        a2.legend(frameon=False)
        a1.set_ylabel(r'Lambda max (nm)'); a1.legend(frameon=False); a1.grid(True,alpha=.25)
        a1.set_title(r'Peak Absorption Wavelength ($\lambda_{\mathrm{max}}$) as a Function of Titrant')
        proteins = sorted(
            fits[
                'DisplayExperiment'
            ].unique()
        ) if 'DisplayExperiment' in fits.columns else sorted(
            fits[
                'Experiment'
            ].unique()
        )
        dates = sorted(
            fits[
                'Date'
            ].unique()
        )
        n = fits['Dataset'].nunique()
        # title = ', '.join(proteins)
        a2.set_title('Fluorescence Anisotropy Binding Curve')
        a2.set(xlabel='Titrant added (uL)',ylabel='Anisotropy, r'); a2.grid(True,alpha=.25)
        summary_fig.tight_layout()

        # Clicking a summary point highlights the matching VV/VH spectra when the backend supports picking.
        def on_pick(event):
            if event.artist.axes not in (a1,a2) or not len(event.ind): return
            x=float(event.artist.get_xdata()[event.ind[0]])
            for (experiment,amount,pol),(pts,line) in self._fit_artists.items():
                selected = (np.isclose(amount, x))
                pts.set_alpha(1.0 if selected else .12); line.set_alpha(1.0 if selected else .12)
                line.set_linewidth(3.5 if selected else 1.5)
            fit_fig.canvas.draw_idle()
        summary_fig.canvas.mpl_connect('pick_event',on_pick)
        summary_fig.canvas.draw()
        plt.show()
        return fits,results,corrected,fit_fig,summary_fig,missing


class UI :
    
    def __init__(self) :

        dt = dataTools()
        at = analysisTools()
        
        self.BufferNames = ['None']
        self.Names = ['']
        self.experiments = {}

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
                display(ipw.Box([Filter,Update_RunList]))
                display(ipw.Box([TitrationFile,LoadTitration]))
                display(ipw.Box([CamSubtraction,DilutionCorrection,CuvetteVolume,NormalizeDisplay]))
                display(Runs_Selected)
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
            UpdatePlot()
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
        
        def LoadTitration_Clicked(b):
            self.data = at.LoadTitration(self.data,TitrationFile.value)   
            RunList()
        LoadTitration = ipw.Button(
            description="Load Titration",
            )
        LoadTitration.on_click(LoadTitration_Clicked)
        
        TitrationFile = ipw.Text(
            value=settings['files']['titration'],
            description='Titration File',
            style = {'description_width': '150px'},
            layout=Layout(width='40%'),
            disabled=False
        )
        
        CamSubtraction = ipw.BoundedFloatText(
            value=0.0,
            min=0.0,
            max=10,
            step=0.01,
            description='CaM Subtraction',
            layout=ipw.Layout(width='160px',margin='0 0 0 170px'),
            style={'description_width': '100px'}
        )
        
        DilutionCorrection = ipw.Checkbox(
            value=False,
            description='Correct for Dilution',
            disabled=False,
            indent=False,
            layout=ipw.Layout(width='160px',margin='0 0 0 50px'),
            style={'description_width': '250px'}
        )
        
        CuvetteVolume = ipw.BoundedFloatText(
            value=1500,
            min=0,
            max=2000,
            step=1,
            description='Volume (μL)',
            layout=ipw.Layout(width='150px'),
            style={'description_width': '70px'}
        )

        GFactor = ipw.BoundedFloatText(value=1.0, min=0.0001, max=1000.0, step=0.01,
            description='G factor:', style={'description_width':'100px'})
        FitMethod = ipw.Dropdown(options=['Skewed Gaussian','Smoothed maximum'], value='Skewed Gaussian',
            description='Lambda max (nm):', style={'description_width':'100px'})
        R2Threshold = ipw.BoundedFloatText(value=0.95, min=0.0, max=1.0, step=0.01,
            description='Min R^2:', style={'description_width':'100px'})
        ErrorBarType = ipw.Dropdown(options=['SD','SEM'], value='SD',
            description='Error bars:', style={'description_width':'100px'})
        TitrantFallback = ipw.Text(
            value='', placeholder='e.g., 0,1,2,3',
            description='Fallback uL:', style={'description_width':'100px'})
        NormalizeDisplay = ipw.Checkbox(value=False, description='Normalize displayed spectra')

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

        def UpdatePlot(change=None):
            # Redraw automatically whenever the selected runs or plotting options change.
            # Do nothing until a file is loaded and at least one run is selected.
            if not hasattr(self, 'data'):
                return
            if len(Runs_Selected.value) == 0:
                with out2:
                    clear_output(wait=True)
                    empty = pd.DataFrame(index=self.data.index)
                    self.fig = dt.plot(empty)
                    display(ipw.Box([SavePlot, SpectraToClipboard, Plot2D]))
                    display(ipw.Box([Integrate, Fit, Analyze]))
                    display(ipw.Box([GFactor, FitMethod, R2Threshold, ErrorBarType, TitrantFallback]))
                    display(LowLim)
                    display(UpLim)
                    display(IntegrationLow)
                    display(IntegrationHigh)
                with out3:
                    clear_output(wait=True)
                wavelengths = np.asarray(self.data.index, dtype=float)
                xmin = float(np.nanmin(wavelengths))
                xmax = float(np.nanmax(wavelengths))
                # Set values while the widgets still have permissive bounds, then
                # narrow the bounds to the actual wavelength range.
                LowLim.value = xmin
                UpLim.value = xmax
                LowLim.min = xmin
                LowLim.max = xmax
                UpLim.min = xmin
                UpLim.max = xmax
                IntegrationLow.value = xmin
                IntegrationHigh.value = xmax
                IntegrationLow.min = xmin
                IntegrationLow.max = xmax
                IntegrationHigh.min = xmin
                IntegrationHigh.max = xmax
                self.spectra = empty
                return

            with out2:
                clear_output(wait=True)
                data = self.data.copy(deep=True)
                if DilutionCorrection.value:
                    data = at.AdjustConcentration(data, CuvetteVolume.value)
                spectra = dt.createSpectra(
                    data, Runs_Selected.value,
                    Buffer=Buffer.value,
                    CamSubtraction=CamSubtraction.value)
                if NormalizeDisplay.value:
                    scale = spectra.abs().max(axis=0).replace(0, np.nan)
                    spectra = spectra.divide(scale, axis=1)
                self.fig = dt.plot(spectra)
                display(ipw.Box([SavePlot, SpectraToClipboard, Plot2D]))
                display(ipw.Box([Integrate, Fit, Analyze]))
                display(ipw.Box([GFactor, FitMethod, R2Threshold, ErrorBarType, TitrantFallback]))
                display(LowLim)
                display(UpLim)
                display(IntegrationLow)
                display(IntegrationHigh)

            with out3:
                clear_output(wait=True)

            # Initialize the existing limit sliders from the selected spectra only
            # when the limits are not valid for the new wavelength range. This keeps
            # user-selected bounds while runs are added/removed from the same dataset.
            xmin = float(np.nanmin(np.asarray(spectra.index, dtype=float)))
            xmax = float(np.nanmax(np.asarray(spectra.index, dtype=float)))
            old_low, old_high = LowLim.value, UpLim.value
            # Existing selections are already inside the loaded wavelength range;
            # set/clamp values before changing bounds to avoid trait validation
            # leaving the sliders at their initial zero values.
            new_low = old_low if xmin <= old_low <= xmax else xmin
            new_high = old_high if xmin <= old_high <= xmax else xmax
            LowLim.value = new_low
            UpLim.value = new_high
            LowLim.min = xmin
            LowLim.max = xmax
            UpLim.min = xmin
            UpLim.max = xmax
            if IntegrationHigh.value <= IntegrationLow.value or not (xmin <= IntegrationLow.value <= xmax) or not (xmin <= IntegrationHigh.value <= xmax):
                IntegrationLow.value = xmin
                IntegrationHigh.value = xmax
            IntegrationLow.min = xmin
            IntegrationLow.max = xmax
            IntegrationHigh.min = xmin
            IntegrationHigh.max = xmax
            if old_high <= old_low or not (xmin <= old_low <= xmax) or not (xmin <= old_high <= xmax):
                LowLim.value = xmin
                UpLim.value = xmax
            else:
                LowLim.value = old_low
                UpLim.value = old_high
            self.spectra = spectra

        Runs_Selected.observe(UpdatePlot, names='value')
        Buffer.observe(UpdatePlot, names='value')
        CamSubtraction.observe(UpdatePlot, names='value')
        DilutionCorrection.observe(UpdatePlot, names='value')
        CuvetteVolume.observe(UpdatePlot, names='value')
        NormalizeDisplay.observe(UpdatePlot, names='value')

        def Analyze_Clicked(b):
            with out3:
                clear_output(wait=True)
                try:
                    if not hasattr(self, 'data') or self.data.empty:
                        raise ValueError('Load a data file before analyzing.')
                    fit_limits = [LowLim.value, UpLim.value]
                    integration_limits = [IntegrationLow.value, IntegrationHigh.value]
                    dilution_volume = CuvetteVolume.value if DilutionCorrection.value else None
                    analysis_data = self.data.copy(deep=True)
                    fallback, _ = at._titrant_axis(analysis_data.columns, TitrantFallback.value)
                    renamed = []
                    for i, col in enumerate(analysis_data.columns):
                        if at.ParseRunName(col)['Titrant (uL)'] is None and 'buffer' not in str(col).lower():
                            renamed.append(f'{col}_{fallback[i]:g}uL')
                        else:
                            renamed.append(col)
                    analysis_data.columns = renamed
                    (self.anisotropyFits, self.anisotropyResults,
                     self.anisotropySpectra, self.fitFig,
                     self.analysisFig, warnings) = at.AnalyzeAnisotropy(
                        analysis_data,
                        fit_limits=fit_limits,
                        integration_limits=integration_limits,
                        dilution_volume=dilution_volume,
                        g_factor=GFactor.value,
                        fit_method=FitMethod.value,
                        r2_threshold=R2Threshold.value,
                        error_bar_type=ErrorBarType.value,
                        dataset_map=self.datasetMap
                    )
                    if warnings:
                        print('Pairing warnings:')
                        for warning in warnings: print(' - ' + warning)
                    display(CopyAnalysisData)
                except Exception as e:
                    print(f'Analysis failed: {e}')
        Analyze = ipw.Button(description="Analyze", button_style='success')
        Analyze.on_click(Analyze_Clicked)
        
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
                self.integratedValues = at.Integrate(spectra, [IntegrationLow.value, IntegrationHigh.value], TitrantFallback.value)
                display(IntegratedToClipboard)
        Integrate = ipw.Button(description="Integrate")
        Integrate.on_click(Integrate_clicked)
        
        def Fit_clicked(b):
            spectra = self.spectra.copy(deep=True)
            with out3 :
                clear_output()
                limits = [LowLim.value, UpLim.value]
                self.fitAnalysis = at.Fit(spectra, limits, method=FitMethod.value, r2_threshold=R2Threshold.value, fallback_values=TitrantFallback.value)
                display(FitAnalysisToClipboard)
        Fit = ipw.Button(description="Fit")
        Fit.on_click(Fit_clicked)

        def CopyAnalysisData_Clicked(b):
            with out3:
                try:
                    if not hasattr(self, 'anisotropyFits') or not hasattr(self, 'anisotropyResults'):
                        print('Run Analyze before copying analysis data.')
                        return

                    fits = self.anisotropyFits.copy()
                    lam = fits.groupby(
                        ['Experiment','Titrant (uL)','Polarization']
                    )['Lambda max (nm)'].agg(
                        ['mean','std','count']).reset_index()
                    lam['SEM'] = lam['std'] / np.sqrt(lam['count'])

                    # Always include the core columns. Add replicate statistics only
                    # when at least one titrant point actually has replicate measurements.
                    has_replicates = bool((lam['count'] > 1).any())

                    vv = lam[lam['Polarization'] == 'VV'].copy()
                    vh = lam[lam['Polarization'] == 'VH'].copy()
                    if has_replicates:
                        vv = vv.rename(columns={'mean':'Lambda max VV (nm)', 'std':'Lambda max VV SD (nm)',
                                                'SEM':'Lambda max VV SEM (nm)', 'count':'VV n'})
                        vh = vh.rename(columns={'mean':'Lambda max VH (nm)', 'std':'Lambda max VH SD (nm)',
                                                'SEM':'Lambda max VH SEM (nm)', 'count':'VH n'})
                        vv = vv[['Experiment','Titrant (uL)','Lambda max VV (nm)','Lambda max VV SD (nm)','Lambda max VV SEM (nm)','VV n']]
                        vh = vh[['Experiment','Titrant (uL)','Lambda max VH (nm)','Lambda max VH SD (nm)','Lambda max VH SEM (nm)','VH n']]
                    else:
                        vv = vv.rename(columns={'mean':'Lambda max VV (nm)'})[['Experiment','Titrant (uL)','Lambda max VV (nm)']]
                        vh = vh.rename(columns={'mean':'Lambda max VH (nm)'})[['Experiment','Titrant (uL)','Lambda max VH (nm)']]

                    table = pd.merge(vv, vh, on=['Experiment','Titrant (uL)'], how='outer')
                    anis = self.anisotropyResults.reset_index().copy()
                    if has_replicates:
                        keep = [c for c in ['Experiment','Titrant (uL)','Anisotropy mean',
                                            'Anisotropy SD','Anisotropy SEM','n pairs'] if c in anis.columns]
                    else:
                        # For a single VV/VH pair, present the anisotropy simply as r.
                        source = 'Anisotropy mean' if 'Anisotropy mean' in anis.columns else 'Anisotropy r'
                        anis['Anisotropy'] = anis[source]
                        keep = [c for c in ['Experiment','Titrant (uL)','Anisotropy'] if c in anis.columns]
                    anis = anis[keep]
                    table = pd.merge(table, anis, on=['Experiment','Titrant (uL)'], how='outer')
                    table = table.sort_values(['Experiment','Titrant (uL)']).reset_index(drop=True)

                    self.analysisTable = table
                    display(table)
                    try:
                        table.to_clipboard(index=False)
                        print('Lambda max and anisotropy data copied to the clipboard.')
                    except Exception:
                        print('The combined table is shown above. Clipboard access is not available in this environment.')
                except Exception as e:
                    print(f'Copy failed: {e}')
        CopyAnalysisData = ipw.Button(description='Copy Analysis Data', button_style='info')
        CopyAnalysisData.on_click(CopyAnalysisData_Clicked)

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
        
        LowLim = ipw.FloatSlider(
            value=0.0, min=-1.0e9, max=1.0e9, step=1.0,
            description='Fit Lower Limit:', disabled=False,
            continuous_update=False, orientation='horizontal', readout=True,
            readout_format='.1f', layout=Layout(width='650px'),
            style={'description_width':'150px'})

        UpLim = ipw.FloatSlider(
            value=0.0, min=-1.0e9, max=1.0e9, step=1.0,
            description='Fit Upper Limit:', disabled=False,
            continuous_update=False, orientation='horizontal', readout=True,
            readout_format='.1f', layout=Layout(width='650px'),
            style={'description_width':'150px'})

        IntegrationLow = ipw.FloatSlider(
            value=0.0, min=-1.0e9, max=1.0e9, step=1.0,
            description='Integration Lower Limit:', continuous_update=False,
            readout_format='.1f', layout=Layout(width='650px'),
            style={'description_width':'180px'})
        IntegrationHigh = ipw.FloatSlider(
            value=0.0, min=-1.0e9, max=1.0e9, step=1.0,
            description='Integration Upper Limit:', continuous_update=False,
            readout_format='.1f', layout=Layout(width='650px'),
            style={'description_width':'180px'})

        # Browser-native file chooser. Multiple fluorescence files can be kept
        # in memory and switched without reloading them from disk.
        self.experiments = {}
        DataFileUpload = ipw.FileUpload(
            accept='.gxz,.txt,.csv,.dat', multiple=True,
            description='Open Data Files',
            layout=Layout(width='180px'))
        LoadedDataFile = ipw.SelectMultiple(
            options=[],
            rows=8,
            value=(),
            description='Data Files:',
            layout=Layout(
                width='700px',
                height='150px'
            ),
            style={
                'description_width':
                '100px'
            }
        )
        ClearLoadedFiles = ipw.Button(description='Clear Files', layout=Layout(width='120px'))
        UploadStatus = ipw.HTML(value='<i>No data files loaded.</i>')

        def _uploaded_entries(value):
            """Normalize ipywidgets 7/8 FileUpload values."""
            if isinstance(value, dict):
                return [(name, item.get('content', b'')) for name, item in value.items()]
            entries = []
            for item in (value or []):
                name = item.get('name', 'data')
                entries.append((name, item.get('content', b'')))
            return entries

        def _load_uploaded_bytes(name, content):
            """Load one uploaded object with the existing GXZ/text parsers."""
            raw = content.tobytes() if hasattr(content, 'tobytes') else bytes(content)
            suffix = Path(name).suffix.lower()
            # Existing loaders are file based, so use a short-lived temporary file.
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / Path(name).name
                path.write_bytes(raw)
                if suffix == '.gxz':
                    return dt.loadGXZ(str(path.parent), path.name)
                return dt.loadText(str(path.parent), path.name)

        def _activate_loaded_file(names):
            if not names:
                return
            selected = list(names)
            combined = []
            experiment_dates = []
            self.datasetMap = {}
            for dataset_num, filename in enumerate(selected, start=1):
                experiment = self.experiments[filename]
                experiment_dates.append(
                    experiment['date']
                )
                self.datasetMap[f'D{dataset_num}'] = {
                    'file': filename,
                    'date': experiment['date'],
                    'experiment': at.ExtractExperimentFromFilename(
                        filename
                    )
                }
                tmp = experiment['data'].copy()
                tmp.columns = [
                    f'{col}_D{dataset_num}'
                    for col in tmp.columns
                ]
                combined.append(tmp)
            self.experimentDates = (
                sorted(
                    set(
                        experiment_dates
                    )
                )
            )
            self.loadedExperimentNames = (
                selected
            )
            self.filename = (
                '; '.join(
                    selected
                )
            )
            self.data = pd.concat(
                combined,
                axis=1
            )
            Buffer.value = 'None'
            Runs_Selected.value = ()
            RunList()
            UpdatePlot()

        def FilesUploaded(change):
            entries = _uploaded_entries(
                change['new']
            )
            if not entries:
                return
            errors = []
            for name, content in entries:
                try:
                    data = _load_uploaded_bytes(
                        name,
                        content
                    )
                    self.experiments[
                        name
                    ] = {
                        'data':
                            data,
                        'date':
                            at.ExtractDateFromFilename(
                                name
                            ),
                        'file':
                            name
                    }
                except Exception as exc:
                    errors.append(
                        f'{name}: {exc}'
                    )
            names = sorted(
                self.experiments.keys()
            )
            display_names = [
                f'D{i+1}: {name}'
                for i, name in enumerate(names)
            ]
            LoadedDataFile.options = [
                (display_names[i], names[i])
                for i in range(len(names))
            ]
            LoadedDataFile.options = names
            if len(names):
                LoadedDataFile.value = tuple(
                    names
                )
            UploadStatus.value = (
                f'<b>{len(names)}</b> '
                'file(s) loaded.'
            )

        def DataFileChanged(change):
            if change['name'] == 'value':
                _activate_loaded_file(
                    change['new']
                )   

        def ClearFilesClicked(b):
            self.experiments.clear()
            LoadedDataFile.options = []
            LoadedDataFile.value = None
            DataFileUpload.value = () if not isinstance(DataFileUpload.value, dict) else {}
            UploadStatus.value = '<i>No data files loaded.</i>'
            if hasattr(self, 'data'):
                del self.data
            Runs_Selected.options = []
            Buffer.options = ['None']
            with out1: clear_output(wait=True)
            with out2: clear_output(wait=True)
            with out3: clear_output(wait=True)

        DataFileUpload.observe(FilesUploaded, names='value')
        LoadedDataFile.observe(DataFileChanged, names='value')
        ClearLoadedFiles.on_click(ClearFilesClicked)

        display(ipw.HBox([DataFileUpload, LoadedDataFile, ClearLoadedFiles]))
        display(UploadStatus)

        with out1:
            display(ipw.Box([Filter,Update_RunList]))
            display(ipw.Box([TitrationFile,LoadTitration]))
            display(ipw.Box([CamSubtraction,DilutionCorrection,CuvetteVolume,NormalizeDisplay]))
            display(Runs_Selected)
            display(ipw.Box([Buffer]))

        display(out1)
        display(out2)
        display(out3)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)
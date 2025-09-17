import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import os, glob, pandas as pd, numpy as np
from math import cos, radians
import geopandas as gpd

# --- File/Dir ---
BASE_DIR = os.getcwd()
BASIN_DIR = os.path.join(BASE_DIR, 'basins')

# ---------- File Finding Helpers ----------
def _first_existing(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[-1]
    return None

def find_nc_file(basin_name, variable_type):
    netcdf_dir = os.path.join(BASIN_DIR, basin_name, 'NetCDF')
    if not os.path.isdir(netcdf_dir): return None
    if variable_type == 'P':
        pats = [os.path.join(netcdf_dir, '*_P_*.nc'), os.path.join(netcdf_dir, '*P*.nc')]
    elif variable_type == 'ET':
        pats = [os.path.join(netcdf_dir, '*_ETa_*.nc'), os.path.join(netcdf_dir, '*_ET_*.nc'), os.path.join(netcdf_dir, '*ET*.nc')]
    elif variable_type == 'LU':
        pats = [os.path.join(netcdf_dir, '*_LU_*.nc'), os.path.join(netcdf_dir, '*LandUse*.nc'), os.path.join(netcdf_dir, '*LU*.nc')]
    else:
        return None
    return _first_existing(pats)

def find_shp_file(basin_name):
    shp_dir = os.path.join(BASIN_DIR, basin_name, 'Shapefile')
    if not os.path.isdir(shp_dir): return None
    pats = [os.path.join(shp_dir, '*.shp')]
    return _first_existing(pats)

# ---------- Data Processing Helpers ----------
def _standardize_latlon(ds):
    lat_names = ['latitude', 'lat', 'y']
    lon_names = ['longitude', 'lon', 'x']
    lat = next((n for n in lat_names if n in ds.coords or n in ds.variables), None)
    lon = next((n for n in lon_names if n in ds.coords or n in ds.variables), None)
    if lat and lat != 'latitude': ds = ds.rename({lat: 'latitude'})
    if lon and lon != 'longitude': ds = ds.rename({lon: 'longitude'})
    return ds

def _pick_data_var(ds):
    exclude = {'time', 'latitude', 'longitude', 'crs', 'spatial_ref'}
    cands = [v for v in ds.data_vars if v not in exclude]
    if not cands: return None
    with_ll = [v for v in cands if {'latitude','longitude'}.issubset(set(ds[v].dims))]
    return with_ll[0] if with_ll else cands[0]

def _compute_mode(arr, axis=None):
    vals, counts = np.unique(arr, return_counts=True)
    if len(counts) == 0: return np.nan
    return vals[np.argmax(counts)]

def _coarsen_to_1km(da, is_categorical=False):
    if 'latitude' not in da.dims or 'longitude' not in da.dims: return da
    lat_vals, lon_vals = da['latitude'].values, da['longitude'].values
    lat_res = float(np.abs(np.diff(lat_vals)).mean()) if lat_vals.size > 1 else 0.009
    lon_res = float(np.abs(np.diff(lon_vals)).mean()) if lon_vals.size > 1 else 0.009
    mean_lat = float(lat_vals.mean()) if lat_vals.size else 7.5
   
    target_deg = 1.0 / 111.0
    f_lat = max(1, int(round(target_deg / lat_res))) if lat_res > 0 else 1
    f_lon = max(1, int(round(target_deg / lon_res))) if lon_res > 0 else 1
   
    coarsen_dict = {'latitude': f_lat, 'longitude': f_lon}
   
    if is_categorical:
        return da.coarsen(coarsen_dict, boundary='trim').reduce(_compute_mode)
    else:
        return da.coarsen(coarsen_dict, boundary='trim').mean(skipna=True)

def load_and_process_data(basin_name, variable_type, year=None, aggregate_time=True):
    fp = find_nc_file(basin_name, variable_type)
    if not fp: return None, None, "NetCDF file not found"
    try:
        ds = xr.open_dataset(fp, decode_times=True)
        ds = _standardize_latlon(ds)
        var = _pick_data_var(ds)
        if not var: return None, None, "No suitable data variable in file"
        da = ds[var]
       
        # Time selection
        if 'time' in ds.coords and year:
            target_start = pd.to_datetime(f"{year}-01-01")
            target_end = pd.to_datetime(f"{year}-12-31")
            da = da.sel(time=slice(target_start, target_end))
       
        # Aggregation
        if 'time' in da.dims and da.sizes.get('time', 0) > 1 and aggregate_time:
            if variable_type in ['P', 'ET']:
                da = da.sum(dim='time', skipna=True)
        elif 'time' in da.dims and not aggregate_time:
            pass  # Return time series
        elif 'time' in da.dims:
            da = da.isel(time=0)
        da_1km = _coarsen_to_1km(da, is_categorical=(variable_type == 'LU'))
        return da_1km, var, os.path.basename(fp)
    except Exception as e:
        return None, None, f"Error processing file: {e}"

# ---------- Plotting Helpers ----------
def add_shapefile_to_fig(fig, basin_name):
    """Adds a basin boundary outline to a Plotly figure."""
    shp_file = find_shp_file(basin_name)
    if shp_file:
        gdf = gpd.read_file(shp_file)
        gdf = gdf.to_crs("EPSG:4326")  # Ensure WGS84
        for geom in gdf.geometry:
            if geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines',
                                         line=dict(color='black', width=2),
                                         name='Basin Boundary', showlegend=False))
    return fig

def create_empty_fig(message="No data to display"):
    """Creates a blank figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': message, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
    )
    return fig

# ---------- Dash App ----------
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --- Layout ---
basin_folders = [d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))] if os.path.isdir(BASIN_DIR) else []
basin_options = [{'label': b, 'value': b} for b in sorted(basin_folders)]

app.layout = html.Div([
    html.H1("Basin Data Dashboard", style={'textAlign': 'center'}),
   
    html.Div([
        html.H3("1. Select Basin"),
        dcc.Dropdown(id='basin-dropdown', options=basin_options, value=(basin_options[0]['value'] if basin_options else None), clearable=False),
        html.P(id='file-info-feedback', style={'fontSize': 12, 'color': '#666', 'marginTop': 10})
    ], style={'width': '80%', 'margin': 'auto', 'padding': '10px'}),
   
    html.Hr(),
   
    html.Div([
        html.H2("Land Use / Land Cover", style={'textAlign': 'center'}),
        html.Div([
            html.H4("Select Year", style={'textAlign': 'center'}),
            dcc.Slider(id='lu-year-slider', min=1990, max=2025, step=1, value=2020, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'width': '60%', 'margin': 'auto'}),
        dcc.Loading(dcc.Graph(id='lu-map-graph', style={'height': '70vh'}))
    ], className='section-container'),
   
    html.Hr(),
   
    html.Div([
        html.H2("Precipitation (P)", style={'textAlign': 'center'}),
        html.Div([
            html.H4("Select Year", style={'textAlign': 'center'}),
            dcc.Slider(id='p-year-slider', min=1990, max=2025, step=1, value=2020, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'width': '60%', 'margin': 'auto'}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='p-map-graph')), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Loading(dcc.Graph(id='p-bar-graph')), style={'width': '50%', 'display': 'inline-block'})
        ], style={'height': '60vh'})
    ], className='section-container'),
   
    html.Hr(),
   
    html.Div([
        html.H2("Evapotranspiration (ET)", style={'textAlign': 'center'}),
        html.Div([
            html.H4("Select Year", style={'textAlign': 'center'}),
            dcc.Slider(id='et-year-slider', min=1990, max=2025, step=1, value=2020, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'width': '60%', 'margin': 'auto'}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='et-map-graph')), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Loading(dcc.Graph(id='et-bar-graph')), style={'width': '50%', 'display': 'inline-block'})
        ], style={'height': '60vh'})
    ], className='section-container'),
   
    html.Hr(),
   
    html.Div([
        html.H2("Water Balance (P - ET)", style={'textAlign': 'center'}),
        html.Div([
            html.H4("Select Year", style={'textAlign': 'center'}),
            dcc.Slider(id='p-et-year-slider', min=1990, max=2025, step=1, value=2020, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'width': '60%', 'margin': 'auto'}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='p-et-map-graph')), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Loading(dcc.Graph(id='p-et-bar-graph')), style={'width': '50%', 'display': 'inline-block'})
        ], style={'height': '60vh'})
    ], className='section-container'),
], style={'fontFamily': 'Arial, sans-serif'})

# --- Callbacks ---
@app.callback(
    [Output('lu-year-slider', 'min'),
     Output('lu-year-slider', 'max'),
     Output('lu-year-slider', 'value'),
     Output('p-year-slider', 'min'),
     Output('p-year-slider', 'max'),
     Output('p-year-slider', 'value'),
     Output('et-year-slider', 'min'),
     Output('et-year-slider', 'max'),
     Output('et-year-slider', 'value'),
     Output('p-et-year-slider', 'min'),
     Output('p-et-year-slider', 'max'),
     Output('p-et-year-slider', 'value'),
     Output('file-info-feedback', 'children')],
    [Input('basin-dropdown', 'value')]
)
def init_controls(basin):
    if not basin: return [1990, 2025, 2020] * 4 + ["Please select a basin."]
   
    p_fp = find_nc_file(basin, 'P')
    et_fp = find_nc_file(basin, 'ET')
    lu_fp = find_nc_file(basin, 'LU')
   
    p_min_yr, p_max_yr, p_val = 1990, 2025, 2020
    et_min_yr, et_max_yr, et_val = 1990, 2025, 2020
    p_et_min_yr, p_et_max_yr, p_et_val = 1990, 2025, 2020
    lu_min_yr, lu_max_yr, lu_val = 1990, 2025, 2020
   
    if p_fp:
        with xr.open_dataset(p_fp) as ds:
            if 'time' in ds.coords and ds.sizes.get('time', 0) > 0:
                times = pd.to_datetime(ds['time'].values)
                p_min_yr = times.min().year
                p_max_yr = p_val = times.max().year
    if et_fp:
        with xr.open_dataset(et_fp) as ds:
            if 'time' in ds.coords and ds.sizes.get('time', 0) > 0:
                times = pd.to_datetime(ds['time'].values)
                et_min_yr = times.min().year
                et_max_yr = et_val = times.max().year
    if p_fp and et_fp:
        with xr.open_dataset(p_fp) as ds_p, xr.open_dataset(et_fp) as ds_et:
            if 'time' in ds_p.coords and 'time' in ds_et.coords:
                times_p = pd.to_datetime(ds_p['time'].values)
                times_et = pd.to_datetime(ds_et['time'].values)
                p_et_min_yr = max(times_p.min().year, times_et.min().year)
                p_et_max_yr = p_et_val = min(times_p.max().year, times_et.max().year)
    if lu_fp:
        with xr.open_dataset(lu_fp) as ds:
            if 'time' in ds.coords and ds.sizes.get('time', 0) > 0:
                times = pd.to_datetime(ds['time'].values)
                lu_min_yr = times.min().year
                lu_max_yr = lu_val = times.max().year
   
    files_found = f"P file: {os.path.basename(p_fp if p_fp else 'Not Found')} | ET file: {os.path.basename(et_fp if et_fp else 'Not Found')} | LU file: {os.path.basename(lu_fp if lu_fp else 'Not Found')}"
    return (lu_min_yr, lu_max_yr, lu_val,
            p_min_yr, p_max_yr, p_val,
            et_min_yr, et_max_yr, et_val,
            p_et_min_yr, p_et_max_yr, p_et_val,
            files_found)

def create_hydrology_outputs(basin, year, vtype):
    if not (basin and year):
        return create_empty_fig("Missing selections."), create_empty_fig()
   
    da_ts, _, fname = load_and_process_data(basin, vtype, year=year, aggregate_time=False)
    if da_ts is None:
        return create_empty_fig(f"{vtype} data not found."), create_empty_fig()
   
    # --- Map Figure (Temporal Sum) ---
    da_map = da_ts.sum(dim='time', skipna=True)
    map_title = f"Total {vtype} ({year})"
    colorscale = 'Blues' if vtype == 'P' else 'YlOrRd'
   
    fig_map = px.imshow(da_map.values, x=da_map['longitude'], y=da_map['latitude'],
                        color_continuous_scale=colorscale, origin='lower', aspect='equal',
                        title=map_title, labels={'color': 'mm'})
    fig_map = add_shapefile_to_fig(fig_map, basin)
   
    # --- Bar Chart (Monthly Average) ---
    fig_bar = create_empty_fig(f"No monthly data for {vtype}.")
    if da_ts.time.size > 0:
        spatial_dims = [d for d in ['latitude', 'longitude'] if d in da_ts.dims]
        spatial_mean_ts = da_ts.mean(dim=spatial_dims, skipna=True)
       
        with np.errstate(invalid='ignore'):
            monthly_data = spatial_mean_ts.groupby('time.month').mean(skipna=True).rename({'month': 'Month'})
       
        if monthly_data.notnull().any():
            month_names = [pd.to_datetime(m, format='%m').strftime('%b') for m in monthly_data['Month'].values]
            y_values = np.asarray(monthly_data.values).flatten()
            fig_bar = px.bar(x=month_names, y=y_values,
                             title=f"Mean Monthly {vtype} ({year})",
                             labels={'x': 'Month', 'y': f'Mean Daily {vtype} (mm)'})
        else:
            fig_bar = create_empty_fig(f"No valid data for {vtype} in this period.")
    return fig_map, fig_bar

@app.callback(
    [Output('p-map-graph', 'figure'), Output('p-bar-graph', 'figure')],
    [Input('basin-dropdown', 'value'), Input('p-year-slider', 'value')]
)
def update_p_outputs(basin, year):
    return create_hydrology_outputs(basin, year, 'P')

@app.callback(
    [Output('et-map-graph', 'figure'), Output('et-bar-graph', 'figure')],
    [Input('basin-dropdown', 'value'), Input('et-year-slider', 'value')]
)
def update_et_outputs(basin, year):
    return create_hydrology_outputs(basin, year, 'ET')

@app.callback(
    Output('lu-map-graph', 'figure'),
    [Input('basin-dropdown', 'value'), Input('lu-year-slider', 'value')]
)
def update_lu_map(basin, year):
    if not (basin and year): return create_empty_fig("Select Basin and Year")
   
    da, _, fname = load_and_process_data(basin, 'LU', year=year)
    if da is None: return create_empty_fig(f"Land Use data not found for {year}")
    title = f"Land Use / Cover for {year}"
    classes = np.unique(da.values[np.isfinite(da.values)]).astype(int)
   
    fig = px.imshow(da.values, x=da['longitude'], y=da['latitude'],
                    color_continuous_scale='Viridis', origin='lower', aspect='equal', title=title)
   
    if len(classes) > 0:
        fig.update_coloraxes(colorbar=dict(tickmode='array', tickvals=classes, ticktext=[str(c) for c in classes]))
    fig = add_shapefile_to_fig(fig, basin)
    return fig

@app.callback(
    [Output('p-et-map-graph', 'figure'), Output('p-et-bar-graph', 'figure')],
    [Input('basin-dropdown', 'value'), Input('p-et-year-slider', 'value')]
)
def update_p_et_outputs(basin, year):
    if not (basin and year):
        return create_empty_fig("Missing selections."), create_empty_fig()
   
    da_p_ts, _, _ = load_and_process_data(basin, 'P', year=year, aggregate_time=False)
    da_et_ts, _, _ = load_and_process_data(basin, 'ET', year=year, aggregate_time=False)
   
    if da_p_ts is None or da_et_ts is None:
        return create_empty_fig("P or ET data missing."), create_empty_fig()
    da_p_aligned, da_et_aligned = xr.align(da_p_ts, da_et_ts, join='inner')
    if da_p_aligned.time.size == 0:
        return create_empty_fig("No overlapping time steps for P and ET."), create_empty_fig()
       
    da_p_et_ts = da_p_aligned - da_et_aligned
   
    # --- Map Figure (P-ET Sum) ---
    da_map = da_p_et_ts.sum(dim='time', skipna=True)
    map_title = f"Total Water Balance (P-ET) ({year})"
   
    fig_map = px.imshow(da_map.values, x=da_map['longitude'], y=da_map['latitude'],
                        color_continuous_scale='RdBu', origin='lower', aspect='equal',
                        title=map_title, labels={'color': 'mm'})
    fig_map = add_shapefile_to_fig(fig_map, basin)
   
    # --- Bar Chart (Monthly Average P-ET) ---
    fig_bar = create_empty_fig("No monthly data for P-ET.")
    if da_p_et_ts.time.size > 0:
        spatial_dims = [d for d in ['latitude', 'longitude'] if d in da_p_et_ts.dims]
        spatial_mean_p_et_ts = da_p_et_ts.mean(dim=spatial_dims, skipna=True)
        with np.errstate(invalid='ignore'):
            monthly_data = spatial_mean_p_et_ts.groupby('time.month').mean(skipna=True).rename({'month': 'Month'})
        if monthly_data.notnull().any():
            month_names = [pd.to_datetime(m, format='%m').strftime('%b') for m in monthly_data['Month'].values]
            y_values = np.asarray(monthly_data.values).flatten()
            fig_bar = px.bar(x=month_names, y=y_values,
                             title=f"Mean Monthly Water Balance (P-ET) ({year})",
                             labels={'x': 'Month', 'y': 'Mean Daily P-ET (mm)'})
            fig_bar.update_traces(marker_color=['red' if v < 0 else 'blue' for v in y_values])
        else:
            fig_bar = create_empty_fig(f"No valid data for P-ET in this period.")
           
    return fig_map, fig_bar

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
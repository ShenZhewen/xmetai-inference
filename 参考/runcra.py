import argparse
import os
import time 
import numpy as np
import xarray as xr
import pandas as pd
"""
# ---------------------- 配置你的路径 ----------------------
ORT_ROOT = "/workspace/onnxruntime-linux-x64-gpu-1.18.1"
ORT_LIB = os.path.join(ORT_ROOT, "lib")
ORT_PY = os.path.join(ORT_ROOT, "python")
# -----------------------------------------------------------

# 1. 强制动态库优先走这里（必须在 import onnxruntime 之前）
os.environ["LD_LIBRARY_PATH"] = ORT_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 2. 强制 Python 优先从这里 import onnxruntime
sys.path.insert(0, ORT_PY)
"""
import ctypes
import os
import onnxruntime as ort

"""
ort_capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
os.environ['LD_LIBRARY_PATH'] = f"{ort_capi_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    ctypes.CDLL(os.path.join(ort_capi_dir, 'libonnxruntime.so.1.23.2'), mode=ctypes.RTLD_GLOBAL)
    print("Pre-loaded libonnxruntime.so.1.23.2")
except Exception as e:
    print(f"Could not pre-load: {e}")

print(ort.get_available_providers())
"""
from copy import deepcopy
from data_util import make_input, print_dataarray
import sys




def save_with_progress(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    ds = ds.astype(dtype)
    obj = ds.to_netcdf(save_name, compute=False)

    with ProgressBar():
        obj.compute()


def save_like(output, input, member, lead_time, save_dir=""):

    if save_dir:
        save_dir = os.path.join(save_dir, f"member/{member:02d}")
        os.makedirs(save_dir, exist_ok=True)
        init_time = pd.to_datetime(input.time.data[-1])

        ds = xr.DataArray(
            data=output,
            dims=['time', 'lead_time', 'level', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                lead_time=[lead_time],
                level=input.level,
                lat=input.lat,
                lon=input.lon,
            )
        ).astype(np.float32)
        #print(ds)
        #import sys
        #sys.exit()
        #print_dataarray(ds)
        save_name = os.path.join(save_dir, f'{lead_time:02d}.nc')
        #selected_channels = ['t2m',  'msl', 'tp','z500','z200','sst','z850']
        #selected_channels = ['z200', 'z500', 'z850', 't200', 't500', 't850', 'u200', 'u500', 'u850', 'v200', 'v500', 'v850', 'q200', 'q500', 'q700', 'q850', 'msl', 'sst', 'u10m', 'v10m', 'u100m', 'v100m', 'tcc', 'ssr', 'ssrd', 'fdir', 'ttr']
        #selected_channels = selected_channels = ['t2m', 't2m_min', 't2m_max',  'tp']
        """
        selected_channels =  ['z850', 'z500', 'z200',
             't850', 't500', 't200',
             'u850', 'u500', 'u200',
             'v850', 'v500', 'v200',
             'q850', 'q500', 'q700',
             't2m', 'd2m', 'sst', 'ttr', 'u10m', 'v10m', 'u100m', 'v100m', 'msl', 'tcw', 'tp','t2m_min', 't2m_max']
        """
        #selected_channels =['z200', 'z500', 'z850', 'u200', 'u500', 'u850', 'v200', 'v500', 'v850',  'q500', 'q700', 'q850', 'msl', 't2m', 't2m_min', 't2m_max', 'sst', 'u10m', 'v10m', 'u100m', 'v100m', 'tcc',  'ssrd', 'ttr',  'tp']
        selected_channels =["msl", "sst", "t2m", "t2m_min", "t2m_max", "tp", "ttr", "u200", "u500", "u850", "v200", "v500", "v850", "z200", "z500", "z850"]
        #selected_channels = ['z850', 'z500', 'z250', 't850', 't500', 't250', 
        #                    'u850', 'u500', 'u250', 'v850', 'v500', 'v250', 
        #                    'q850', 'q500', 'q250', 't2m', 'd2m', 'sst', 
        #                    'ttr', '10u', '10v', '100u', '100v', 'msl', 
        #                    'tcwv', 'tp']
        da = ds.sel(level=selected_channels)
        #da = ds
        #print(da)
        # Handle string datatypes which are not compatible with netCDF4
        
        for var_name in da.coords:
            if hasattr(ds[var_name], 'dtype') and str(da[var_name].dtype).startswith('object'):
                #logger.info(f"Converting string dtype in coordinate {var_name} to object dtype")
                da = da.assign_coords({var_name: da[var_name].astype(str)})
        
        #print(save_name)
        #print(da)
        #sys.exit()
        da.to_netcdf(save_name)

# def load_model(model_name, device,device_id=0):
#     ort.set_default_logger_severity(3)
#     options = ort.SessionOptions()
#     options.enable_cpu_mem_arena=False
#     options.enable_mem_pattern = False
#     options.enable_mem_reuse = False
    
#     if device == "cuda":
#         providers =  [
#             ('CUDAExecutionProvider', {
#                 'device_id': device_id,  # 指定 CUDA 设备 ID
#                 'arena_extend_strategy': 'kSameAsRequested',
#                 'cudnn_conv_algo_search': 'EXHAUSTIVE',
#                 'do_copy_in_default_stream': True
 
#             })
#         ]

#     else:
#         raise ValueError("device must be cpu or cuda!")

#     session = ort.InferenceSession(
#         model_name,  
#         sess_options=options, 
#         providers=providers
#     )
#     inp = session.get_inputs()[0]
#     #print("name:", inp.name)
#     #print("shape:", inp.shape)     # 例如 ['batch', 2, 'height', 'width']
#     #print("dtype:", inp.type)
#     #import sys
#     #sys.exit()
#     return session

def load_model(model_name, device, gpu_id=0):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False

    if device == "cuda":
        providers = [('CUDAExecutionProvider', {
            'arena_extend_strategy': 'kSameAsRequested',
            'device_id': gpu_id  # 指定GPU设备ID
        })]
    elif device == "cpu":
        providers = ['CPUExecutionProvider']
        options.intra_op_num_threads = 30
    else:
        raise ValueError("device must be cpu or cuda!")
    
    #print('lib-------------------》', flush=True)
    #options.register_custom_ops_library('/gpu/zhouchg/FUXI_S2S/chenl_onnx/xmetai_onnx_plugins.cpython-311-x86_64-linux-gnu (2).so')
    #print('lib ok----------------》', flush=True)
    # 开启最详细的日志（0=VERBOSE，1=INFO，2=WARNING，3=ERROR）
    """
    ort.set_default_logger_severity(0)
    ort.set_default_logger_verbosity(999)

    options = ort.SessionOptions()
    options.log_severity_level = 0
    options.log_verbosity_level = 999
    """

    import time
    import random
    # 每个进程随机等 0–2 秒，错开初始化高峰
    time.sleep(random.uniform(0, 2))
    print('lib-------------------》', flush=True)
    options.register_custom_ops_library(
        '/gpu/zhouchg/FUXI_S2S/chenl_onnx/xmetai_onnx_plugins_gcc9_cuda12_ort1.24.4.so'
    )
    print('lib ok----------------》', flush=True)
    
    session = ort.InferenceSession(
        model_name,
        sess_options=options,
        providers=providers
    )
    print('session ok--------------------》', flush=True)
    return session


def run_inference(
    model, 
    input, 
    total_step, 
    total_member, 
    save_dir=""
):
    input_names = [input.name for input in model.get_inputs()]
    #hist_time = pd.to_datetime(input.time.values[-2])
    init_time = pd.to_datetime(input.time.values[-1])
    #assert init_time - hist_time == pd.Timedelta(days=1)
    
    lat = input.lat.values 
    lon = input.lon.values 
    batch = input.values[None]
    
    assert lat[0] == 90 and lat[-1] == -90
    #print(f'Model initial Time: {init_time.strftime(("%Y%m%d%H"))}')
    #print(f"Region: {lat[0]:.2f} ~ {lat[-1]:.2f}, {lon[0]:.2f} ~ {lon[-1]:.2f}")

    for member in range(total_member):
        #print(f'Inference member {member:02d} ...')
        new_input = deepcopy(batch)

        start = time.perf_counter()
        for step in range(total_step):
            lead_time = (step + 1)

            inputs = {'input': new_input}        

            if "step" in input_names:
                inputs['step'] = np.array([step], dtype=np.float32)

            if "doy" in input_names:
                valid_time = init_time + pd.Timedelta(days=step)
                doy = min(365, valid_time.day_of_year)/365 
                inputs['doy'] = np.array([doy], dtype=np.float32)

            if True:
                #some need hour
                inputs['hour'] = np.array([0], dtype=np.float32)

            #inputs['perturb_std_scale'] = np.array([1,1,1], dtype=np.float32)

            istart = time.perf_counter()
            #print(inputs['input'].min(),inputs['input'].max())
            #print(step,inputs['input'].min(),inputs['input'].max())
            new_input = model.run(None, inputs)
            #print(len(new_input),new_input[0].shape,new_input[0].min(),new_input[0].max())
            #print(len(new_input))
            #print(new_input[0].shape,new_input[1].shape)
            #sys.exit()
            #new_input = new_input[0]
            #np.save('fuckerror.npy',new_input[0])
            #sys.exit()
            #print(new_input[0].shape)
            output = new_input[0][:, -1:]
            #np.save("out.npy", output)
            #sys.exit()
            #new_input = new_input[0]
            #print(new_input.dtype)
            #sys.exit()
            #print(new_input.shape)
            #print(inputs['input'].shape)
            #sys.exit()
            new_input = np.concatenate((inputs['input'], new_input[0][:, -1:]), axis=1)
            new_input = new_input[:,-2:]
            #print(output.dtype,output.shape)
            #import sys
            #sys.exit()
            step_time = time.perf_counter() - istart

            #print(f"member: {member:02d}, step {step+1:02d}, step_time: {step_time:.3f} sec")
            save_like(output, input, member, lead_time, save_dir)
            #time.sleep(0.5)
            
            if step > total_step:
                break

        run_time = time.perf_counter() - start
        #print(f'Inference member done, take {run_time:.2f} sec')


def land_to_nan(input, mask, names=['sst']):
    channel = input.channel.data.tolist()
    for ch in names:
        v = input.sel(channel=ch)
        v = v.where(mask)
        idx = channel.index(ch)
        input.data[:, idx] = v.data
    return input



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model/fuxi_s2s.onnx", help="FuXi-S2S onnx model file")
    parser.add_argument('--input', type=str, default="data/input.nc", help="The input netcdf data file")
    parser.add_argument('--device', type=str, default="cuda", help="The device to run FuXi model")
    parser.add_argument('--save_dir', type=str, default="output")
    parser.add_argument('--total_step', type=int, default=60)
    parser.add_argument('--total_member', type=int, default=100)
    args = parser.parse_args()

    if os.path.exists(args.input):
        input = xr.open_dataarray(args.input)
    else:
        input = make_input("data/sample")
        input.to_netcdf("data/input.nc")

    mask = xr.open_dataarray("data/mask.nc")
    # input = land_to_nan(input, mask)    
    print_dataarray(input)        

    print(f'Load FuXi ...')       
    start = time.perf_counter()
    model = load_model(args.model, args.device)
    input_names = [input.name for input in model.get_inputs()]
    print(f'Load FuXi take {time.perf_counter() - start:.2f} sec')

    run_inference(
        model, 
        input, 
        args.total_step, 
        args.total_member,  
        save_dir=args.save_dir
    )

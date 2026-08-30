import argparse
import os
import time 
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort
from copy import deepcopy
#from cra1_5_util import make_input, print_dataarray
import sys
import ctypes
import os
import onnxruntime 
from datetime import datetime, timedelta

ort_capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
#os.environ['LD_LIBRARY_PATH'] = f"{ort_capi_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
print(os.environ.get('LD_LIBRARY_PATH', ''))
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{ort_capi_dir}"

try:
    ctypes.CDLL(os.path.join(ort_capi_dir, 'libonnxruntime.so.1.24.4'), mode=ctypes.RTLD_LOCAL)
    print("Pre-loaded libonnxruntime.so.1.24.4")
except Exception as e:
    print(f"Could not pre-load: {e}")

import torch
# ====== 输入/输出根目录 ======
BASE_DIR = "/gpu/zhouchg/FUXI_S2S/guagnxi-fengshun2.0pre/fdp"


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
            dims=['time', 'lead_time', 'channel', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                lead_time=[lead_time],
                channel=input.channel,
                lat=input.lat,
                lon=input.lon,
            )
        ).astype(np.float32)

        save_name = os.path.join(save_dir, f'{lead_time:02d}.nc')

        # selected_channels =  ['z200', 'z500', 'z850', 't200', 't500', 't850', 'u200', 'u500', 'u850',
        #                         'v200', 'v500', 'v850', 'q200', 'q500', 'q700', 'q850', 'msl', 't2m', 
        #                         't2m_min', 't2m_max', 'sst', 'u10m', 'v10m', 'u100m', 'v100m', 'tcc', 
        #                         'ssr', 'ssrd', 'fdir', 'ttr', 'tp']

        selected_channels =  [ 't2m','tp']

        da = ds.sel(channel=selected_channels)

        for var_name in da.coords:
            if hasattr(ds[var_name], 'dtype') and str(da[var_name].dtype).startswith('object'):
                da = da.assign_coords({var_name: da[var_name].astype(str)})
                               
        da.to_netcdf(save_name)

def load_model(model_name, device, gpu_id=0):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False

    options.intra_op_num_threads = 2   # new
    options.inter_op_num_threads = 1   # new

    options.register_custom_ops_library('/gpu/zhouchg/FUXI_S2S/chenl_onnx/xmetai_onnx_plugins_gcc9_cuda12_ort1.24.4.so')


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

    session = ort.InferenceSession(
        model_name,
        sess_options=options,
        providers=providers
    )

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
    
    #assert lat[0] == 90 and lat[-1] == -90
    print(f'Model initial Time: {init_time.strftime(("%Y%m%d%H"))}')

    for member in range(total_member):
        print(f'Inference member {member:02d} ...')
        new_input = deepcopy(batch)

        start = time.perf_counter()
        for step in range(total_step):
            lead_time = (step + 1)

            inputs = {'input': new_input}        

            if "step" in input_names:
                inputs['step'] = np.array([step], dtype=np.float32)

            if "doy" in input_names:
                valid_time = init_time + pd.Timedelta(hours=step * 6)
                print(valid_time)
                doy = min(365, valid_time.day_of_year)/365 
                inputs['doy'] = np.array([doy], dtype=np.float32)

            if True:
                #inputs['hour'] = np.array([0], dtype=np.float32)
                hour_val = valid_time.hour
                print(hour_val)
                inputs['hour'] = np.array([hour_val], dtype=np.float32)

            #inputs['perturb_std_scale'] = np.array([1,1,1], dtype=np.float32)

            istart = time.perf_counter()
            new_input = model.run(None, inputs)
            output = new_input[0][:, -1:]
            new_input = np.concatenate((inputs['input'], new_input[0][:, -1:]), axis=1)
            new_input = new_input[:,-2:]
  
            step_time = time.perf_counter() - istart

            print(f"member: {member:02d}, step {step+1:02d}, step_time: {step_time:.3f} sec")
            save_like(output, input, member, lead_time, save_dir)
            
            if step > total_step:
                break

        run_time = time.perf_counter() - start
        print(f'Inference member done, take {run_time:.2f} sec')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="/gpu/douzsh/models/fdp_v1.5.onnx", help="FengShunV2.0 onnx model file")
    parser.add_argument('--input', type=str, default=None, help="Single input file (single-day mode)")
    parser.add_argument('--start', type=str, default=None, help="Loop start date (YYYYMMDD)")
    parser.add_argument('--end', type=str, default=None, help="Loop end date (YYYYMMDD)")
    parser.add_argument('--device', type=str, default="cuda", help="The device to run FengShunV2.0 model")
    parser.add_argument('--device_id', type=int, default=0, help="Which gpu to use")
    parser.add_argument('--save_dir', type=str, default=None, help="Single-day save dir override")
    parser.add_argument('--total_step', type=int, default=60)
    parser.add_argument('--total_member', type=int, default=100)
    args = parser.parse_args()

    # ====== 加载模型（只加载一次，循环复用）======
    print(f'Load FengShunV2.0-beta ...')       
    start = time.perf_counter()
    print(start)
    model = load_model(args.model, args.device)
    print(model)
    input_names = [input.name for input in model.get_inputs()]
    print(f'Load FengShunV2.0 take {time.perf_counter() - start:.2f} sec')

    # ====== 判断运行模式 ======
    if args.start and args.end:
        # ====== 循环模式：从 start 到 end 逐日推理 ======
        start_date = datetime.strptime(args.start, '%Y%m%d%H')
        end_date = datetime.strptime(args.end, '%Y%m%d%H')
        current = start_date

        while current <= end_date:
            date_str = current.strftime('%Y%m%d%H')
            input_file = os.path.join(BASE_DIR, f"input_{date_str}.nc")
            save_dir = os.path.join('/workspace/data/hjhdatasets/guangxi', date_str)

            print(f'\n{"="*60}')
            print(f'Inference date: {date_str}')
            print(f'Input:  {input_file}')
            print(f'Output: {save_dir}')
            print(f'{"="*60}')

            if not os.path.exists(input_file):
                print(f'❌ Input file not found: {input_file}, skipping...')
                current += timedelta(days=1)
                continue

            input = xr.open_dataarray(input_file)
            #print_dataarray(input)
            print(input)

            os.makedirs(save_dir, exist_ok=True)

            #try:
            run_inference(
                model,
                input,
                args.total_step,
                args.total_member,
                save_dir=save_dir
            )
            print(f'✅ {date_str} done!')
            #except Exception as e:
                #print(f'❌ Error processing {date_str}: {e}')

            del input
            current += timedelta(days=1)

        print(f'\nAll done! {start_date.strftime("%Y%m%d")} ~ {end_date.strftime("%Y%m%d")}')

    elif args.input:
        # ====== 单日模式（兼容原有用法）======
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        input = xr.open_dataarray(args.input)
        print_dataarray(input)

        save_dir = args.save_dir if args.save_dir else "output"
        os.makedirs(save_dir, exist_ok=True)

        run_inference(
            model,
            input,
            args.total_step,
            args.total_member,
            save_dir=save_dir
        )
    else:
        parser.error("Either --start/--end (loop mode) or --input (single mode) must be specified.")
 
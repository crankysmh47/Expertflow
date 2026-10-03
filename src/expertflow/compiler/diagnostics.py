"""Read-only diagnostic sensors; mandatory memory state belongs to the runner."""

import ctypes
import math
from ctypes import wintypes


def _gpu_reader(base):
    api, device = base.nvml, base.device
    api.nvmlDeviceGetClockInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    api.nvmlDeviceGetTemperature.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    api.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    class Utilization(ctypes.Structure):
        _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]
    api.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(Utilization)]

    def read(pid):
        result = {'available': True}
        for name, function, args in (
            ('sm_clock_mhz', api.nvmlDeviceGetClockInfo, (device, 1)),
            ('memory_clock_mhz', api.nvmlDeviceGetClockInfo, (device, 2)),
            ('temperature_c', api.nvmlDeviceGetTemperature, (device, 0)),
            ('power_mw', api.nvmlDeviceGetPowerUsage, (device,)),
        ):
            value = ctypes.c_uint()
            status = function(*args, ctypes.byref(value))
            result[name] = value.value if status == 0 else None
            if status:
                result[name + '_unavailable_status'] = status
        utilization = Utilization()
        status = api.nvmlDeviceGetUtilizationRates(device, ctypes.byref(utilization))
        result.update(gpu_utilization_pct=utilization.gpu if status == 0 else None,
                      memory_utilization_pct=utilization.memory if status == 0 else None)
        if status:
            result['utilization_unavailable_status'] = status
        return result
    return read


def _cpu_reader(base):
    counters = {}
    for name, path in (
        ('performance_pct', r'\Processor Information(_Total)\% Processor Performance'),
        ('reported_frequency_mhz', r'\Processor Information(_Total)\Processor Frequency'),
        ('system_utilization_pct', r'\Processor Information(_Total)\% Processor Time'),
    ):
        counter = wintypes.HANDLE()
        if base.pdh.PdhAddEnglishCounterW(base.query, path, 0, ctypes.byref(counter)) == 0:
            counters[name] = counter
    base.pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                    ctypes.c_void_p, ctypes.c_void_p]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]

    def read(pid):
        result = {'available': bool(counters), 'temperature_c': None,
                  'temperature_unavailable_reason': 'no trusted CPU temperature sensor'}
        for name, counter in counters.items():
            item = base.Item()
            status = base.pdh.PdhGetFormattedCounterValue(counter, 0x200, None, ctypes.byref(item.FmtValue))
            result[name] = item.FmtValue.doubleValue if status == 0 and item.FmtValue.CStatus in (0, 1) else None
        handle = kernel.OpenProcess(0x1000, False, pid)
        result['owned_cpu_time_100ns'] = None
        if handle:
            try:
                values = [wintypes.FILETIME() for _ in range(4)]
                if kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
                    result['owned_cpu_time_100ns'] = sum((v.dwHighDateTime << 32) | v.dwLowDateTime for v in values[2:])
            finally:
                kernel.CloseHandle(handle)
        return result
    return read


class DiagnosticSampler:
    def __init__(self, base, *, gpu_reader=None, cpu_reader=None):
        self.base = base
        self.gpu_reader = gpu_reader if gpu_reader is not None else self._configure(_gpu_reader, base)
        self.cpu_reader = cpu_reader if cpu_reader is not None else self._configure(_cpu_reader, base)

    @staticmethod
    def _configure(factory, base):
        try:
            return factory(base)
        except Exception as error:
            reason = str(error)
            return lambda pid: {'available': False, 'reason': reason}

    @staticmethod
    def _read(reader, pid):
        try:
            result = reader(pid)
            if not isinstance(result, dict):
                raise ValueError('diagnostic sensor did not return an object')
            return {key: None if type(value) is float and not math.isfinite(value) else value
                    for key, value in result.items()}
        except Exception as error:
            return {'available': False, 'reason': str(error)}

    def __call__(self, pid):
        memory = self.base(pid)
        if not isinstance(memory, dict):
            return memory
        return {**memory, 'diagnostics': {'gpu': self._read(self.gpu_reader, pid),
                                         'cpu': self._read(self.cpu_reader, pid)}}

    def check_idle(self):
        return self.base.check_idle()

    def close(self):
        self.base.close()

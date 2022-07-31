# import adaptfilt as ad
from timeit import default_timer as timer
import padasip as pa
import pyroomacoustics as pr
import numpy as np

# initialize the filter
# creation of data
N = 1000
filt_len = 30

pa_rls_times = []
room_rls_times = []
pa_lms_times = []
room_lms_times = []

for i in range(100):
    x_raw = np.random.normal(0, 1, N)  # input matrix
    v = np.random.normal(0, 0.1, N - filt_len + 1)  # noise
    x = pa.input_from_history(x_raw, filt_len)
    d = np.sum(x, axis=1) + v
    print(f"processing experiment n. {i}")
    start = timer()
    f = pa.filters.FilterRLS(n=filt_len, mu=0.5, w="random")
    y, e, w = f.run(d, x)
    stop = timer()
    pa_rls_times.append(stop - start)

    start = timer()
    f = pa.filters.FilterNLMS(n=filt_len, mu=0.5, w="random")
    y, e, w = f.run(d, x)
    stop = timer()
    pa_lms_times.append(stop - start)

    start = timer()
    rls = pr.adaptive.RLS(filt_len)
    # run the filter on a stream of samples
    for i in range(N - filt_len):
        rls.update(x_raw[i + filt_len], d[i])
    stop = timer()
    room_rls_times.append(stop - start)

    start = timer()
    nlms = pr.adaptive.NLMS(filt_len)
    # run the filter on a stream of samples
    for i in range(N - filt_len):
        nlms.update(x_raw[i + filt_len], d[i])
    stop = timer()
    room_lms_times.append(stop - start)

print(f"padasip RLS average: {1000 * np.average(np.asarray(pa_rls_times)):.2f} ms")
print(f"pyroom RLS average: {1000 * np.average(np.asarray(room_rls_times)):.2f} ms")
print(f"padasip NLMS average: {1000 * np.average(np.asarray(pa_lms_times)):.2f} ms")
print(f"pyroom NLMS average: {1000 * np.average(np.asarray(room_lms_times)):.2f} ms")
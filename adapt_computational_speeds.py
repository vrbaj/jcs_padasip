# import adaptfilt as ad
from timeit import default_timer as timer
import padasip as pa
import pyroomacoustics as pr
import numpy as np

def measure_x():
    # it produces input vector of size 3
    x = np.random.random(3)
    return x

def measure_d(x):
    # meausure system output
    d = 2*x[0] + 1*x[1] - 1.5*x[2]
    return d

N = 100000
log_d = np.zeros(N)
log_y = np.zeros(N)
filt = pa.filters.FilterNLMS(3, mu=1.)
rls = pr.adaptive.NLMS(3)
room_rls_times = []
pa_rls_times = []

for k in range(N):
    # measure input
    x = measure_x()
    # predict new value
    #y = filt.predict(x)
    # do the important stuff with prediction output
    # measure output
    d = measure_d(x)
    # update filter
    start = timer()
    filt.adapt(d, x)
    stop = timer()
    pa_rls_times.append(stop - start)
    start = timer()
    rls.update(x[0], d)
    stop = timer()
    room_rls_times.append(stop - start)


print(f"padasip RLS average: {1000 * np.average(np.asarray(pa_rls_times)):.5f} ms")
print(f"pyroom RLS average: {1000 * np.average(np.asarray(room_rls_times)):.5f} ms")

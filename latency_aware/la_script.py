from tcp_latency import measure_latency
import statistics
from tcppinglib import tcpping

host = tcpping('localhost', port=5000, interval=1.0)
print(host.avg_rtt)

##latency_array = measure_latency(host='127.0.0.1')
##print(latency_array)

##mean_latency = statistics.mean(latency_array) 
##print(mean_latency)
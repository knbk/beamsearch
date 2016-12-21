from loaders import *

reader = load_meta_data()
customer_url_trace = {}
i = 0
for row in reader.x:
    if i > 5000:
        break
    i += 1
    customer_url_trace.setdefault(row[7], [])
    customer_url_trace[row[7]].append(row[1])
    customer_url_trace[row[7]].append(row[2])
    customer_url_trace[row[7]].append(row[4])
    customer_url_trace[row[7]].append(row[13])
    customer_url_trace[row[7]].append(row[29])

for key in customer_url_trace:
    print(key, customer_url_trace[key])
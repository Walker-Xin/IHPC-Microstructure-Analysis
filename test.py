import numpy as np

image = np.load(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Data extraction\Marker_IHPC.npy')

merged = np.load(r'C:\Users\Xin Wenkang\Documents\Scripts\IPHC\Pics\Data extraction\Marker_IHPC_merged.npy')

def area(image: np.ndarray):
    label, area = np.unique(image, return_counts=True)
    data = list(zip(label, area))
    data = data[2:]
    data = {data[i][0]: data[i][1] for i in range(0, len(data))}
    return data

print(len(area(image)))
print(len(area(merged)))
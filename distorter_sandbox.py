import os
import distortions

# Path
path = os.path.join("data.csv")

# 1. Create the distorter
distorter = distortions.Distorter()

# 2. Read in the data and specify the columns
distorter.read_data(path, ["T1"])

# 3. Apply distortions
#distorter.gaussian_noise(5, count=5)
distorter.offset(5, anomaly_size=0.1, count=1)
#distorter.zero()
#distorter.pure_noise(23, 5)
#distorter.point(5, 5)

# 4. View the data
distorter.view_chart("distorted")

# 5. Output the data to a csv
distorter.distorted_data.to_csv('distorted.csv')

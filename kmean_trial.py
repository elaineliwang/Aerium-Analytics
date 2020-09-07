from yellow_color import Yellow_Color
from Kmeans_cached import KMeans_Cached

y = Yellow_Color();

filename = "Parking Lot/Lot3.tif"

K = 5
kmc = KMeans_Cached(K, filename)
kmc.fit()
kmc.output_image()
kmc.output_mask(y.bgr)

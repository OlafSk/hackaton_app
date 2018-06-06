select
	ST_Y(ST_Transform(ST_Centroid(geometria92), 4326)),
	ST_X(ST_Transform(ST_Centroid(geometria92), 4326))
from locit_sample.locit_datasets.grid250
where eurogrid_0250 = %s
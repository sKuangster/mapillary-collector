"""Fixed properties of the outside world. Not tuning knobs -- those live in config."""

MAPILLARY_ENTITY_URL = "https://graph.mapillary.com/{image_id}"
MAPILLARY_TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"

NATURAL_EARTH_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)

# Everything we ask the graph API for, per image. computed_* are the
# structure-from-motion corrected values: better position labels for a lat/lng
# regression model than raw phone GPS.
MAPILLARY_FIELDS = (
    "id",
    "geometry",
    "computed_geometry",
    "compass_angle",
    "computed_compass_angle",
    "captured_at",
    "is_pano",
    "quality_score",
    "sequence",
    "camera_type",
    "width",
    "height",
    "thumb_1024_url",
    "thumb_2048_url",
)

SHARD_NAME_FMT = "shard-{idx:06d}.tar"

# a densely covered spot (central Berlin) used to probe whether the tile server
# is answering. any tile works; a well-covered one also proves data comes back
TILE_PROBE_LNGLAT = (13.405, 52.52)

WEB_MERCATOR_MAX_LAT = 85.05112878  # spherical mercator singularity guard

# shard lifecycle
SHARD_LOCAL = "local"        # packed on disk, not yet uploaded
SHARD_UPLOADING = "uploading"
SHARD_UPLOADED = "uploaded"

# country lifecycle
COUNTRY_IN_PROGRESS = "in_progress"
COUNTRY_COMPLETED = "completed"    # hit its proportional quota
COUNTRY_EXHAUSTED = "exhausted"    # ran out of coverage before the quota
COUNTRY_FAILED = "failed"          # no usable polygon

# tile lifecycle
TILE_PENDING = "pending"
TILE_FETCHED = "fetched"
TILE_EMPTY = "empty"
TILE_ERROR = "error"
MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"
TILE_LAYER_IMAGE = "image"
TILE_LAYER_SEQUENCE = "sequence"
PARENT_TILE_ZOOM = 10
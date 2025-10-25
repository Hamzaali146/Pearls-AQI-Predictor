import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
fg_raw = fs.get_feature_group(
                name="karachi_air_quality_features",
                version=1
            )

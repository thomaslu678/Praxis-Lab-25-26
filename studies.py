from typing import Literal, TypedDict, override

import ee
import pandas as pd

# Definitions
project_name = "gee-481701"
dataset_path = "LANDSAT/LC08/C02/T1_L2"

# Initial setup

ee.Authenticate()
ee.Initialize(project=project_name)

landsat: ee.ImageCollection = ee.ImageCollection(dataset_path)

# Scale Landsat data to useful measurements
def _scale_landsat(image: ee.Image) -> ee.Image:
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    return image.addBands(optical_bands, None, True).addBands(
        thermal_bands,
        None,
        True
    )

landsat = landsat.map(_scale_landsat)

# Mask pixels with clouds or water, or saturated pixels
def _apply_mask(image: ee.Image) -> ee.Image:
    # Bit 0: Unused
    # Bit 1: Dilated Cloud
    # Bit 2: Cirrus (high confidence)
    # Bit 3: Cloud
    # Bit 4: Cloud Shadow
    # Bit 7: Water
    qa_mask = image.select("QA_PIXEL").bitwiseAnd(int("10011111", 2)).Not()
    saturation_mask = image.select("QA_RADSAT").Not()

    return image.updateMask(qa_mask).updateMask(saturation_mask)

landsat = landsat.map(_apply_mask)

# Only include data from June, July, and August
landsat = landsat.filter(
    ee.Filter.calendarRange(6, 8, "month")
)


class CaseStudy(TypedDict):
    """Collection of information for a singular case study."""

    name: str
    """Name of the case study."""

    start: int
    """Year of construction beginning."""

    end: int
    """Year of construction concluding."""

    outline: ee.Geometry.Polygon
    """Bounds of case study area."""

    points: ee.FeatureCollection
    """Collection of transect samples for case study.
    
    Each feature should be an ee.Point with a distance property.
    """

    station: str
    """WMO URL used to obtain local meterological station data."""


# Primary function used to interact with this module
def getData(study: CaseStudy) -> pd.DataFrame:
    """Obtain relevant Landsat data samples for a specified case study.

    The data is returned as a pandas DataFrame with the following columns:
        study_name
        similarity
        distance
        time
        lst
        ndvi
        station_temp
        station_humidity
    
    study_name and similarity will be identical for all rows of a singular case study.

    Args:
        case_study: The case study to obtain data for.
    Returns: The data for the case study.
    """

    # Validate parameters

    # TODO: Add validation for similarity once similarity score defined

    if (study["start"] > study["end"]):
        raise ValueError(f"{study["name"]}: End year should be after start year")
    
    if (not study["points"].propertyNames().contains("distance")):
        raise ValueError(f"{study["name"]}: Collection of points should have \"distance\" property.")
    
    try:
        study["points"].map(lambda point: ee.Geometry.Point(point.geometry()))
    except:
        raise ValueError(f"{study["name"]}: Points should be convertable to ee.Geometry.Point instances")
    
    if (study["points"].filter(
        lambda point: point.get("distance") 
    ) > 0):
        raise ValueError("All point distances should be non-negative")
    
    # Include data one year before and 5 years after implementation
    start = study["start"] - 1
    end = study["end"] + 6 # End bound is exclusive

    dataset: ee.FeatureCollection = landsat.filterDate(str(start), str(end))
    
    # Map each point to a feature collection containing Landsat data
    def map_point(feature: ee.Feature) -> ee.FeatureCollection:

        point = ee.Geometry.Point(feature.geometry())

        # Only include Landsat images that intersect point
        point_data = dataset.filterBounds(point)

        # Map each image to a feature with relevant data and no geometry
        point_data.map(
            lambda image: ee.Feature(
                None, {
                    "study_name": study["name"],
                    "similarity": study["similarity"],
                    "time": image.get("DATE_PRODUCT_GENERATED"),
                    "distance": feature.get("distance"),
                    "lst": image.select("ST_B10"),
                    "ndvi": image.select("NDVI"),
                    # TODO: look into fetching station data
                    "station_temp": None,
                    "station_humidity": None
                }
            )
        )
    
    # Convert collection of collections to DataFrame
    return pd.DataFrame(
        study["points"].map(map_point).flatten()
    )
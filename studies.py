from typing import TypedDict

import ee
import pandas as pd
import requests

# Initial setup
project_name = "gee-481701"
dataset_path = "LANDSAT/LE07/C02/T1_L2"

ee.Authenticate()
ee.Initialize(project=project_name)

# Combine Landsat collections
ls9: ee.ImageCollection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
ls9 = ls9.select([
    "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL", "QA_RADSAT"
], [
    "blue", "green", "red", "nir", "swir1", "swir2", "lst", "QA_PIXEL", "QA_RADSAT"
])

ls8: ee.ImageCollection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
ls8 = ls8.select([
    "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL", "QA_RADSAT"
], [
    "blue", "green", "red", "nir", "swir1", "swir2", "lst", "QA_PIXEL", "QA_RADSAT"
])

ls7: ee.ImageCollection = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
ls7 = ls7.select([
    "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA_PIXEL", "QA_RADSAT"
], [
    "blue", "green", "red", "nir", "swir1", "swir2", "lst", "QA_PIXEL", "QA_RADSAT"
])

ls5: ee.ImageCollection = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
ls5 = ls5.select([
    "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA_PIXEL", "QA_RADSAT"
], [
    "blue", "green", "red", "nir", "swir1", "swir2", "lst", "QA_PIXEL", "QA_RADSAT"
])

ls4: ee.ImageCollection = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2")
ls4 = ls4.select([
    "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA_PIXEL", "QA_RADSAT"
], [
    "blue", "green", "red", "nir", "swir1", "swir2", "lst", "QA_PIXEL", "QA_RADSAT"
])

landsat: ee.ImageCollection = ls9.merge(ls8).merge(ls7).merge(ls5).merge(ls4)

# Only include data from June, July, and August
landsat = landsat.filter(
    ee.Filter.calendarRange(6, 8, "month")
)

# Scale Landsat data to useful measurements
def _scale_landsat(image: ee.Image) -> ee.Image:
    optical_bands = image.select("blue", "green", "red", "nir", "swir1", "swir2").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("lst").multiply(0.00341802).add(149.0)

    return image.addBands(optical_bands, None, True).addBands(
        thermal_bands,
        None,
        True
    )

landsat = landsat.map(_scale_landsat)


# Add extra bands: time, ndvi
landsat = landsat.map(
    lambda image: image.addBands(
        ee.Image.constant(image.get("system:time_start")).rename("time")
    ).addBands(
        image.normalizedDifference(["nir", "red"]).rename("ndvi")
    )
)

# Mask pixels with clouds or water, or saturated pixels
def _apply_mask(image: ee.Image) -> ee.Image:
    # Bit 0: Fill
    # Bit 1: Dilated Cloud
    # Bit 2: Cirrus (high confidence)
    # Bit 3: Cloud
    # Bit 4: Cloud Shadow
    # Bit 7: Water
    qa_mask = image.select("QA_PIXEL").bitwiseAnd(int("10011111", 2)).eq(0)

    saturation_mask = image.select("QA_RADSAT").eq(0)

    return image.updateMask(qa_mask).updateMask(saturation_mask)

landsat = landsat.map(_apply_mask)

class CaseStudy(TypedDict):
    """Collection of information for a singular case study."""

    name: str
    """Name of the case study."""

    start: int
    """Year of construction beginning."""

    end: int
    """Year of construction concluding."""
    
    points: ee.FeatureCollection
    """URL to GeoJSON file containing samples points of study.

    Each feature should be an ee.Point with a distance property.
    """

    station: str
    """WMO URL used to obtain local meterological station data."""


def _generateData(study: CaseStudy):
    """Generate relevant Landsat data samples for a specified case study.

    The data is returned as a feature collection. Each feature has the following properties:
        study_name
        similarity
        distance
        time
        station_temp
        station_humidity
        blue
        green
        red
        nir
        swir1
        swir2
        lst

    study_name and similarity will be identical for all data for a singular case study.

    Args:
        case_study:
            The case study to obtain data for.
    Returns:
        A feature collection. Each feature represents a singular sample in time and space.
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

    # Include data one year before and 5 years after implementation
    start = study["start"] - 1
    end = study["end"] + 6 # End bound is exclusive

    images: ee.ImageCollection = landsat.filterDate(str(start), str(end))

    # Only include images that intersect at least one sample point
    images = images.filterBounds(study["points"].bounds())

    # For each image, take samples at all intersecting points
    samples = images.map(
        lambda image: image.reduceRegions(
            collection = study["points"],
            reducer = ee.Reducer.first(),
            scale = 30, # meters
        )
    )

    # Convert FeatureCollection to DataFrame
    samples = samples.flatten()

    # Remove samples that didn't have an unmasked pixel intersecting point
    samples = samples.filter(ee.Filter.neq("time", None))

    data = samples.first().propertyNames().iterate(
        lambda curr, prev: ee.Dictionary(prev).set(curr, samples.aggregate_array(curr)),
        ee.Dictionary()
    )

    # Return samples
    return samples
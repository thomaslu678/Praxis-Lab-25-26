import ee

# Initial setup
_landsat: ee.ImageCollection = None


def initialize(project_name: str) -> None:
  """Initialize module.

  Args:
    project_name: Name of Google Earth Engine project to use
  """
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


  # Add extra bands: time
  landsat = landsat.map(
      lambda image: image.addBands(
          ee.Image.constant(image.get("system:time_start")).rename("time")
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

  # Remove QA bands
  landsat = landsat.select(
      landsat.first().bandNames().filter(
          ee.Filter.And(
              ee.Filter.neq("item", "QA_PIXEL"),
              ee.Filter.neq("item", "QA_RADSAT")
          )
      )
  )
  global _landsat
  _landsat = landsat


def _map_study(study: ee.Feature, all_points: ee.FeatureCollection) -> ee.FeatureCollection:
  """Samples Landsat data for a singular case study."""

  # Augment points with study data
  points: ee.FeatureCollection = all_points.filter(ee.Filter.eq("name", study.get("name")))
  points = points.map(
      lambda point: ee.Feature(point).set({
          "start": study.get("start"),
          "end": study.get("end"),
          "similarity": study.get("similarity")
      })
  )

  # Only include Landsat data between 5 years before and after implementation
  images: ee.ImageCollection = _landsat.filterDate(
      ee.String(ee.Number(study.get("start")).add(-5)),
      ee.String(ee.Number(study.get("end")).add(6))
  )

  # Only include images containing a sample point
  images = images.filterBounds(points)

  # For each image, take samples at all points
  samples: ee.FeatureCollection = images.map(
    lambda image: ee.Image(image).reduceRegions(
        collection = points,
        reducer = ee.Reducer.first(), # A point shouldn't intersect multiple pixels
        scale = 30, # meters
    )
  ).flatten()

  # Remove samples on masked pixels
  samples = samples.filter(ee.Filter.neq("time", None))

  return samples

# Main function used to interact with module
def generate_samples(
    studies: ee.FeatureCollection,
    points: ee.FeatureCollection,
    output_path: str) -> ee.batch.Task:
  """Samples Landsat data for provided case studies.

  Args:
    studies:
      A FeatureCollection consisting of outlines for every case study. The following
      properties are expected for each case study:

      * name
      * start
      * end
      * similarity
      * station

    points:
      A FeatureCollection consisting of points for every case study. The following
      properties are expected for each point:

      * study
      * distance

      The "study" property should match the name of a case study.

    output_path:
      The resulting samples will be exported as a Google Earth Engine asset using
      the provided path.
  Returns:
    An unstarted task to upload a Google Earth Engine asset.
    The asset is a FeatureCollection containing a Feature for every sample for
    every point for every case study, with the following properties for every sample:
      * name
      * start
      * end
      * similarity
      * distance
      * time
      * station_temp
      * station_humidity
      * blue
      * green
      * red
      * niw
      * swir1
      * swir2
      * lst
  Raises:
    ValueError: If this function is called without calling "initialize" prior.
  """

  if(_landsat == None):
    raise ValueError('Case study module not initialized. Try calling "initialize".')

  samples: ee.FeatureCollection = studies.map(
      lambda study: _map_study(study, points)
  ).flatten()

  # Return task to export data
  return ee.batch.Export.table.toAsset(
      collection = samples,
      assetId = output_path,
  )
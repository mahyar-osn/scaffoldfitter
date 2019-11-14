import math
import os
import unittest
from opencmiss.zinc.result import RESULT_OK
from scaffoldfitter.scaffit import Scaffit
from scaffoldfitter.fit_step_align import FitStepAlign
from scaffoldfitter.fit_step_fitgeometry import FitStepFitGeometry
from scaffoldfitter.utils.zinc_utils import ZincCacheChanges

here = os.path.abspath(os.path.dirname(__file__))

def createScaffitForCubeToSphere(dataFileName):
    zinc_model_file = os.path.join(here, "resources", "cube_to_sphere.exf")
    zinc_data_file = os.path.join(here, "resources", dataFileName)
    scaffit = Scaffit(zinc_model_file, zinc_data_file)
    scaffit.setModelCoordinatesFieldByName("coordinates")
    scaffit.setDataCoordinatesFieldByName("data_coordinates")
    return scaffit

def assertAlmostEqualList(testcase, actualList, expectedList, delta):
    assert len(actualList) == len(expectedList)
    for actual, expected in zip(actualList, expectedList):
        testcase.assertAlmostEqual(actual, expected, delta=delta)

def scaleList(values, scale):
    return [ value*scale for value in values]

def getRotationMatrix(eulerAngles):
    """
    From OpenCMISS-Zinc graphics_library.cpp, transposed.
    :param eulerAngles: 3-component field of angles in radians, components:
    1 = azimuth (about z)
    2 = elevation (about rotated y)
    3 = roll (about rotated x)
    :return: 9-component rotation matrix varying fastest across, suitable for pre-multiplying [x, y, z].
    """
    cos_azimuth = math.cos(eulerAngles[0])
    sin_azimuth = math.sin(eulerAngles[0])
    cos_elevation = math.cos(eulerAngles[1])
    sin_elevation = math.sin(eulerAngles[1])
    cos_roll = math.cos(eulerAngles[2])
    sin_roll = math.sin(eulerAngles[2])
    return [
        cos_azimuth*cos_elevation,
        cos_azimuth*sin_elevation*sin_roll - sin_azimuth*cos_roll,
        cos_azimuth*sin_elevation*cos_roll + sin_azimuth*sin_roll,
        sin_azimuth*cos_elevation,
        sin_azimuth*sin_elevation*sin_roll + cos_azimuth*cos_roll,
        sin_azimuth*sin_elevation*cos_roll - cos_azimuth*sin_roll,
        -sin_elevation,
        cos_elevation*sin_roll,
        cos_elevation*cos_roll,
        ]

def rigidBodyTransform(transformationMatrix, translation, x):
    """
    :return: x offset by translation then multiplied by 9-component transformationMatrix.
    """
    assert (len(transformationMatrix) == 9) and (len(translation) == 3) and (len(x) == 3)
    x2 = [ (x[c] + translation[c]) for c in range(3) ]
    x3 = []
    for c in range(3):
        v = 0.0
        for d in range(3):
            v += transformationMatrix[c*3 + d]*x2[d]
        x3.append(v)
    return x3

class FitCubeToSphereTestCase(unittest.TestCase):

    def test_alignFixedRandomData(self):
        """
        Test alignment of model and data to known transformations.
        """
        scaffit = createScaffitForCubeToSphere("cube_to_sphere_data_random.exf")
        bottomCentre1 = scaffit.evaluateNodeGroupMeanCoordinates("bottom", "coordinates", isData = False)
        sidesCentre1 = scaffit.evaluateNodeGroupMeanCoordinates("sides", "coordinates", isData = False)
        topCentre1 = scaffit.evaluateNodeGroupMeanCoordinates("top", "coordinates", isData = False)
        assertAlmostEqualList(self, bottomCentre1, [ 0.5, 0.5, 0.0 ], delta=1.0E-7)
        assertAlmostEqualList(self, sidesCentre1, [ 0.5, 0.5, 0.5 ], delta=1.0E-7)
        assertAlmostEqualList(self, topCentre1, [ 0.5, 0.5, 1.0 ], delta=1.0E-7)
        bottomDataCentre1 = scaffit.evaluateNodeGroupMeanCoordinates("bottom", "data_coordinates", isData = True)
        sidesDataCentre1 = scaffit.evaluateNodeGroupMeanCoordinates("sides", "data_coordinates", isData = True)
        topDataCentre1 = scaffit.evaluateNodeGroupMeanCoordinates("top", "data_coordinates", isData = True)
        assertAlmostEqualList(self, bottomDataCentre1, [-0.009562266158420317, -0.0405228759928804, -0.42009164286986966], delta=1.0E-7)
        assertAlmostEqualList(self, sidesDataCentre1, [0.0008220550281536953, -0.043375097752615326, 0.030232577990781825], delta=1.0E-7)
        assertAlmostEqualList(self, topDataCentre1, [0.029623056646825956, 0.05656568887086555, 0.43379576749997456], delta=1.0E-7)
        align = FitStepAlign(scaffit)
        align.setRotation([ math.pi/4.0, math.pi/8.0, math.pi/2.0 ])
        align.setTranslation([ 0.1, 0.2, 0.3 ])
        align.setScale(1.1)
        align.setAlignMarkers(False)
        errorString = align.run()
        self.assertIsNone(errorString, errorString)
        rotation = align.getRotation()
        scale = align.getScale()
        translation = align.getTranslation()
        bottomCentre2 = scaffit.evaluateNodeGroupMeanCoordinates("bottom", "coordinates", isData = False)
        sidesCentre2 = scaffit.evaluateNodeGroupMeanCoordinates("sides", "coordinates", isData = False)
        topCentre2 = scaffit.evaluateNodeGroupMeanCoordinates("top", "coordinates", isData = False)
        assertAlmostEqualList(self, bottomCentre2, scaleList(bottomCentre1, scale), delta=1.0E-7)
        assertAlmostEqualList(self, sidesCentre2, scaleList(sidesCentre1, scale), delta=1.0E-7)
        assertAlmostEqualList(self, topCentre2, scaleList(topCentre1, scale), delta=1.0E-7)
        bottomDataCentre2 = scaffit.evaluateNodeGroupMeanCoordinates("bottom", "data_coordinates", isData = True)
        sidesDataCentre2 = scaffit.evaluateNodeGroupMeanCoordinates("sides", "data_coordinates", isData = True)
        topDataCentre2 = scaffit.evaluateNodeGroupMeanCoordinates("top", "data_coordinates", isData = True)
        rotationMatrix = getRotationMatrix(rotation)
        assertAlmostEqualList(self, bottomDataCentre2, rigidBodyTransform(rotationMatrix, translation, bottomDataCentre1), delta=1.0E-7)
        assertAlmostEqualList(self, sidesDataCentre2, rigidBodyTransform(rotationMatrix, translation, sidesDataCentre1), delta=1.0E-7)
        assertAlmostEqualList(self, topDataCentre2, rigidBodyTransform(rotationMatrix, translation, topDataCentre1), delta=1.0E-7)

    def test_alignMarkersFitRegularData(self):
        """
        Test automatic alignment of model and data using fiducial markers.
        """
        scaffit = createScaffitForCubeToSphere("cube_to_sphere_data_regular.exf")

        scaffit.getRegion().writeFile(os.path.join(here, "resources", "km_fitgeometry0.exf"))
        coordinates = scaffit.getModelCoordinatesField()
        fieldmodule = scaffit.getFieldmodule()
        with ZincCacheChanges(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, scaffit.getMesh(2))
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, scaffit.getMesh(3))
            volumeField.setNumbersOfPoints(3)
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 6.0, delta=1.0E-6)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume, 1.0, delta=1.0E-7)

        align = FitStepAlign(scaffit)
        align.setAlignMarkers(True)
        errorString = align.run()
        self.assertIsNone(errorString, errorString)
        rotation = align.getRotation()
        scale = align.getScale()
        translation = align.getTranslation()
        assertAlmostEqualList(self, rotation, [ 0.25*math.pi, 0.0, 0.0 ], delta=1.0E-4)
        self.assertAlmostEqual(scale, 0.8047378562670332, places=5)
        assertAlmostEqualList(self, translation, [ 0.5690355977219919, 6.62241378307982e-06, 0.4023689282781041 ], delta=1.0E-7)
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 3.8856181038555655, delta=1.0E-6)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume, 0.5211506638615166, delta=1.0E-6)

        fitGeometry0 = FitStepFitGeometry(scaffit)
        fitGeometry0.setNumberOfIterations(0)
        errorString = fitGeometry0.run()
        scaffit.getRegion().writeFile(os.path.join(here, "resources", "km_fitgeometry1.exf"))

        fitGeometry1 = FitStepFitGeometry(scaffit)
        fitGeometry1.setMarkerWeight(1.0)
        fitGeometry1.setCurvaturePenaltyWeight(0.1)
        fitGeometry1.setNumberOfIterations(3)
        fitGeometry1.setUpdateReferenceCoordinates(True)
        errorString = fitGeometry1.run()
        self.assertIsNone(errorString, errorString)
        scaffit.getRegion().writeFile(os.path.join(here, "resources", "km_fitgeometry2.exf"))

        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 3.1892107388163824, delta=1.0E-6)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume, 0.5276197810475188, delta=1.0E-6)


if __name__ == "__main__":
    unittest.main()

"""
Main class for fitting scaffolds.
"""

import json
from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.result import RESULT_OK
from scaffoldfitter.utils.zinc_utils import assignFieldParameters, createFieldClone, evaluateNodesetMeanCoordinates


class Scaffit:

    def __init__(self, zincModelFileName, zincDataFileName):
        self._context = Context("Scaffoldfitter")
        self._region = None
        self._fieldmodule = None
        self._zincModelFileName = zincModelFileName
        self._zincDataFileName = zincDataFileName
        self._modelCoordinatesField = None
        self._modelReferenceCoordinatesField = None
        self._dataCoordinatesField = None
        self._mesh = None
        self._landmarkGroup = None
        self._fitSteps = []
        self._loadModel()

    def _loadModel(self):
        self._region = self._context.createRegion()
        self._fieldmodule = self._region.getFieldmodule()
        result = self._region.readFile(self._zincModelFileName)
        assert result == RESULT_OK, "Failed to load model file" + str(self._zincModelFileName)
        for dimension in range(3, 0, -1):
            self._mesh = self._fieldmodule.findMeshByDimension(dimension)
            if self._mesh.getSize() > 0:
                break
        else:
            assert self._mesh.getSize() > 0, "No elements in model"
        result = self._region.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file" + str(self._zincDataFileName)
        landmarkGroup = self._fieldmodule.findFieldByName("fiducial").castGroup()
        if landmarkGroup.isValid():
            self._landmarkGroup = landmarkGroup

    def _addFitStep(self, fitStep):
        self._fitSteps.append(fitStep)
        print('_addFitStep type', fitStep.getTypeId())

    def getLandmarkGroup(self):
        return self._landmarkGroup

    def getLandmarkDataFields(self):
        """
        Only call if landmarkGroup exists.
        :return: landmarkDataGroup, landmarkDataCoordinates, landmarkDataLabel
        """
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        landmarkPrefix = self._landmarkGroup.getName()
        landmarkDataGroup = self._landmarkGroup.getFieldNodeGroup(datapoints).getNodesetGroup()
        landmarkDataCoordinates = self._fieldmodule.findFieldByName(landmarkPrefix + "_data_coordinates")
        landmarkDataLabel = self._fieldmodule.findFieldByName(landmarkPrefix + "_data_label")
        return landmarkDataGroup, landmarkDataCoordinates, landmarkDataLabel

    def getMesh(self):
        return self._mesh

    def evaluateNodeGroupMeanCoordinates(self, groupName, coordinatesFieldName, isData = False):
        group = self._fieldmodule.findFieldByName(groupName).castGroup()
        assert group.isValid()
        nodeset = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS if isData else Field.DOMAIN_TYPE_NODES)
        nodesetGroup = group.getFieldNodeGroup(nodeset).getNodesetGroup()
        assert nodesetGroup.isValid()
        coordinates = self._fieldmodule.findFieldByName(coordinatesFieldName)
        return evaluateNodesetMeanCoordinates(nodesetGroup, coordinates)

    def getDataCoordinatesField(self):
        return self._dataCoordinatesField

    def setDataCoordinatesField(self, dataCoordinatesField):
        finiteElementField = dataCoordinatesField.castFiniteElement()
        assert finiteElementField.isValid() and (finiteElementField.getNumberOfComponents() == 3)
        self._dataCoordinatesField = finiteElementField

    def setDataCoordinatesFieldByName(self, dataCoordinatesFieldName):
        dataCoordinatesField = self._fieldmodule.findFieldByName(dataCoordinatesFieldName).castFiniteElement()
        self.setDataCoordinatesField(dataCoordinatesField)

    def getModelCoordinatesField(self):
        return self._modelCoordinatesField

    def setModelCoordinatesField(self, modelCoordinatesField):
        finiteElementField = modelCoordinatesField.castFiniteElement()
        assert finiteElementField.isValid() and (finiteElementField.getNumberOfComponents() == 3)
        self._modelCoordinatesField = finiteElementField
        self._modelReferenceCoordinatesField = createFieldClone(self._modelCoordinatesField, "reference_" + self._modelCoordinatesField.getName())

    def setModelCoordinatesFieldByName(self, modelCoordinatesFieldName):
        modelCoordinatesField = self._fieldmodule.findFieldByName(modelCoordinatesFieldName).castFiniteElement()
        self.setModelCoordinatesField(modelCoordinatesField)

    def updateModelReferenceCoordinates(self):
        assignFieldParameters(self._modelReferenceCoordinatesField, self._modelCoordinatesField)

    def writeModel(self, fileName):
        sir = self._region.createStreaminformationRegion()
        sr = sir.createStreamresourceFile(fileName)
        sir.setFieldNames([ self._modelCoordinatesField.getName() ])
        sir.setResourceDomainTypes(sr, Field.DOMAIN_TYPE_NODES)
        self._region.write(sir)

    def writeData(self, fileName):
        sir = self._region.createStreaminformationRegion()
        sr = sir.createStreamresourceFile(fileName)
        sir.setResourceDomainTypes(sr, Field.DOMAIN_TYPE_DATAPOINTS)
        self._region.write(sir)

class FitStep:
    """
    Base class for fitting steps.
    """

    def __init__(self, fitter : Scaffit):
        self._fitter = fitter
        fitter._addFitStep(self)
        self._hasRun = False

    @classmethod
    def getTypeId(cls):
        return 'FitStep'

    def hasRun(self):
        return self._hasRun

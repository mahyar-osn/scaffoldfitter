"""
Main class for fitting scaffolds.
"""

import json
from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.result import RESULT_OK
from scaffoldfitter.utils.zinc_utils import assignFieldParameters, createFieldClone, evaluateNodesetMeanCoordinates, \
    findNodeWithName, getOrCreateFieldFiniteElement, getOrCreateFieldMeshLocation, getUniqueFieldName, ZincCacheChanges


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
        self._mesh = []  # [dimension - 1]
        self._dataProjectionMeshLocationField = [ ]  # [dimension - 1]
        self._dataProjectionNodeGroupField = []  # [dimension - 1]
        self._dataProjectionNodesetGroup = []  # [dimension - 1]
        self._dataProjectionDirectionField = None  # for storing original projection direction unit vector
        self._markerGroup = None
        self._markerDataLocationField = None
        self._markerDataLocationNodeGroupField = None
        self._markerDataLocationNodesetGroup = None
        self._fitSteps = []
        self._loadModel()

    def _loadModel(self):
        self._region = self._context.createRegion()
        self._fieldmodule = self._region.getFieldmodule()
        result = self._region.readFile(self._zincModelFileName)
        assert result == RESULT_OK, "Failed to load model file" + str(self._zincModelFileName)
        result = self._region.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file" + str(self._zincDataFileName)
        markerGroup = self._fieldmodule.findFieldByName("marker").castGroup()
        self._mesh = [ self._fieldmodule.findMeshByDimension(d + 1) for d in range(3) ]
        if markerGroup.isValid():
            self._markerGroup = markerGroup
            self._calculateMarkerDataLocations()
        with ZincCacheChanges(self._fieldmodule):
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            for d in range(2):
                mesh = self._mesh[d]
                self._dataProjectionMeshLocationField.append(getOrCreateFieldMeshLocation(self._fieldmodule, mesh, namePrefix = "data_projection_location_"))
                field = self._fieldmodule.createFieldNodeGroup(datapoints)
                field.setName(getUniqueFieldName(self._fieldmodule, "data_projection_" + mesh.getName()))
                self._dataProjectionNodeGroupField.append(field)
                self._dataProjectionNodesetGroup.append(field.getNodesetGroup())
            self._dataProjectionDirectionField = getOrCreateFieldFiniteElement(self._fieldmodule, "data_projection_direction", 3, [ "x", "y", "z" ])

    def _addFitStep(self, fitStep):
        self._fitSteps.append(fitStep)
        print('_addFitStep type', fitStep.getTypeId())

    def _calculateMarkerDataLocations(self):
        """
        Called when markerGroup exists.
        Find matching marker mesh locations for marker data points.
        Only finds matching location where there is one datapoint and one node
        for each name in marker group.
        Adds those that are found into _markerDataLocationNodesetGroup.
        """
        markerDataGroup, markerDataCoordinates, markerDataName = self.getMarkerDataFields()
        markerNodeGroup, markerLocation, markerName = self.getMarkerModelFields()
        # assume marker locations are in highest dimension mesh (GRC can't yet query host mesh for markerLocation)
        mesh = self.getHighestDimensionMesh()
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        meshDimension = mesh.getDimension()
        fieldcache = self._fieldmodule.createFieldcache()
        with ZincCacheChanges(self._fieldmodule):
            self._markerDataLocationField = getOrCreateFieldMeshLocation(self._fieldmodule, mesh, "marker_data_location_")
            self._markerDataLocationNodeGroupField = self._fieldmodule.createFieldNodeGroup(datapoints)
            self._markerDataLocationNodeGroupField.setName(getUniqueFieldName(self._fieldmodule, "marker_data_location_group"))
            self._markerDataLocationNodesetGroup = self._markerDataLocationNodeGroupField.getNodesetGroup()
            nodetemplate = markerDataGroup.createNodetemplate()
            nodetemplate.defineField(self._markerDataLocationField)
            datapointIter = markerDataGroup.createNodeiterator()
            datapoint = datapointIter.next()
            while datapoint.isValid():
                fieldcache.setNode(datapoint)
                name = markerDataName.evaluateString(fieldcache)
                # if this is the only datapoint with name:
                if name and findNodeWithName(markerDataGroup, markerDataName, name):
                    node = findNodeWithName(markerNodeGroup, markerName, name)
                    if node:
                        fieldcache.setNode(node)
                        element, xi = markerLocation.evaluateMeshLocation(fieldcache, meshDimension)
                        if element.isValid():
                            datapoint.merge(nodetemplate)
                            self._markerDataLocationNodesetGroup.addNode(datapoint)
                            fieldcache.setNode(datapoint)
                            self._markerDataLocationField.assignMeshLocation(fieldcache, element, xi)
                datapoint = datapointIter.next()
        # Warn about datapoints without a location in model
        markerDataGroupSize = markerDataGroup.getSize()
        markerDataLocationGroupSize = self._markerDataLocationNodesetGroup.getSize()
        markerNodeGroupSize = markerNodeGroup.getSize()
        if markerDataLocationGroupSize < markerDataGroupSize:
            print("Warning: Only " + str(markerDataLocationGroupSize) + " of " + str(markerDataGroupSize) + " marker data points have model locations")
        if markerDataLocationGroupSize < markerNodeGroupSize:
            print("Warning: Only " + str(markerDataLocationGroupSize) + " of " + str(markerNodeGroupSize) + " marker model locations used")

    def getMarkerGroup(self):
        return self._markerGroup

    def getMarkerDataFields(self):
        """
        Only call if markerGroup exists.
        :return: markerDataGroup, markerDataCoordinates, markerDataName
        """
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        markerPrefix = self._markerGroup.getName()
        markerDataGroup = self._markerGroup.getFieldNodeGroup(datapoints).getNodesetGroup()
        markerDataCoordinates = self._fieldmodule.findFieldByName(markerPrefix + "_data_coordinates")
        markerDataName = self._fieldmodule.findFieldByName(markerPrefix + "_data_name")
        return markerDataGroup, markerDataCoordinates, markerDataName

    def getMarkerModelFields(self):
        """
        Only call if markerGroup exists.
        :return: markerNodeGroup, markerLocation, markerName
        """
        nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerPrefix = self._markerGroup.getName()
        markerNodeGroup = self._markerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        markerLocation = self._fieldmodule.findFieldByName(markerPrefix + "_location")
        markerName = self._fieldmodule.findFieldByName(markerPrefix + "_name")
        return markerNodeGroup, markerLocation, markerName

    def clearDataProjectionNodesetGroups(self):
        with ZincCacheChanges(self._fieldmodule):
            for d in range(2):
                self._dataProjectionNodesetGroup[d].removeAllNodes()

    def getDataProjectionDirectionField(self):
        return self._dataProjectionDirectionField

    def getDataProjectionNodeGroupField(self, dimension):
        assert 1 <= dimension <= 2
        return self._dataProjectionNodeGroupField[dimension - 1]

    def getDataProjectionNodesetGroup(self, dimension):
        assert 1 <= dimension <= 2
        return self._dataProjectionNodesetGroup[dimension - 1]

    def getDataProjectionMeshLocationField(self, dimension):
        assert 1 <= dimension <= 2
        return self._dataProjectionMeshLocationField[dimension - 1]

    def getMarkerDataLocationNodesetGroup(self):
        return self._markerDataLocationNodesetGroup

    def getMarkerDataLocationField(self):
        return self._markerDataLocationField

    def getRegion(self):
        return self._region

    def getFieldmodule(self):
        return self._fieldmodule

    def getMesh(self, dimension):
        assert 1 <= dimension <= 3
        return self._mesh[dimension - 1]

    def getHighestDimensionMesh(self):
        """
        :return: Highest dimension mesh with elements in it, or None if none.
        """
        for d in range(2, -1, -1):
            mesh = self._mesh[d]
            if mesh.getSize() > 0:
                return mesh
        return None

    def evaluateNodeGroupMeanCoordinates(self, groupName, coordinatesFieldName, isData = False):
        group = self._fieldmodule.findFieldByName(groupName).castGroup()
        assert group.isValid()
        nodeset = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS if isData else Field.DOMAIN_TYPE_NODES)
        nodesetGroup = group.getFieldNodeGroup(nodeset).getNodesetGroup()
        assert nodesetGroup.isValid()
        coordinates = self._fieldmodule.findFieldByName(coordinatesFieldName)
        return evaluateNodesetMeanCoordinates(coordinates, nodesetGroup)

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

    def getModelReferenceCoordinatesField(self):
        return self._modelReferenceCoordinatesField

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
